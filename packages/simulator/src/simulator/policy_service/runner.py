"""Loads a LeRobot policy and turns simulator observations into actions (host side, needs lerobot + torch).

This mirrors ``LeRobotSyncPolicy`` in ``scripts/rollout.py`` for the Franka tasks (the same lerobot calls in the same
order), but takes plain numpy arrays instead of Isaac Lab tensors so it can run in an environment without Isaac Lab.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simulator import FRANKA_JOINT_NAMES

FRANKA_ACTION_DIM = 8


class LeRobotPolicyRunner:
    def __init__(self, policy_type: str, checkpoint: str, device: str = "cuda"):
        # Imported here so that the protocol / server modules stay importable without lerobot.
        import torch
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self._torch = torch
        self.device = device
        print(f"[policy-server] loading '{policy_type}' policy from {checkpoint} ...", flush=True)
        policy_class = get_policy_class(policy_type)
        self.policy = policy_class.from_pretrained(checkpoint, local_files_only=True)
        self.policy.to(device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = self._make_processors(make_pre_post_processors, checkpoint)
        print("[policy-server] policy is ready.", flush=True)

        self.camera_infos: dict[str, tuple[int, int]] = {}
        self.lerobot_features: dict[str, dict] = {}

    def _make_processors(self, make_pre_post_processors, checkpoint: str):
        device_override = {"device": self.device}
        # The processor pipelines of different policies have different steps; an override for a step that does not
        # exist is an error, so fall back to fewer overrides.
        attempts = [
            (
                {"device_processor": device_override, "rename_observations_processor": {"rename_map": {}}},
                {"device_processor": device_override},
            ),
            ({"device_processor": device_override}, {"device_processor": device_override}),
            (None, None),
        ]
        last_error: Exception | None = None
        for pre_overrides, post_overrides in attempts:
            try:
                return make_pre_post_processors(
                    self.policy.config,
                    pretrained_path=checkpoint,
                    preprocessor_overrides=pre_overrides,
                    postprocessor_overrides=post_overrides,
                )
            except Exception as error:  # noqa: BLE001
                last_error = error
                print(f"[policy-server] processor overrides {list(pre_overrides or {})} rejected: {error!r}", flush=True)
        raise RuntimeError("could not build the pre/post processors") from last_error

    def setup(self, camera_infos: dict[str, tuple[int, int]], task_type: str, actions_per_chunk: int) -> None:
        if task_type != "franka_panda":
            raise ValueError(f"task type {task_type!r} is not supported by the policy server (franka_panda only)")
        self.camera_infos = dict(camera_infos)
        features: dict[str, dict] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(FRANKA_JOINT_NAMES),),
                "names": [f"{name}.pos" for name in FRANKA_JOINT_NAMES],
            }
        }
        for key, (height, width) in camera_infos.items():
            features[f"observation.images.{key}"] = {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channels"],
            }
        self.lerobot_features = features
        self.reset()

    def reset(self) -> None:
        policy_reset = getattr(self.policy, "reset", None)
        if callable(policy_reset):
            policy_reset()

    def _collate(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        if len(observations) == 1:
            return observations[0]
        collated: dict[str, Any] = {}
        for key in observations[0]:
            values = [observation[key] for observation in observations]
            if isinstance(values[0], self._torch.Tensor):
                collated[key] = self._torch.cat(values, dim=0)
            elif isinstance(values[0], list):
                collated[key] = [item for value in values for item in value]
            else:
                collated[key] = values
        return collated

    def act(self, images: dict[str, np.ndarray], joint_pos: np.ndarray, task: str) -> np.ndarray:
        from lerobot.async_inference.helpers import raw_observation_to_observation

        if not self.lerobot_features:
            raise RuntimeError("setup() must be called before act()")
        prepared = []
        for env_index in range(joint_pos.shape[0]):
            raw_observation: dict[str, Any] = {key: images[key][env_index] for key in self.camera_infos}
            raw_observation["task"] = task
            for joint_index, name in enumerate(FRANKA_JOINT_NAMES):
                raw_observation[f"{name}.pos"] = float(joint_pos[env_index, joint_index])
            observation = raw_observation_to_observation(
                raw_observation, self.lerobot_features, self.policy.config.image_features
            )
            prepared.append(self.preprocessor(observation))
        batch = self._collate(prepared)
        with self._torch.inference_mode():
            action = self.policy.select_action(batch)
        actions = self.postprocessor(action).to("cpu").numpy().astype(np.float32)
        if actions.shape[-1] != FRANKA_ACTION_DIM:
            raise ValueError(f"expected {FRANKA_ACTION_DIM} action values, got {actions.shape[-1]}")
        return actions
