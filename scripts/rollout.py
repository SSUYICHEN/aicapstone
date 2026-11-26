# Synchronous LeRobot rollout script for LeIsaac.
# Derived partially from upstream LeIsaac
# `scripts/evaluation/policy_inference.py`
# (https://github.com/LightwheelAI/leisaac/blob/main/scripts/evaluation/policy_inference.py
# @ SHA 6b933e80786a69eb27d47503d11725c9c846566e), trimmed to local LeRobot
# inference and extended with a dual-viewport setup, a debug shape printer,
# and an in-process LeRobotSyncPolicy. Entry point lives at the top of
# `scripts/` (NOT under `scripts/evaluation/`) per AUT-81.

"""Run local LeRobot policy inference in the same process as Isaac Sim."""

"""Launch Isaac Sim Simulator first."""
import json as _json
import multiprocessing
from pathlib import Path as _Path


# Fields that newer LeRobot adds at training time but the inference-side
# LeRobot installed in the worker image doesn't accept. They're all
# training-only (LoRA, torch.compile, image-preproc) and safe to strip
# from the checkpoint's config.json before from_pretrained() reads it.
# Extend whenever draccus.utils.DecodingError surfaces a new field.
_LEROBOT_INCOMPAT_CONFIG_FIELDS: tuple[str, ...] = (
    "use_peft",
    "resize_shape",
    "crop_ratio",
    "compile_model",
    "compile_mode",
)


def _patch_lerobot_config(checkpoint_dir: str) -> None:
    """Strip known-incompatible fields from <checkpoint>/config.json.

    Idempotent — running twice is fine. Errors are swallowed; if config.json
    is missing or unreadable the original from_pretrained call will still
    surface a helpful message.
    """
    cfg_path = _Path(checkpoint_dir) / "config.json"
    if not cfg_path.is_file():
        return
    try:
        with cfg_path.open("r") as f:
            cfg = _json.load(f)
    except (OSError, ValueError) as exc:
        print(f"[rollout] config.json read skipped: {exc}", flush=True)
        return
    stripped = [k for k in _LEROBOT_INCOMPAT_CONFIG_FIELDS if k in cfg]
    if not stripped:
        return
    for k in stripped:
        cfg.pop(k, None)
    try:
        with cfg_path.open("w") as f:
            _json.dump(cfg, f, indent=2)
        print(
            f"[rollout] stripped LeRobot-incompatible config fields: {stripped}",
            flush=True,
        )
    except OSError as exc:
        print(f"[rollout] config.json patch skipped: {exc}", flush=True)




if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Synchronous LeRobot inference for LeIsaac simulation."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--step_hz", type=int, default=60, help="Environment stepping rate in Hz."
)
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument(
    "--episode_length_s", type=float, default=60.0, help="Episode length in seconds."
)
parser.add_argument(
    "--eval_rounds",
    type=int,
    default=0,
    help=(
        "Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual"
        " reset."
    ),
)
parser.add_argument(
    "--policy_type",
    type=str,
    default="lerobot-smolvla",
    help="Local LeRobot policy type. Use lerobot-, for example lerobot-smolvla.",
)
parser.add_argument(
    "--policy_action_horizon",
    type=int,
    default=16,
    help="Number of actions to execute per policy call.",
)
parser.add_argument(
    "--policy_language_instruction",
    type=str,
    default=None,
    help="Language instruction of the policy.",
)
parser.add_argument(
    "--policy_checkpoint_path",
    type=str,
    default=None,
    help="Path to the local LeRobot checkpoint (required unless --policy_server is given).",
)
parser.add_argument(
    "--policy_server",
    type=str,
    default=None,
    help=(
        "HOST:PORT of a running scripts/policy_server.py. When set, the policy runs in that process (typically on"
        " the host, in the environment it was trained in) instead of inside this container. Needed for policies whose"
        " dependencies cannot be installed next to Isaac Lab, e.g. GR00T."
    ),
)
parser.add_argument(
    "--debug_policy_shapes",
    action="store_true",
    help="Print observation and action tensor shapes around each local LeRobot inference call.",
)
parser.add_argument(
    "--save_video_dir",
    type=str,
    default=None,
    help=(
        "If set, save the front/wrist camera observations of every evaluated episode as mp4 files in this directory"
        " (file name: ep<idx>_<camera>_<success|fail>.mp4). Off by default."
    ),
)
parser.add_argument(
    "--video_stride",
    type=int,
    default=2,
    help="Keep every N-th environment step when saving videos (reduces memory use). Default: 2.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help=(
        "Number of environments to simulate in parallel. Each round evaluates this many"
        " episodes concurrently; the policy is batched across envs and reset between rounds."
    ),
)

AppLauncher.add_app_launcher_args(parser)
# Isaac Lab 3.0's launcher no longer defines --enable_cameras. It picks the Kit experience (with or without the RTX
# renderer) from this setting, and only auto-detects cameras for the new launch_simulation(cfg, ...) API, not for the
# deprecated AppLauncher used here. All HCIS tasks use cameras, so always enable them; the flag is still accepted.
try:
    parser.add_argument(
        "--enable_cameras",
        action="store_true",
        default=True,
        help="Enable camera rendering (always on: all HCIS tasks use cameras). Kept for backward compatibility.",
    )
except argparse.ArgumentError:
    pass
args_cli = parser.parse_args()
if not args_cli.policy_server and not args_cli.policy_checkpoint_path:
    parser.error("--policy_checkpoint_path is required unless --policy_server is given")

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import time
from typing import Any

import omni.ui as ui
import omni.kit.app
import omni.kit.viewport.utility as vp_util

import carb
import gymnasium as gym
import numpy as np
import omni
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import Camera
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.async_inference.helpers import raw_observation_to_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES

from leisaac.utils.env_utils import (
    dynamic_reset_gripper_effort_limit_sim,
    get_task_type,
)
from leisaac.utils.robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)

import leisaac  # noqa: F401
import simulator.tasks  # noqa: F401
from simulator.tasks.external import resolve_task
from simulator import FRANKA_JOINT_NAMES


def setup_dual_viewports():
    """Setup dual viewports: main perspective view and GoPro camera view."""
    perspective_path = "/World/envs/env_0/Robot/panda_hand/wrist"

    # Get main viewport window
    v1_window = ui.Workspace.get_window("Viewport")
    if not v1_window:
        print("Error: Main viewport window not found")
        return

    v1_api = vp_util.get_viewport_from_window_name("Viewport")
    if v1_api:
        v1_api.camera_path = perspective_path

    # Get or create secondary viewport window
    v2_window = ui.Workspace.get_window("Viewport 2")
    if not v2_window:
        v2_window = vp_util.create_viewport_window("Viewport 2")
        # Important: Wait for UI to register the new window
        omni.kit.app.get_app().update()  # Synchronous frame update

    v2_api = vp_util.get_viewport_from_window_name("Viewport 2")
    if v2_api:
        v2_api.camera_path = "/World/envs/env_0/front_camera"

    # Ensure both windows exist before docking
    if v1_window and v2_window:
        # Wait for UI to stabilize before docking
        omni.kit.app.get_app().update()

        # Attempt docking with error handling
        try:
            v2_window.dock_in(v1_window, ui.DockPosition.RIGHT)
            print("Viewports docked: [Viewport (Persp)] | [Viewport 2 (Camera)]")
        except Exception as e:
            print(f"Docking failed: {str(e)}")
            # Alternative docking approach if direct docking fails
            try:
                # Try docking after another frame
                omni.kit.app.get_app().update()
                v2_window.dock_in(v1_window, ui.DockPosition.RIGHT)
                print("Viewports docked on second attempt")
            except Exception as e2:
                print(f"Second docking attempt failed: {str(e2)}")
    else:
        print("Error: Could not find one or both viewport windows for docking.")


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class Controller:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self.reset_state = False

    def __del__(self):
        if (
            hasattr(self, "_input")
            and hasattr(self, "_keyboard")
            and hasattr(self, "_keyboard_sub")
        ):
            self._input.unsubscribe_from_keyboard_events(
                self._keyboard, self._keyboard_sub
            )
            self._keyboard_sub = None

    def reset(self):
        self.reset_state = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_state = True
        return True


def _shape_summary(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    return type(value).__name__


def _print_mapping_shapes(title: str, values: dict[str, Any]) -> None:
    print(title)
    for key in sorted(values):
        print(f"  {key}: {_shape_summary(values[key])}")


class LeRobotSyncPolicy:
    """Local LeRobot inference path matching the async server pipeline."""

    def __init__(
        self,
        policy_type: str,
        pretrained_name_or_path: str,
        task_type: str,
        camera_infos: dict[str, tuple[int, int]],
        actions_per_chunk: int,
        device: str,
        debug_policy_shapes: bool = False,
    ):
        if actions_per_chunk <= 0:
            raise ValueError(
                f"policy_action_horizon must be positive, got {actions_per_chunk}."
            )

        self.task_type = task_type
        self.actions_per_chunk = actions_per_chunk
        self.device = device
        self.debug_policy_shapes = debug_policy_shapes

        if task_type == "so101leader":
            self.state_joint_names = SINGLE_ARM_JOINT_NAMES
            self.action_dim = len(SINGLE_ARM_JOINT_NAMES)
        elif task_type == "franka_panda":
            self.state_joint_names = FRANKA_JOINT_NAMES
            self.action_dim = 8
        else:
            raise ValueError(
                f"Task type {task_type} not supported for synchronous LeRobot inference yet."
            )

        self.lerobot_features = self._build_lerobot_features(camera_infos)
        self.camera_keys = list(camera_infos.keys())

        print(
            f"Loading local LeRobot policy '{policy_type}' from {pretrained_name_or_path}...",
            flush=True,
        )
        # Strip training-only fields that newer LeRobot adds but the
        # inference-side LeRobot doesn't accept. Safe because these flags
        # never affect inference. See _patch_lerobot_config above.
        _patch_lerobot_config(pretrained_name_or_path)
        policy_class = get_policy_class(policy_type)
        self.policy = policy_class.from_pretrained(pretrained_name_or_path, local_files_only=True)
        self.policy.to(device)
        self.policy.eval()

        device_override = {"device": device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": {}},
            },
            postprocessor_overrides={"device_processor": device_override},
        )
        print("Local LeRobot policy is ready.", flush=True)

    def reset(self):
        policy_reset = getattr(self.policy, "reset", None)
        if callable(policy_reset):
            policy_reset()

    def _build_lerobot_features(
        self, camera_infos: dict[str, tuple[int, int]]
    ) -> dict[str, dict]:
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.state_joint_names),),
                "names": [f"{joint_name}.pos" for joint_name in self.state_joint_names],
            }
        }
        for camera_key, camera_image_shape in camera_infos.items():
            features[f"observation.images.{camera_key}"] = {
                "dtype": "image",
                "shape": (camera_image_shape[0], camera_image_shape[1], 3),
                "names": ["height", "width", "channels"],
            }
        return features

    def _build_raw_observation(
        self, observation_dict: dict, env_index: int
    ) -> dict[str, Any]:
        raw_observation = {
            key: observation_dict[key].cpu().numpy().astype(np.uint8)[env_index]
            for key in self.camera_keys
        }
        raw_observation["task"] = observation_dict["task_description"]

        if self.task_type == "so101leader":
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
        elif self.task_type == "franka_panda":
            joint_pos = observation_dict["joint_pos"].cpu().numpy()
        else:
            raise ValueError(
                f"Task type {self.task_type} not supported for synchronous LeRobot inference yet."
            )

        for joint_index, joint_name in enumerate(self.state_joint_names):
            raw_observation[f"{joint_name}.pos"] = joint_pos[env_index, joint_index].item()

        return raw_observation

    def _collate_observations(self, obs_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Stack per-env prepared observations into a single batched observation.

        Each entry in ``obs_list`` carries a leading batch dim of 1 (one env),
        so tensors concatenate along dim 0 into a (num_envs, ...) batch that the
        policy's ``select_action`` consumes in one call — keeping its internal
        action/observation queues batch-consistent across envs.
        """
        if len(obs_list) == 1:
            return obs_list[0]
        collated: dict[str, Any] = {}
        for key in obs_list[0]:
            values = [obs[key] for obs in obs_list]
            first = values[0]
            if isinstance(first, torch.Tensor):
                collated[key] = torch.cat(values, dim=0)
            elif isinstance(first, list):
                collated[key] = [item for value in values for item in value]
            else:
                collated[key] = values
        return collated

    def _config_horizon_summary(self) -> str:
        names = ["chunk_size", "n_action_steps", "action_chunk_size", "action_horizon"]
        values = []
        for name in names:
            if hasattr(self.policy.config, name):
                values.append(f"{name}={getattr(self.policy.config, name)}")
        return ", ".join(values) if values else "no known horizon fields found"

    def _prepare_observation(self, raw_observation: dict[str, Any]) -> dict[str, Any]:
        observation = raw_observation_to_observation(
            raw_observation,
            self.lerobot_features,
            self.policy.config.image_features,
        )
        if self.debug_policy_shapes:
            _print_mapping_shapes("[SyncPolicy] Prepared observation:", observation)

        observation = self.preprocessor(observation)
        if self.debug_policy_shapes:
            _print_mapping_shapes("[SyncPolicy] Preprocessed observation:", observation)
        return observation

    def _predict_lerobot_actions(self, observation: dict[str, Any]) -> torch.Tensor:
        with torch.inference_mode():
            action = self.policy.select_action(observation)
        return self.postprocessor(action)

    def _convert_actions_to_leisaac(self, action_tensor: torch.Tensor) -> np.ndarray:
        if self.task_type == "so101leader":
            actions = convert_lerobot_action_to_leisaac(action_tensor)
        elif self.task_type == "franka_panda":
            actions = action_tensor.to("cpu").numpy()
        else:
            raise ValueError(
                f"Task type {self.task_type} not supported for synchronous LeRobot inference yet."
            )

        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected {self.action_dim} action values for task type {self.task_type}, got {actions.shape[-1]}."
            )
        return actions

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        num_envs = observation_dict["joint_pos"].shape[0]
        obs_batch = []
        for env_index in range(num_envs):
            raw_observation = self._build_raw_observation(observation_dict, env_index)
            if self.debug_policy_shapes and env_index == 0:
                _print_mapping_shapes("[SyncPolicy] Raw observation (env 0):", raw_observation)
            obs_batch.append(self._prepare_observation(raw_observation))

        observation = self._collate_observations(obs_batch)
        action_tensor = self._predict_lerobot_actions(observation)
        actions = self._convert_actions_to_leisaac(action_tensor)
        # actions: (num_envs, action_dim) — one action per env for this step.
        return torch.from_numpy(actions)


class RemoteLeRobotPolicy:
    """Policy that runs in another process (scripts/policy_server.py). Same interface as LeRobotSyncPolicy."""

    def __init__(
        self,
        address: str,
        task_type: str,
        camera_infos: dict[str, tuple[int, int]],
        actions_per_chunk: int,
    ):
        from simulator.policy_service.client import RemotePolicyClient

        host, _, port = address.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"--policy_server must look like HOST:PORT, got '{address}'")
        print(f"[rollout] connecting to policy server {address} ...", flush=True)
        self.camera_keys = list(camera_infos.keys())
        self.client = RemotePolicyClient(host, int(port))
        self.client.setup(camera_infos, task_type, actions_per_chunk)
        print("[rollout] remote policy is ready.", flush=True)

    def reset(self):
        self.client.reset()

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        images = {key: observation_dict[key].cpu().numpy().astype(np.uint8) for key in self.camera_keys}
        joint_pos = observation_dict["joint_pos"].cpu().numpy()
        actions = self.client.act(images, joint_pos, observation_dict["task_description"])
        # actions: (num_envs, action_dim) - one action per env for this step.
        return torch.from_numpy(actions)


def preprocess_obs_dict(obs_dict: dict, language_instruction: str):
    obs_dict["task_description"] = language_instruction
    return obs_dict


def get_policy_type(policy_type_arg: str) -> str:
    if not policy_type_arg.startswith("lerobot-"):
        raise ValueError(
            f"policy_inference_sync.py only supports local LeRobot policies, got '{policy_type_arg}'. "
            "Use --policy_type=lerobot-."
        )
    return policy_type_arg.split("lerobot-", 1)[1]


def get_camera_infos(
    env: ManagerBasedRLEnv, policy_obs_dict: dict
) -> dict[str, tuple[int, int]]:
    camera_infos = {}
    for key, sensor in env.scene.sensors.items():
        if isinstance(sensor, Camera) and key in policy_obs_dict:
            camera_infos[key] = sensor.image_shape
    return camera_infos


def _write_episode_video(frames: list, path: str, fps: int) -> None:
    """Encode a list of (H, W, 3) uint8 RGB frames as an mp4 file (opt-in debugging aid, see --save_video_dir)."""
    import av

    def _encode(codec: str) -> None:
        with av.open(path, "w") as container:
            stream = container.add_stream(codec, rate=fps)
            stream.height, stream.width, stream.pix_fmt = frames[0].shape[0], frames[0].shape[1], "yuv420p"
            for frame in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    try:
        _encode("libx264")
    except Exception:  # noqa: BLE001 - fall back to a codec that is always available in PyAV
        _encode("mpeg4")


def main():
    task_id = resolve_task(args_cli.task)
    args_cli.task = task_id
    env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    task_type = get_task_type(task_id)
    robot_name = getattr(env_cfg, "robot_name", None)
    policy_task_type = "franka_panda" if robot_name == "franka_panda" else task_type
    teleop_device = "keyboard" if policy_task_type == "franka_panda" else task_type
    env_cfg.use_teleop_device(teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s

    if args_cli.eval_rounds <= 0:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    max_episode_count = args_cli.eval_rounds
    env_cfg.recorders = None

    env: ManagerBasedRLEnv = gym.make(task_id, cfg=env_cfg).unwrapped

    # Warm up the renderer before the first reset. Headless Isaac Sim with
    # camera observations otherwise hangs the first env.reset() while the
    # Vulkan / DLSS / shader pipeline compiles — the worker sees no output
    # for several minutes and the eval looks dead. A handful of app updates
    # forces shader compilation and material warm-up to happen here, where
    # we can attribute it.
    print("[rollout] warming up renderer (20 app updates)...", flush=True)
    for _ in range(20):
        simulation_app.update()
    print("[rollout] resetting environment...", flush=True)
    # Every env interaction -- reset included -- must run under inference_mode.
    # IsaacLab's asset data buffers (e.g. RigidObjectData.root_link_pose_w) are
    # lazily refilled from PhysX on first access after a sim step, so whichever
    # mode the *step* runs in decides whether those cached tensors are inference
    # tensors. A reset outside inference_mode then does an in-place write
    # (`self._data.root_link_pose_w[env_ids] = ...`) on an inference tensor and
    # raises "Inplace update to inference tensor outside InferenceMode".
    with torch.inference_mode():
        obs_dict, _ = env.reset()
    print("[rollout] env.reset() returned", flush=True)

    language_instruction = args_cli.policy_language_instruction
    if language_instruction is None:
        language_instruction = getattr(env_cfg, "task_description", None)

    policy_obs_dict = preprocess_obs_dict(obs_dict["policy"], language_instruction)
    camera_infos = get_camera_infos(env, policy_obs_dict)
    print(
        f"[rollout] camera_infos = {camera_infos}; loading policy...",
        flush=True,
    )

    if args_cli.policy_server:
        policy = RemoteLeRobotPolicy(
            address=args_cli.policy_server,
            task_type=policy_task_type,
            camera_infos=camera_infos,
            actions_per_chunk=args_cli.policy_action_horizon,
        )
    else:
        policy = LeRobotSyncPolicy(
            policy_type=get_policy_type(args_cli.policy_type),
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            task_type=policy_task_type,
            camera_infos=camera_infos,
            actions_per_chunk=args_cli.policy_action_horizon,
            device=args_cli.device,
            debug_policy_shapes=args_cli.debug_policy_shapes,
        )

    rate_limiter = RateLimiter(args_cli.step_hz)
    controller = Controller()
    controller.reset()

    setup_dual_viewports()


    # Parallel evaluation runs `num_envs` episodes per synchronized round. All
    # envs reset together, step in lockstep on a batched policy call, and the
    # policy is reset at each round boundary so its internal queues never mix
    # state across episodes. IsaacLab auto-resets an env once it terminates, so
    # we latch each env's FIRST outcome (success vs. time out) via `done_mask`
    # and ignore any further activity until the round ends.
    num_envs = env.num_envs
    success_count, episodes_done = 0, 0
    video_cameras = ("front", "wrist")
    if args_cli.save_video_dir:
        _Path(args_cli.save_video_dir).mkdir(parents=True, exist_ok=True)
        video_fps = max(1, int(round(1.0 / env.step_dt / args_cli.video_stride)))
    while simulation_app.is_running():
        if max_episode_count > 0 and episodes_done >= max_episode_count:
            break
        # Last round may need fewer than num_envs episodes; only count those.
        batch = (
            num_envs
            if max_episode_count <= 0
            else min(num_envs, max_episode_count - episodes_done)
        )
        print(
            f"[Evaluation] Evaluating episodes {episodes_done + 1}-{episodes_done + batch} "
            f"across {num_envs} parallel env(s)..."
        )

        with torch.inference_mode():
            obs_dict, _ = env.reset()
        policy.reset()
        controller.reset()
        done_mask = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        success_mask = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        video_frames = {(e, c): [] for e in range(num_envs) for c in video_cameras} if args_cli.save_video_dir else None
        video_step = 0

        while simulation_app.is_running():
            with torch.inference_mode():
                if controller.reset_state:
                    controller.reset()
                    break

                if video_frames is not None and video_step % args_cli.video_stride == 0:
                    for e in range(num_envs):
                        if bool(done_mask[e]):
                            continue  # frames after an env's first termination belong to the auto-reset episode
                        for c in video_cameras:
                            if c in obs_dict["policy"]:
                                video_frames[(e, c)].append(obs_dict["policy"][c][e].cpu().numpy())
                video_step += 1

                policy_obs_dict = preprocess_obs_dict(
                    obs_dict["policy"], language_instruction
                )
                actions = policy.get_action(policy_obs_dict).to(env.device)
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, teleop_device)
                obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(actions)

                newly_done = (reset_terminated | reset_time_outs) & ~done_mask
                # A success DoneTerm reports via `terminated`; time out via `truncated`.
                success_mask |= reset_terminated & newly_done
                done_mask |= reset_terminated | reset_time_outs
                if bool(done_mask.all()):
                    break
                if rate_limiter:
                    rate_limiter.sleep(env)

        round_success = int(success_mask[:batch].sum().item())
        if video_frames is not None:
            for e in range(batch):
                outcome = "success" if bool(success_mask[e]) else "fail"
                for c in video_cameras:
                    if video_frames[(e, c)]:
                        video_path = str(
                            _Path(args_cli.save_video_dir) / f"ep{episodes_done + e + 1:03d}_{c}_{outcome}.mp4"
                        )
                        _write_episode_video(video_frames[(e, c)], video_path, video_fps)
                        print(f"[rollout] saved {video_path} ({len(video_frames[(e, c)])} frames)", flush=True)
        success_count += round_success
        episodes_done += batch
        print(
            f"[Evaluation] round done: {round_success}/{batch} succeeded; "
            f"cumulative success rate: {success_count / episodes_done:.3f} "
            f"[{success_count}/{episodes_done}]"
        )

    if max_episode_count > 0:
        print(
            f"[Evaluation] Final success rate: {success_count / max_episode_count:.3f} "
            f" [{success_count}/{max_episode_count}]"
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
