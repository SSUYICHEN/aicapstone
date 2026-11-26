"""Client side of the policy service (used inside the simulator container, needs only numpy)."""

from __future__ import annotations

import socket

import numpy as np

from .protocol import recv_message, send_message

IMAGE_PREFIX = "image/"


class RemotePolicyClient:
    """Talks to ``scripts/policy_server.py``. One instance = one connection."""

    def __init__(self, host: str, port: int, timeout_s: float | None = 600.0):
        # The very first inference of a large policy can take a while (model warm-up), hence the long default.
        self._sock = socket.create_connection((host, port), timeout=timeout_s)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _request(self, header: dict, arrays: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        send_message(self._sock, header, arrays)
        reply, reply_arrays = recv_message(self._sock)
        if not reply.get("ok", False):
            raise RuntimeError(f"policy server error:\n{reply.get('error', 'unknown error')}")
        return reply_arrays

    def setup(self, camera_infos: dict[str, tuple[int, int]], task_type: str, actions_per_chunk: int) -> None:
        self._request(
            {
                "type": "setup",
                "camera_infos": {key: [int(shape[0]), int(shape[1])] for key, shape in camera_infos.items()},
                "task_type": task_type,
                "actions_per_chunk": int(actions_per_chunk),
            }
        )

    def reset(self) -> None:
        self._request({"type": "reset"})

    def act(self, images: dict[str, np.ndarray], joint_pos: np.ndarray, task: str) -> np.ndarray:
        """images: ``{camera: (num_envs, H, W, 3) uint8}``, joint_pos: ``(num_envs, num_joints)``. Returns ``(num_envs, action_dim)``."""
        arrays = {f"{IMAGE_PREFIX}{key}": value for key, value in images.items()}
        arrays["joint_pos"] = np.asarray(joint_pos, dtype=np.float32)
        return self._request({"type": "act", "task": task}, arrays)["actions"]

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
