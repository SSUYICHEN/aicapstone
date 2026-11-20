"""Tests for the policy service (protocol, client, serve loop) using a fake runner: no GPU, lerobot or Isaac Lab."""

from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

# `simulator` is not installed in the default (uv) environment; these modules only need the standard library + numpy.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "simulator" / "src"))

from simulator.policy_service.client import RemotePolicyClient  # noqa: E402
from simulator.policy_service.protocol import recv_message, send_message  # noqa: E402
from simulator.policy_service.server import serve  # noqa: E402


class FakeRunner:
    def __init__(self):
        self.calls: list[tuple] = []

    def setup(self, camera_infos, task_type, actions_per_chunk):
        self.calls.append(("setup", camera_infos, task_type, actions_per_chunk))

    def reset(self):
        self.calls.append(("reset",))

    def act(self, images, joint_pos, task):
        self.calls.append(("act", sorted(images), joint_pos.shape, task))
        if task == "boom":
            raise ValueError("fake failure")
        # one action per env: the first 8 joint values doubled
        return (joint_pos[:, :8] * 2.0).astype(np.float32)


@pytest.fixture()
def server():
    runner = FakeRunner()
    stop = threading.Event()
    ready = threading.Event()
    port: list[int] = []

    def on_listening(p: int) -> None:
        port.append(p)
        ready.set()

    thread = threading.Thread(
        target=serve, args=(runner,), kwargs={"host": "127.0.0.1", "port": 0, "stop": stop, "on_listening": on_listening}
    )
    thread.start()
    assert ready.wait(5)
    yield runner, port[0]
    stop.set()
    thread.join(5)


def test_protocol_roundtrip_preserves_header_and_arrays():
    a, b = socket.socketpair()
    with a, b:
        arrays = {
            "image": np.random.randint(0, 255, (2, 4, 6, 3), dtype=np.uint8),
            "joint_pos": np.random.rand(2, 9).astype(np.float32),
            "empty": np.zeros((0, 3), dtype=np.float32),
        }
        send_message(a, {"type": "act", "task": "pick up the cup"}, arrays)
        header, received = recv_message(b)
    assert header == {"type": "act", "task": "pick up the cup"}
    assert set(received) == set(arrays)
    for key, value in arrays.items():
        assert received[key].dtype == value.dtype
        np.testing.assert_array_equal(received[key], value)


def test_protocol_rejects_object_arrays():
    a, b = socket.socketpair()
    with a, b, pytest.raises(ValueError):
        send_message(a, {}, {"bad": np.array([object()], dtype=object)})


def test_client_server_full_flow(server):
    runner, port = server
    client = RemotePolicyClient("127.0.0.1", port, timeout_s=5)
    client.setup({"front": (480, 640), "wrist": (480, 640)}, "franka_panda", 1)
    client.reset()
    joint_pos = np.arange(18, dtype=np.float32).reshape(2, 9)
    images = {k: np.zeros((2, 480, 640, 3), dtype=np.uint8) for k in ("front", "wrist")}
    actions = client.act(images, joint_pos, "stack the cups")
    client.close()

    np.testing.assert_allclose(actions, joint_pos[:, :8] * 2.0)
    assert actions.dtype == np.float32
    assert runner.calls[0] == ("setup", {"front": (480, 640), "wrist": (480, 640)}, "franka_panda", 1)
    assert runner.calls[1] == ("reset",)
    assert runner.calls[2] == ("act", ["front", "wrist"], (2, 9), "stack the cups")


def test_server_reports_errors_and_keeps_serving(server):
    _, port = server
    client = RemotePolicyClient("127.0.0.1", port, timeout_s=5)
    client.setup({"front": (4, 4)}, "franka_panda", 1)
    images = {"front": np.zeros((1, 4, 4, 3), dtype=np.uint8)}
    joint_pos = np.zeros((1, 9), dtype=np.float32)
    with pytest.raises(RuntimeError, match="fake failure"):
        client.act(images, joint_pos, "boom")
    # the same connection still works after a failed request
    assert client.act(images, joint_pos, "ok").shape == (1, 8)
    client.close()


def test_server_accepts_clients_one_after_another(server):
    runner, port = server
    for _ in range(2):
        client = RemotePolicyClient("127.0.0.1", port, timeout_s=5)
        client.setup({"front": (4, 4)}, "franka_panda", 1)
        client.close()
    assert [call[0] for call in runner.calls].count("setup") == 2
