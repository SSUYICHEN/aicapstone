"""Generic serve loop of the policy service. Works with any ``runner`` that provides ``setup``, ``reset`` and ``act``."""

from __future__ import annotations

import socket
import threading
import traceback
from collections.abc import Callable

from .client import IMAGE_PREFIX
from .protocol import recv_message, send_message


def handle_connection(runner, conn: socket.socket) -> None:
    """Serve one client until it disconnects. Errors in a request are reported to the client, not fatal."""
    while True:
        try:
            header, arrays = recv_message(conn)
        except (ConnectionError, OSError):
            return
        try:
            message_type = header.get("type")
            reply_arrays = None
            if message_type == "setup":
                camera_infos = {key: tuple(shape) for key, shape in header["camera_infos"].items()}
                runner.setup(camera_infos, header["task_type"], int(header["actions_per_chunk"]))
            elif message_type == "reset":
                runner.reset()
            elif message_type == "act":
                images = {key[len(IMAGE_PREFIX) :]: value for key, value in arrays.items() if key.startswith(IMAGE_PREFIX)}
                reply_arrays = {"actions": runner.act(images, arrays["joint_pos"], header.get("task", ""))}
            elif message_type == "ping":
                pass
            else:
                raise ValueError(f"unknown message type {message_type!r}")
            send_message(conn, {"ok": True}, reply_arrays)
        except Exception:  # noqa: BLE001 - report to the client and keep serving
            try:
                send_message(conn, {"ok": False, "error": traceback.format_exc()})
            except OSError:
                return


def serve(
    runner,
    host: str = "127.0.0.1",
    port: int = 5555,
    stop: threading.Event | None = None,
    on_listening: Callable[[int], None] | None = None,
) -> None:
    """Accept clients one at a time (a rollout process at a time) until ``stop`` is set."""
    stop = stop or threading.Event()
    with socket.create_server((host, port)) as server:
        server.settimeout(0.5)
        if on_listening is not None:
            on_listening(server.getsockname()[1])
        while not stop.is_set():
            try:
                conn, address = server.accept()
            except socket.timeout:
                continue
            print(f"[policy-server] client connected: {address[0]}:{address[1]}", flush=True)
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(None)
                handle_connection(runner, conn)
            print("[policy-server] client disconnected", flush=True)
