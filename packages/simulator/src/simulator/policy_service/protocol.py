"""Binary request / response protocol: a JSON header followed by raw numpy array bytes.

Wire format of one message::

    [4 bytes: header length, big endian][header: UTF-8 JSON][array 0 bytes][array 1 bytes]...

The JSON header carries the user fields plus ``_arrays``: a list of ``{"name", "dtype", "shape"}`` describing the
array bytes that follow, in order. Only plain numeric dtypes are accepted (no pickle, no object arrays), so a message
can never execute code on the receiving side.
"""

from __future__ import annotations

import json
import socket
import struct

import numpy as np

_LENGTH = struct.Struct("!I")
MAX_HEADER_BYTES = 1 << 20  # 1 MiB
MAX_ARRAY_BYTES = 1 << 30  # 1 GiB per array
_ALLOWED_KINDS = "biuf"  # bool, int, uint, float


def _recv_exact(sock: socket.socket, size: int) -> bytearray:
    buffer = bytearray(size)
    view = memoryview(buffer)
    received = 0
    while received < size:
        count = sock.recv_into(view[received:], size - received)
        if count == 0:
            raise ConnectionError("connection closed while receiving a message")
        received += count
    return buffer


def send_message(sock: socket.socket, header: dict, arrays: dict[str, np.ndarray] | None = None) -> None:
    arrays = arrays or {}
    prepared = []
    specs = []
    for name, array in arrays.items():
        array = np.ascontiguousarray(array)
        if array.dtype.kind not in _ALLOWED_KINDS:
            raise ValueError(f"array '{name}' has unsupported dtype {array.dtype}")
        specs.append({"name": name, "dtype": array.dtype.str, "shape": list(array.shape)})
        prepared.append(array)
    payload = json.dumps({**header, "_arrays": specs}).encode("utf-8")
    if len(payload) > MAX_HEADER_BYTES:
        raise ValueError("message header too large")
    sock.sendall(_LENGTH.pack(len(payload)) + payload)
    for array in prepared:
        if array.nbytes:
            sock.sendall(memoryview(array).cast("B"))


def recv_message(sock: socket.socket) -> tuple[dict, dict[str, np.ndarray]]:
    (header_length,) = _LENGTH.unpack(_recv_exact(sock, _LENGTH.size))
    if header_length > MAX_HEADER_BYTES:
        raise ValueError(f"message header too large ({header_length} bytes)")
    header = json.loads(bytes(_recv_exact(sock, header_length)).decode("utf-8"))
    arrays: dict[str, np.ndarray] = {}
    for spec in header.pop("_arrays", []):
        dtype = np.dtype(spec["dtype"])
        if dtype.kind not in _ALLOWED_KINDS:
            raise ValueError(f"array '{spec['name']}' has unsupported dtype {dtype}")
        shape = tuple(int(dim) for dim in spec["shape"])
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if nbytes > MAX_ARRAY_BYTES:
            raise ValueError(f"array '{spec['name']}' too large ({nbytes} bytes)")
        raw = _recv_exact(sock, nbytes) if nbytes else bytearray()
        arrays[spec["name"]] = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return header, arrays
