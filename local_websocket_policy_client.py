"""Minimal localhost-only client for the OpenPI websocket wire protocol.

The NumPy msgpack codec follows OpenPI's Apache-2.0 openpi-client package, but
this narrow client avoids its obsolete ``numpy<2`` packaging constraint in the
MuJoCo 3 simulation environment.
"""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client


def _pack_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in (
        "V", "O", "c",
    ):
        raise ValueError(f"unsupported dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_array(value: dict) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class LocalWebsocketPolicyClient:
    """Synchronous client that refuses non-loopback policy endpoints."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("policy client is restricted to a loopback endpoint")
        self.uri = f"ws://{host}:{int(port)}"
        self._packer = _Packer()
        self._connection = websockets.sync.client.connect(
            self.uri, compression=None, max_size=None
        )
        self._metadata = _unpackb(self._connection.recv())

    def get_server_metadata(self) -> dict:
        return self._metadata

    def infer(self, observation: dict) -> dict:
        self._connection.send(self._packer.pack(observation))
        response = self._connection.recv()
        if isinstance(response, str):
            raise RuntimeError(f"policy server error:\n{response}")
        return _unpackb(response)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        self.close()
