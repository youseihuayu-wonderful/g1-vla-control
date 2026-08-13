import sys
from pathlib import Path
import unittest

import msgpack
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_websocket_policy_client import (
    LocalWebsocketPolicyClient, _pack_array, _unpack_array,
)


class LocalWebsocketPolicyClientTests(unittest.TestCase):
    def test_numpy_msgpack_round_trip(self):
        source = {
            "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
            "state": np.linspace(0.0, 1.0, 16, dtype=np.float32),
        }
        packed = msgpack.packb(source, default=_pack_array)
        restored = msgpack.unpackb(packed, object_hook=_unpack_array)
        np.testing.assert_array_equal(restored["image"], source["image"])
        np.testing.assert_array_equal(restored["state"], source["state"])

    def test_non_loopback_endpoint_is_rejected_before_connect(self):
        with self.assertRaises(ValueError):
            LocalWebsocketPolicyClient("0.0.0.0", 8000)
        with self.assertRaises(ValueError):
            LocalWebsocketPolicyClient("10.0.0.1", 8000)

    def test_object_array_is_rejected(self):
        with self.assertRaises(ValueError):
            _pack_array(np.asarray([object()], dtype=object))


if __name__ == "__main__":
    unittest.main()
