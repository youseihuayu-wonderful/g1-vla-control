import ast
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import yaml

from g1_camera_server_readonly import CAMERA_KEYS, load_camera_only_config, run


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "g1_camera_server_readonly.py"


class G1CameraServerReadonlyTests(unittest.TestCase):
    def _config(self):
        return {
            name: {"enable_zmq": True, "enable_webrtc": True, "zmq_port": 55555 + index}
            for index, name in enumerate(CAMERA_KEYS)
        }

    def test_load_config_requires_three_zmq_cameras_and_disables_webrtc(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.yaml"
            path.write_text(yaml.safe_dump(self._config()))
            config = load_camera_only_config(path)
        for name in CAMERA_KEYS:
            self.assertTrue(config[name]["enable_zmq"])
            self.assertFalse(config[name]["enable_webrtc"])

    def test_missing_or_non_zmq_camera_rejects(self):
        for mutate in ("missing", "disabled"):
            config = self._config()
            if mutate == "missing":
                config.pop("right_wrist_camera")
            else:
                config["right_wrist_camera"]["enable_zmq"] = False
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "camera.yaml"
                path.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):
                    load_camera_only_config(path)

    def test_run_uses_upstream_wait_for_full_cleanup(self):
        calls = []

        class FakeServer:
            def __init__(self, config, **kwargs):
                calls.append(("init", config, kwargs))

            def start(self):
                calls.append(("start",))

            def stop(self):
                calls.append(("stop",))

            def wait(self):
                calls.append(("wait",))

        upstream = SimpleNamespace(
            reload_uvc_driver=lambda: calls.append(("unsafe_reload",)),
            ImageServer=FakeServer,
        )
        teleimager = ModuleType("teleimager")
        teleimager.image_server = upstream
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "camera.yaml"
            status_path = Path(directory) / "status.json"
            config_path.write_text(yaml.safe_dump(self._config()))
            with mock.patch.dict("sys.modules", {"teleimager": teleimager}), mock.patch(
                "g1_camera_server_readonly.signal.signal"
            ):
                run(config_path, status_path)
            self.assertTrue(status_path.exists())
        self.assertFalse(any(call[0] == "unsafe_reload" for call in calls))
        self.assertEqual([call[0] for call in calls], ["init", "start", "wait"])

    def test_module_has_no_robot_or_privileged_execution_api(self):
        tree = ast.parse(MODULE.read_text())
        imported_modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        identifiers = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertNotIn("subprocess", imported_modules)
        self.assertNotIn("os", imported_modules)
        self.assertFalse({
            "ChannelPublisher", "ChannelSubscriber", "LowCmd_",
            "MotionSwitcherClient", "SportClient", "system", "Popen",
        } & (identifiers | attributes))


if __name__ == "__main__":
    unittest.main()
