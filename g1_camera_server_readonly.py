#!/usr/bin/env python3
"""Camera-only Teleimager launcher for the real-G1 read-only Shadow stage.

This wrapper disables WebRTC, skips Teleimager's sudo UVC-driver reload, and
starts only configured local camera capture plus ZMQ image publication. It has
no Unitree SDK, DDS, controller, mode-switch, or robot-command path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal

import yaml


CAMERA_KEYS = ("head_camera", "left_wrist_camera", "right_wrist_camera")


def load_camera_only_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("camera config must be a mapping")
    for name in CAMERA_KEYS:
        camera = config.get(name)
        if not isinstance(camera, dict):
            raise ValueError(f"missing camera config: {name}")
        if not camera.get("enable_zmq", False):
            raise ValueError(f"ZMQ must be enabled for {name}")
        camera["enable_webrtc"] = False
    return config


def run(config_path: Path, status_path: Path) -> None:
    from teleimager import image_server as upstream

    config = load_camera_only_config(config_path)
    # Upstream CameraFinder unconditionally attempts `sudo modprobe`. The
    # read-only stage must not unload/reload a kernel driver or invoke sudo.
    upstream.reload_uvc_driver = lambda: None
    server = upstream.ImageServer(
        config,
        realsense_enable=False,
        camera_finder_verbose=False,
    )
    status = {
        "schema_version": "g1_camera_server_readonly_v1",
        "camera_keys": list(CAMERA_KEYS),
        "camera_zmq_enabled": True,
        "camera_webrtc_enabled": False,
        "uvc_driver_reload_skipped": True,
        "sudo_invoked": False,
        "unitree_dds_imported": False,
        "robot_command_publisher_created": False,
        "mode_change_requested": False,
        "robot_command_sent": False,
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2) + "\n")

    def stop(_signum, _frame):
        server.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    server.start()
    status["server_start_returned"] = True
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    # Upstream wait() blocks on the same stop event and then performs the full
    # publisher-thread, socket, and camera cleanup. Do not replace this with a
    # wrapper-only event: that can leave non-daemon Teleimager threads alive.
    server.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
