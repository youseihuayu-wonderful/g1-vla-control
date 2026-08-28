#!/usr/bin/env python3
"""Fixed three-camera read-only probe for the G1 teleimager image server.

This utility imports no Unitree DDS API and contains no robot publisher, mode
switcher, controller, or command path. Camera requests are sequential and are
reported as such; they must not be represented as a synchronized snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


def _scalar_metadata(frame: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "timestamp", "timestamp_ns", "capture_time", "capture_time_ns",
        "frame_id", "sequence", "seq",
    ):
        value = getattr(frame, name, None)
        if isinstance(value, (str, int, float, bool)):
            result[name] = value
    return result


def probe_three_cameras(
    host: str,
    request_port: int,
    *,
    frame_timeout_s: float = 5.0,
    retry_interval_s: float = 0.02,
) -> dict[str, Any]:
    if not 0.1 <= frame_timeout_s <= 30.0:
        raise ValueError("frame_timeout_s must be in [0.1,30]")
    if not 0.001 <= retry_interval_s <= 0.5:
        raise ValueError("retry_interval_s must be in [0.001,0.5]")
    try:
        from teleimager import ImageClient
    except ImportError:
        from teleimager.image_client import ImageClient

    result: dict[str, Any] = {
        "schema_version": "g1_three_camera_readonly_probe_v1",
        "scope": "Read-only teleimager configuration query and one sequential frame per camera.",
        "source_color_order": "BGR",
        "true_simultaneous_snapshot": False,
        "safety": {
            "camera_read_only": True,
            "unitree_dds_imported": False,
            "publisher_created": False,
            "mode_change_requested": False,
            "robot_command_sent": False,
        },
    }
    client = None
    try:
        started = time.monotonic_ns()
        client = ImageClient(
            host=host, request_port=request_port, request_bgr=True
        )
        config = client.get_cam_config()
        result["connect_and_config_ms"] = (
            time.monotonic_ns() - started
        ) / 1_000_000.0
        result["config"] = {}
        for name in (
            "head_camera", "left_wrist_camera", "right_wrist_camera"
        ):
            camera = config.get(name, {})
            result["config"][name] = {
                "present": name in config,
                "enable_zmq": bool(camera.get("enable_zmq", False)),
                "binocular": bool(camera.get("binocular", False)),
            }

        frames = {}
        for name, getter in (
            ("head", client.get_head_frame),
            ("left_wrist", client.get_left_wrist_frame),
            ("right_wrist", client.get_right_wrist_frame),
        ):
            request_start = time.monotonic_ns()
            deadline_ns = request_start + int(frame_timeout_s * 1_000_000_000)
            attempts = 0
            frame = None
            image = None
            while time.monotonic_ns() < deadline_ns:
                attempts += 1
                frame = getter()
                image = frame.bgr
                if image is not None:
                    break
                time.sleep(retry_interval_s)
            received_monotonic_ns = time.monotonic_ns()
            frames[name] = {
                "success": image is not None,
                "shape": list(image.shape) if image is not None else None,
                "dtype": str(image.dtype) if image is not None else None,
                "finite": bool(np.all(np.isfinite(image))) if image is not None else False,
                "request_ms": (
                    received_monotonic_ns - request_start
                ) / 1_000_000.0,
                "attempts": attempts,
                "frame_timeout_s": frame_timeout_s,
                "received_unix_ns": time.time_ns(),
                "received_monotonic_ns": received_monotonic_ns,
                "sha256": (
                    hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                    if image is not None else None
                ),
                "source_metadata": _scalar_metadata(frame) if frame is not None else {},
            }
        result["frames"] = frames
        result["all_three_frames_available"] = all(
            frame["success"] for frame in frames.values()
        )
        result["sequential_capture_span_ms"] = (
            frames["right_wrist"]["received_monotonic_ns"]
            - frames["head"]["received_monotonic_ns"]
        ) / 1_000_000.0
        result["success"] = bool(
            result["all_three_frames_available"]
            and all(camera["enable_zmq"] for camera in result["config"].values())
        )
    except Exception as exc:
        result["success"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--request-port", type=int, default=60000)
    parser.add_argument("--frame-timeout-s", type=float, default=5.0)
    parser.add_argument("--retry-interval-s", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = probe_three_cameras(
        args.host,
        args.request_port,
        frame_timeout_s=args.frame_timeout_s,
        retry_interval_s=args.retry_interval_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print("CAMERA_PROBE_JSON=" + json.dumps(report, separators=(",", ":")))
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
