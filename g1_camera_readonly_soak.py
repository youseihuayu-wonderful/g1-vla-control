#!/usr/bin/env python3
"""Bounded three-camera freshness/freeze soak for read-only real-G1 Shadow.

The utility subscribes only to Teleimager camera ZMQ streams. It imports no
Unitree SDK, creates no robot DDS participant or command Publisher, and stores
only frame metadata and SHA-256 digests—not image bytes. Camera getters are
called sequentially and are never represented as a synchronized snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np


CAMERAS = ("head", "left_wrist", "right_wrist")
EXPECTED_SHAPE = [480, 640, 3]
MIN_AVAILABILITY_RATE = 0.99
MIN_UNIQUE_HASH_FRACTION = 0.80
MAX_CONSECUTIVE_IDENTICAL = 3
MAX_CYCLE_GAP_MS = 200.0


def _max_identical_run(values: list[str]) -> int:
    maximum = 0
    current = 0
    previous = None
    for value in values:
        if value == previous:
            current += 1
        else:
            previous = value
            current = 1
        maximum = max(maximum, current)
    return maximum


def summarize_camera_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [sample for sample in samples if sample["available"]]
    hashes = [sample["sha256"] for sample in successful]
    count = len(samples)
    available = len(successful)
    unique = len(set(hashes))
    return {
        "attempted_samples": count,
        "available_samples": available,
        "missing_samples": count - available,
        "availability_rate": available / count if count else 0.0,
        "all_expected_shape": bool(successful) and all(
            sample["shape"] == EXPECTED_SHAPE for sample in successful
        ),
        "all_uint8": bool(successful) and all(
            sample["dtype"] == "uint8" for sample in successful
        ),
        "all_finite": bool(successful) and all(sample["finite"] for sample in successful),
        "unique_hash_count": unique,
        "unique_hash_fraction": unique / available if available else 0.0,
        "max_consecutive_identical_hashes": _max_identical_run(hashes),
        "source_fps_median": (
            statistics.median(
                sample["source_fps"]
                for sample in successful
                if isinstance(sample.get("source_fps"), (int, float))
                and math.isfinite(sample["source_fps"])
            )
            if any(
                isinstance(sample.get("source_fps"), (int, float))
                and math.isfinite(sample["source_fps"])
                for sample in successful
            )
            else None
        ),
    }


def run_soak(
    host: str,
    request_port: int,
    *,
    duration_s: float = 10.0,
    sample_hz: float = 15.0,
    initial_wait_s: float = 5.0,
) -> dict[str, Any]:
    if not 1.0 <= duration_s <= 60.0:
        raise ValueError("duration_s must be in [1,60]")
    if not 1.0 <= sample_hz <= 30.0:
        raise ValueError("sample_hz must be in [1,30]")
    if not 0.1 <= initial_wait_s <= 30.0:
        raise ValueError("initial_wait_s must be in [0.1,30]")
    try:
        from teleimager import ImageClient
    except ImportError:
        from teleimager.image_client import ImageClient

    report: dict[str, Any] = {
        "schema_version": "g1_three_camera_readonly_freshness_soak_v1",
        "scope": "Bounded sequential metadata/hash-only camera freshness soak.",
        "source_color_order": "BGR",
        "true_simultaneous_snapshot": False,
        "image_bytes_saved": False,
        "requested": {
            "duration_s": duration_s,
            "sample_hz": sample_hz,
            "initial_wait_s": initial_wait_s,
        },
        "thresholds": {
            "minimum_availability_rate": MIN_AVAILABILITY_RATE,
            "minimum_unique_hash_fraction": MIN_UNIQUE_HASH_FRACTION,
            "maximum_consecutive_identical_hashes": MAX_CONSECUTIVE_IDENTICAL,
            "maximum_cycle_gap_ms": MAX_CYCLE_GAP_MS,
            "expected_shape": EXPECTED_SHAPE,
            "expected_dtype": "uint8",
        },
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
        client = ImageClient(host=host, request_port=request_port, request_bgr=True)
        config = client.get_cam_config()
        report["config"] = {
            name: {
                "present": name in config,
                "enable_zmq": bool(config.get(name, {}).get("enable_zmq", False)),
            }
            for name in ("head_camera", "left_wrist_camera", "right_wrist_camera")
        }
        getters = {
            "head": client.get_head_frame,
            "left_wrist": client.get_left_wrist_frame,
            "right_wrist": client.get_right_wrist_frame,
        }

        initial_available = {name: False for name in CAMERAS}
        initial_deadline = time.monotonic() + initial_wait_s
        while time.monotonic() < initial_deadline and not all(initial_available.values()):
            for name, getter in getters.items():
                frame = getter()
                initial_available[name] = initial_available[name] or frame.bgr is not None
            if not all(initial_available.values()):
                time.sleep(0.02)
        report["initial_frames_available"] = initial_available

        target_cycles = max(1, round(duration_s * sample_hz))
        interval_s = 1.0 / sample_hz
        samples: dict[str, list[dict[str, Any]]] = {name: [] for name in CAMERAS}
        cycle_times_ns: list[int] = []
        started_ns = time.monotonic_ns()
        next_cycle = time.monotonic()
        for cycle in range(target_cycles):
            cycle_times_ns.append(time.monotonic_ns())
            for name, getter in getters.items():
                frame = getter()
                image = frame.bgr
                sample: dict[str, Any] = {
                    "cycle": cycle,
                    "received_monotonic_ns": time.monotonic_ns(),
                    "available": image is not None,
                    "source_fps": getattr(frame, "fps", None),
                }
                if image is not None:
                    contiguous = np.ascontiguousarray(image)
                    sample.update({
                        "shape": list(image.shape),
                        "dtype": str(image.dtype),
                        "finite": bool(np.all(np.isfinite(image))),
                        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
                    })
                else:
                    sample.update({
                        "shape": None,
                        "dtype": None,
                        "finite": False,
                        "sha256": None,
                    })
                samples[name].append(sample)
            next_cycle += interval_s
            remaining = next_cycle - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_cycle = time.monotonic()
        ended_ns = time.monotonic_ns()

        cycle_gaps_ms = [
            (right - left) / 1_000_000.0
            for left, right in zip(cycle_times_ns, cycle_times_ns[1:])
        ]
        report["observed"] = {
            "cycles": target_cycles,
            "elapsed_s": (ended_ns - started_ns) / 1_000_000_000.0,
            "cycle_gap_ms": {
                "minimum": min(cycle_gaps_ms) if cycle_gaps_ms else None,
                "median": statistics.median(cycle_gaps_ms) if cycle_gaps_ms else None,
                "maximum": max(cycle_gaps_ms) if cycle_gaps_ms else None,
            },
        }
        report["camera_summary"] = {
            name: summarize_camera_samples(camera_samples)
            for name, camera_samples in samples.items()
        }
        report["samples"] = samples
        report["decision"] = {
            "initial_all_three_available": all(initial_available.values()),
            "availability_passed": all(
                summary["availability_rate"] >= MIN_AVAILABILITY_RATE
                for summary in report["camera_summary"].values()
            ),
            "shape_dtype_finite_passed": all(
                summary["all_expected_shape"]
                and summary["all_uint8"]
                and summary["all_finite"]
                for summary in report["camera_summary"].values()
            ),
            "freeze_check_passed": all(
                summary["unique_hash_fraction"] >= MIN_UNIQUE_HASH_FRACTION
                and summary["max_consecutive_identical_hashes"]
                <= MAX_CONSECUTIVE_IDENTICAL
                for summary in report["camera_summary"].values()
            ),
            "cycle_gap_passed": bool(cycle_gaps_ms)
            and max(cycle_gaps_ms) <= MAX_CYCLE_GAP_MS,
            "live_gate_promoted": False,
            "robot_motion_allowed": False,
        }
        report["success"] = bool(
            all(report["decision"][name] for name in (
                "initial_all_three_available",
                "availability_passed",
                "shape_dtype_finite_passed",
                "freeze_check_passed",
                "cycle_gap_passed",
            ))
        )
    except Exception as exc:
        report["success"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--request-port", type=int, default=60000)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--sample-hz", type=float, default=15.0)
    parser.add_argument("--initial-wait-s", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_soak(
        args.host,
        args.request_port,
        duration_s=args.duration_s,
        sample_hz=args.sample_hz,
        initial_wait_s=args.initial_wait_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
