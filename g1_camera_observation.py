#!/usr/bin/env python3
"""Camera-frame boundary for the three LGG100 RGB observations.

The protocol follows Yuhao's CameraClient: frames arrive as BGR, the head feed
uses its left half only when configured as binocular, and the three frames are
sequential rather than a synchronized snapshot. No camera or robot network API
is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from g1_policy_contract import (
    IMAGE_KEYS,
    SOURCE_IMAGE_SHAPE,
    preprocess_rgb_image,
    validate_observation,
)


CAMERA_NAMES = ("head_left", "left_wrist", "right_wrist")
CAMERA_TO_POLICY_KEY = dict(zip(CAMERA_NAMES, IMAGE_KEYS, strict=True))


@dataclass(frozen=True)
class BGRFrame:
    name: str
    bgr: np.ndarray
    received_monotonic_ns: int
    received_unix_ns: int | None = None
    source_timestamp_ns: int | None = None
    binocular: bool = False


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def bgr_frame_to_rgb224(frame: BGRFrame) -> tuple[np.ndarray, dict[str, Any]]:
    if frame.name not in CAMERA_NAMES:
        raise ValueError(f"unexpected camera name: {frame.name}")
    bgr = np.asarray(frame.bgr)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError(f"{frame.name} must be uint8 HWC BGR, got {bgr.dtype} {bgr.shape}")
    selected = bgr
    head_left_crop_applied = False
    if frame.name == "head_left" and frame.binocular:
        if bgr.shape[1] % 2:
            raise ValueError("binocular head frame width must be even")
        selected = bgr[:, : bgr.shape[1] // 2, :]
        head_left_crop_applied = True
    if selected.shape != SOURCE_IMAGE_SHAPE:
        raise ValueError(
            f"{frame.name} selected source must be {SOURCE_IMAGE_SHAPE}, got {selected.shape}"
        )
    rgb = np.ascontiguousarray(selected[..., ::-1])
    policy = preprocess_rgb_image(rgb)
    metadata = {
        "camera": frame.name,
        "source_color_order": "BGR",
        "policy_color_order": "RGB",
        "source_shape": list(bgr.shape),
        "selected_shape": list(selected.shape),
        "policy_shape": list(policy.shape),
        "head_binocular": bool(frame.binocular) if frame.name == "head_left" else None,
        "head_left_crop_applied": head_left_crop_applied,
        "received_monotonic_ns": int(frame.received_monotonic_ns),
        "received_unix_ns": frame.received_unix_ns,
        "source_timestamp_ns": frame.source_timestamp_ns,
        "source_bgr_sha256": _sha256(bgr),
        "selected_bgr_sha256": _sha256(selected),
        "policy_rgb_sha256": _sha256(policy),
    }
    return policy, metadata


def build_three_camera_observation(
    frames: list[BGRFrame], state: np.ndarray, prompt: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_name = {frame.name: frame for frame in frames}
    if len(frames) != len(by_name) or set(by_name) != set(CAMERA_NAMES):
        raise ValueError("exactly one head-left, left-wrist, and right-wrist frame is required")
    ordered = [by_name[name] for name in CAMERA_NAMES]
    received = [int(frame.received_monotonic_ns) for frame in ordered]
    if any(right < left for left, right in zip(received, received[1:])):
        raise ValueError("sequential camera receive timestamps must be nondecreasing")

    observation: dict[str, Any] = {
        "observation/state": np.asarray(state, dtype=np.float32),
        "prompt": prompt,
    }
    metadata = []
    for frame in ordered:
        image, image_metadata = bgr_frame_to_rgb224(frame)
        observation[CAMERA_TO_POLICY_KEY[frame.name]] = image
        metadata.append(image_metadata)
    validate_observation(observation)
    return observation, {
        "camera_frames": metadata,
        "true_simultaneous_snapshot": False,
        "sequential_capture_span_ms": (received[-1] - received[0]) / 1_000_000.0,
        "all_policy_images_uint8_rgb_224": True,
    }
