"""Single policy contract shared by MuJoCo and the future G1 EDU adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "g1_policy_contract.yaml"


def load_contract() -> dict[str, Any]:
    """Load the JSON-compatible YAML without adding a runtime YAML dependency."""
    contract = json.loads(CONTRACT_PATH.read_text())
    if contract.get("contract_id") != "g1_edu_dual_dex1_eef_v1":
        raise ValueError("Unexpected G1 policy contract id")
    return contract


CONTRACT = load_contract()
CONTRACT_ID = str(CONTRACT["contract_id"])
CONTRACT_VERSION = str(CONTRACT["contract_version"])
CONTRACT_SHA256 = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
IMAGE_KEYS = tuple(CONTRACT["observation"]["image_keys"])
STATE_KEY = str(CONTRACT["observation"]["state_key"])
PROMPT_KEY = str(CONTRACT["observation"]["prompt_key"])
SOURCE_IMAGE_SHAPE = tuple(CONTRACT["observation"]["source_image_shape"])
IMAGE_SHAPE = tuple(CONTRACT["observation"]["image_shape"])
STATE_DIM = int(CONTRACT["observation"]["state_dimension"])
ACTION_DIM = int(CONTRACT["action"]["dimension"])
POLICY_RATE_HZ = float(CONTRACT["timing"]["policy_rate_hz"])
ACTION_HORIZON = int(CONTRACT["timing"]["action_horizon"])
QUATERNION_NORM_TOLERANCE = float(CONTRACT["action"]["quaternion_norm_tolerance"])
GRIPPER_RANGE_RAD = tuple(float(x) for x in CONTRACT["action"]["gripper_range_rad"])


def contract_metadata(*, verified: bool) -> dict[str, Any]:
    return {
        "g1_policy_contract_id": CONTRACT_ID,
        "g1_policy_contract_version": CONTRACT_VERSION,
        "g1_policy_contract_sha256": CONTRACT_SHA256,
        "g1_contract_verified": bool(verified),
    }


def preprocess_rgb_image(image: Any) -> np.ndarray:
    """Replicate OpenPI ``resize_with_pad`` for a frozen 640×480 RGB image."""
    image = np.asarray(image)
    if image.shape != SOURCE_IMAGE_SHAPE or image.dtype != np.uint8:
        raise ValueError(
            f"G1 source image must be uint8 {SOURCE_IMAGE_SHAPE}, got {image.dtype} {image.shape}"
        )
    source_height, source_width = image.shape[:2]
    target_height, target_width = IMAGE_SHAPE[:2]
    ratio = max(source_width / target_width, source_height / target_height)
    resized_height = int(source_height / ratio)
    resized_width = int(source_width / ratio)
    resized = Image.fromarray(image, mode="RGB").resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    padded = Image.new("RGB", (target_width, target_height), color=0)
    pad_height = max(0, int((target_height - resized_height) / 2))
    pad_width = max(0, int((target_width - resized_width) / 2))
    padded.paste(resized, (pad_width, pad_height))
    result = np.asarray(padded, dtype=np.uint8)
    if result.shape != IMAGE_SHAPE:
        raise RuntimeError(f"G1 image preprocessing produced {result.shape}")
    return result


def validate_state(state: Any) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"G1 state must have shape ({STATE_DIM},), got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("G1 state contains NaN/Inf")
    for quat_slice in (slice(3, 7), slice(10, 14)):
        error = abs(float(np.linalg.norm(state[quat_slice])) - 1.0)
        if error > QUATERNION_NORM_TOLERANCE:
            raise ValueError(f"G1 state quaternion norm error {error:.6f}")
    low, high = GRIPPER_RANGE_RAD
    if np.any(state[14:16] < low) or np.any(state[14:16] > high):
        raise ValueError("G1 state Dex1 value is outside the contract range")
    return state


def validate_observation(observation: dict[str, Any]) -> dict[str, Any]:
    expected = {*IMAGE_KEYS, STATE_KEY, PROMPT_KEY}
    missing = expected - set(observation)
    if missing:
        raise ValueError(f"G1 observation missing keys: {sorted(missing)}")
    for key in IMAGE_KEYS:
        image = np.asarray(observation[key])
        if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
            raise ValueError(
                f"{key} must be uint8 {IMAGE_SHAPE}, got {image.dtype} {image.shape}"
            )
    validate_state(observation[STATE_KEY])
    prompt = observation[PROMPT_KEY]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("G1 prompt must be a non-empty canonical instruction")
    return observation


def validate_action_chunk(
    actions: Any,
    timestamps: Any | None = None,
    *,
    expected_horizon: int | None = None,
    expected_rate_hz: float | None = None,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or len(actions) < 2:
        raise ValueError(f"G1 actions must have shape (T,{ACTION_DIM}) with T>=2, got {actions.shape}")
    if expected_horizon is not None and len(actions) != expected_horizon:
        raise ValueError(f"G1 action horizon must be {expected_horizon}, got {len(actions)}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("G1 action chunk contains NaN/Inf")
    for quat_slice in (slice(3, 7), slice(10, 14)):
        errors = np.abs(np.linalg.norm(actions[:, quat_slice], axis=1) - 1.0)
        if float(errors.max()) > QUATERNION_NORM_TOLERANCE:
            raise ValueError(f"G1 action quaternion norm error {float(errors.max()):.6f}")
    low, high = GRIPPER_RANGE_RAD
    if np.any(actions[:, 14:16] < low) or np.any(actions[:, 14:16] > high):
        raise ValueError("G1 action Dex1 value is outside the contract range")
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if timestamps.shape != (len(actions),) or not np.all(np.diff(timestamps) > 0):
            raise ValueError("G1 action timestamps must be finite and strictly increasing")
        if not np.all(np.isfinite(timestamps)):
            raise ValueError("G1 action timestamps contain NaN/Inf")
        if expected_rate_hz is not None and not np.allclose(
            np.diff(timestamps), 1.0 / expected_rate_hz, rtol=0.0, atol=1e-6
        ):
            raise ValueError(f"G1 action timestamps must be {expected_rate_hz:g} Hz")
    return actions


def require_verified_contract(metadata: dict[str, Any]) -> None:
    expected = contract_metadata(verified=True)
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Policy is not verified for the frozen G1 EDU contract: {mismatches}")
