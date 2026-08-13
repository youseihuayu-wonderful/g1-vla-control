"""Explicit audit of raw neural 16-D chunks before semantic validation.

Bounded quaternion normalization is produced only as a quarantined analysis
artifact. It does not prove channel order, frame, units, absolute/delta
semantics, EEF site, timing, or simulation/hardware eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from g1_policy_contract import (
    ACTION_DIM,
    ACTION_HORIZON,
    GRIPPER_RANGE_RAD,
    QUATERNION_NORM_TOLERANCE,
    validate_action_chunk,
)


@dataclass(frozen=True)
class NeuralActionAudit:
    raw_actions: np.ndarray
    canonicalized_actions_for_analysis: np.ndarray | None
    finite_shape_passed: bool
    raw_contract_passed: bool
    bounded_quaternion_normalization_passed: bool
    normalization_applied: bool
    raw_max_quaternion_norm_error: float | None
    maximum_quaternion_component_adjustment: float | None
    raw_sha256: str | None
    canonicalized_sha256: str | None
    reasons: tuple[str, ...]
    simulation_eligible: bool = False
    hardware_eligible: bool = False


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def audit_neural_action_chunk(
    actions,
    *,
    expected_horizon: int = ACTION_HORIZON,
    permit_bounded_normalization_for_analysis: bool = True,
    maximum_normalizable_norm_error: float = 0.01,
) -> NeuralActionAudit:
    """Audit raw output and optionally create a quarantined normalized copy.

    The returned canonicalized copy is for semantic comparison only. Both
    simulation_eligible and hardware_eligible deliberately remain false.
    """
    raw = np.asarray(actions, dtype=np.float64)
    reasons: list[str] = []
    finite_shape = bool(
        raw.shape == (expected_horizon, ACTION_DIM)
        and np.all(np.isfinite(raw))
    )
    if not finite_shape:
        if raw.shape != (expected_horizon, ACTION_DIM):
            reasons.append("shape_mismatch")
        if raw.size and not np.all(np.isfinite(raw)):
            reasons.append("non_finite_output")
        return NeuralActionAudit(
            raw, None, False, False, False, False, None, None,
            _sha256(raw) if raw.size else None, None, tuple(reasons),
        )

    low, high = GRIPPER_RANGE_RAD
    grippers_ok = bool(
        np.all(raw[:, 14:16] >= low) and np.all(raw[:, 14:16] <= high)
    )
    if not grippers_ok:
        reasons.append("gripper_outside_contract_range")

    norms = np.concatenate((
        np.linalg.norm(raw[:, 3:7], axis=1),
        np.linalg.norm(raw[:, 10:14], axis=1),
    ))
    maximum_error = float(np.max(np.abs(norms - 1.0)))
    minimum_norm = float(np.min(norms))
    raw_contract_passed = False
    try:
        validate_action_chunk(raw, expected_horizon=expected_horizon)
        raw_contract_passed = True
    except ValueError as exc:
        reasons.append(f"raw_contract:{exc}")

    if raw_contract_passed:
        canonical = raw.copy()
        return NeuralActionAudit(
            raw_actions=raw,
            canonicalized_actions_for_analysis=canonical,
            finite_shape_passed=True,
            raw_contract_passed=True,
            bounded_quaternion_normalization_passed=True,
            normalization_applied=False,
            raw_max_quaternion_norm_error=maximum_error,
            maximum_quaternion_component_adjustment=0.0,
            raw_sha256=_sha256(raw),
            canonicalized_sha256=_sha256(canonical),
            reasons=tuple(reasons),
        )

    bounded = bool(
        permit_bounded_normalization_for_analysis
        and grippers_ok
        and minimum_norm > 1e-6
        and maximum_error <= maximum_normalizable_norm_error
    )
    if not bounded:
        if not permit_bounded_normalization_for_analysis:
            reasons.append("analysis_normalization_not_permitted")
        elif minimum_norm <= 1e-6:
            reasons.append("zero_or_near_zero_quaternion")
        elif maximum_error > maximum_normalizable_norm_error:
            reasons.append("quaternion_error_above_analysis_bound")
        return NeuralActionAudit(
            raw, None, True, False, False, False, maximum_error, None,
            _sha256(raw), None, tuple(reasons),
        )

    canonical = raw.copy()
    for quaternion_slice in (slice(3, 7), slice(10, 14)):
        quaternion_norms = np.linalg.norm(
            canonical[:, quaternion_slice], axis=1, keepdims=True
        )
        canonical[:, quaternion_slice] /= quaternion_norms
    adjustment = float(np.max(np.abs(canonical - raw)))
    try:
        validate_action_chunk(canonical, expected_horizon=expected_horizon)
    except ValueError as exc:
        reasons.append(f"canonicalized_contract:{exc}")
        return NeuralActionAudit(
            raw, None, True, False, False, True, maximum_error, adjustment,
            _sha256(raw), None, tuple(reasons),
        )
    reasons.append("quaternion_normalized_for_quarantined_analysis_only")
    return NeuralActionAudit(
        raw_actions=raw,
        canonicalized_actions_for_analysis=canonical,
        finite_shape_passed=True,
        raw_contract_passed=False,
        bounded_quaternion_normalization_passed=True,
        normalization_applied=True,
        raw_max_quaternion_norm_error=maximum_error,
        maximum_quaternion_component_adjustment=adjustment,
        raw_sha256=_sha256(raw),
        canonicalized_sha256=_sha256(canonical),
        reasons=tuple(reasons),
    )
