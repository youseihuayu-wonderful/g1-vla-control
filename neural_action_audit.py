"""Audit raw LGG100 actions and apply the documented consumer boundary.

Yuhao's pinned ``UnitreeG1EEFOutputs`` transform explicitly states that neural
quaternions are not guaranteed to be unit norm and that consumers should
normalize ``q[3:7]`` and ``q[10:14]`` before IK. This module preserves and
hashes the raw chunk, performs only that bounded quaternion projection, and
validates the resulting canonical action. It does not by itself qualify IK,
collision, timing, simulation dynamics, or hardware execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from g1_policy_contract import (
    ACTION_DIM,
    ACTION_HORIZON,
    MAXIMUM_RAW_QUATERNION_NORM_ERROR,
    QUATERNION_NORM_TOLERANCE,
    canonicalize_policy_action_chunk,
    validate_action_chunk,
)


@dataclass(frozen=True)
class NeuralActionAudit:
    raw_actions: np.ndarray
    official_postprocessed_actions: np.ndarray | None
    finite_shape_passed: bool
    raw_quaternion_exact_unit_passed: bool
    official_consumer_postprocess_passed: bool
    normalization_applied: bool
    raw_max_quaternion_norm_error: float | None
    maximum_quaternion_component_adjustment: float | None
    raw_sha256: str | None
    official_postprocessed_sha256: str | None
    reasons: tuple[str, ...]
    simulation_eligible: bool = False
    hardware_eligible: bool = False

    # Compatibility aliases keep historical readers functional while making
    # clear that these old names must not be interpreted as the model contract.
    @property
    def canonicalized_actions_for_analysis(self) -> np.ndarray | None:
        return self.official_postprocessed_actions

    @property
    def raw_contract_passed(self) -> bool:
        return self.raw_quaternion_exact_unit_passed

    @property
    def bounded_quaternion_normalization_passed(self) -> bool:
        return self.official_consumer_postprocess_passed

    @property
    def canonicalized_sha256(self) -> str | None:
        return self.official_postprocessed_sha256


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def audit_neural_action_chunk(
    actions,
    *,
    expected_horizon: int = ACTION_HORIZON,
    permit_official_consumer_postprocessing: bool = True,
    maximum_normalizable_norm_error: float = MAXIMUM_RAW_QUATERNION_NORM_ERROR,
) -> NeuralActionAudit:
    """Audit raw output and apply the pinned official consumer projection.

    A passing post-processing result establishes only the canonical action
    boundary. ``simulation_eligible`` and ``hardware_eligible`` remain false
    until their independent gates pass.
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
            raw_actions=raw,
            official_postprocessed_actions=None,
            finite_shape_passed=False,
            raw_quaternion_exact_unit_passed=False,
            official_consumer_postprocess_passed=False,
            normalization_applied=False,
            raw_max_quaternion_norm_error=None,
            maximum_quaternion_component_adjustment=None,
            raw_sha256=_sha256(raw) if raw.size else None,
            official_postprocessed_sha256=None,
            reasons=tuple(reasons),
        )

    norms = np.concatenate((
        np.linalg.norm(raw[:, 3:7], axis=1),
        np.linalg.norm(raw[:, 10:14], axis=1),
    ))
    maximum_error = float(np.max(np.abs(norms - 1.0)))
    raw_exact_unit = bool(maximum_error <= QUATERNION_NORM_TOLERANCE)
    if not raw_exact_unit:
        reasons.append("raw_quaternion_not_exactly_unit_before_required_postprocessing")

    if not permit_official_consumer_postprocessing:
        reasons.append("official_consumer_postprocessing_not_permitted")
        return NeuralActionAudit(
            raw_actions=raw,
            official_postprocessed_actions=None,
            finite_shape_passed=True,
            raw_quaternion_exact_unit_passed=raw_exact_unit,
            official_consumer_postprocess_passed=False,
            normalization_applied=False,
            raw_max_quaternion_norm_error=maximum_error,
            maximum_quaternion_component_adjustment=None,
            raw_sha256=_sha256(raw),
            official_postprocessed_sha256=None,
            reasons=tuple(reasons),
        )

    try:
        canonical = canonicalize_policy_action_chunk(
            raw,
            expected_horizon=expected_horizon,
            maximum_quaternion_norm_error=maximum_normalizable_norm_error,
        )
        validate_action_chunk(canonical, expected_horizon=expected_horizon)
    except ValueError as exc:
        reasons.append(f"official_consumer_postprocessing_rejected:{exc}")
        return NeuralActionAudit(
            raw_actions=raw,
            official_postprocessed_actions=None,
            finite_shape_passed=True,
            raw_quaternion_exact_unit_passed=raw_exact_unit,
            official_consumer_postprocess_passed=False,
            normalization_applied=False,
            raw_max_quaternion_norm_error=maximum_error,
            maximum_quaternion_component_adjustment=None,
            raw_sha256=_sha256(raw),
            official_postprocessed_sha256=None,
            reasons=tuple(reasons),
        )

    adjustment = float(np.max(np.abs(canonical - raw)))
    normalization_applied = not np.array_equal(canonical, raw)
    if normalization_applied:
        reasons.append("official_consumer_quaternion_normalization_applied")
    return NeuralActionAudit(
        raw_actions=raw,
        official_postprocessed_actions=canonical,
        finite_shape_passed=True,
        raw_quaternion_exact_unit_passed=raw_exact_unit,
        official_consumer_postprocess_passed=True,
        normalization_applied=normalization_applied,
        raw_max_quaternion_norm_error=maximum_error,
        maximum_quaternion_component_adjustment=adjustment,
        raw_sha256=_sha256(raw),
        official_postprocessed_sha256=_sha256(canonical),
        reasons=tuple(reasons),
    )
