"""Fail-closed, context-aware time scaling for frozen G1 16-D action chunks.

This module may change timestamps only. It never changes action samples and it
never authorizes simulation or hardware execution. Runtime safety, semantic
attestation, IK/collision preflight, command filters, watchdog, and E-stop are
separate mandatory gates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from action_schema import EEFActionChunk, LEFT_POS, LEFT_QUAT, RIGHT_POS, RIGHT_QUAT
from safety_governor import MotionEnvelope


_ALLOWED_PHASES = frozenset({
    "free_space", "approach", "grasp", "lift", "place", "retreat",
})


@dataclass(frozen=True)
class AdaptiveSafetyContext:
    """Inputs that must accompany every action sample used by adaptive timing."""

    task_phase: str
    distance_to_goal_m: float
    minimum_clearance_m: float
    eef_tracking_error_m: float
    observation_age_ms: float
    policy_response_age_ms: float
    ik_margin_rad: float
    joint_limit_margin_rad: float
    pelvis_stability: float
    gripper_tracking_error_rad: float = 0.0
    contact: bool = False
    network_timeout: bool = False
    hard_safety_gate_passed: bool = True
    ik_reachable: bool = True
    collision_free: bool = True
    command_limits_passed: bool = True


@dataclass(frozen=True)
class ContextRetimerConfig:
    maximum_observation_age_ms: float = 100.0
    maximum_policy_response_age_ms: float = 500.0
    acceleration_clearance_m: float = 0.10
    hard_minimum_clearance_m: float = 0.005
    acceleration_tracking_error_m: float = 0.015
    hard_tracking_error_m: float = 0.050
    acceleration_gripper_error_rad: float = 0.15
    hard_gripper_error_rad: float = 0.40
    acceleration_margin_rad: float = 0.08
    hard_margin_rad: float = 0.02
    acceleration_stability: float = 0.85
    hard_minimum_stability: float = 0.50
    near_distance_m: float = 0.025
    far_distance_m: float = 0.16
    precision_scale: float = 0.50
    maximum_scale: float = 1.60
    maximum_scale_increase_per_s: float = 2.0
    safety_minimum_scale: float = 0.05
    numerical_tolerance: float = 1e-6
    maximum_limit_iterations: int = 16


@dataclass(frozen=True)
class SpeedDecision:
    hold: bool
    target_scale: float
    acceleration_allowed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ContextRetimingResult:
    accepted: bool
    hold: bool
    chunk: EEFActionChunk | None
    scale_profile: np.ndarray
    decisions: tuple[SpeedDecision, ...]
    metrics: dict[str, float]
    reasons: tuple[str, ...]
    path_actions_byte_identical: bool


def _finite_context(context: AdaptiveSafetyContext) -> bool:
    values = (
        context.distance_to_goal_m,
        context.minimum_clearance_m,
        context.eef_tracking_error_m,
        context.observation_age_ms,
        context.policy_response_age_ms,
        context.ik_margin_rad,
        context.joint_limit_margin_rad,
        context.pelvis_stability,
        context.gripper_tracking_error_rad,
    )
    return bool(np.all(np.isfinite(values)))


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def decide_speed(
    context: AdaptiveSafetyContext,
    config: ContextRetimerConfig | None = None,
) -> SpeedDecision:
    """Return a fail-closed phase/context decision for one action sample."""
    cfg = config or ContextRetimerConfig()
    reasons: list[str] = []
    if not _finite_context(context):
        return SpeedDecision(True, 0.0, False, ("non_finite_context",))
    if context.task_phase not in _ALLOWED_PHASES:
        return SpeedDecision(True, 0.0, False, ("unknown_task_phase",))
    if context.network_timeout:
        return SpeedDecision(True, 0.0, False, ("network_timeout",))
    if context.observation_age_ms > cfg.maximum_observation_age_ms:
        reasons.append("stale_observation")
    if context.policy_response_age_ms > cfg.maximum_policy_response_age_ms:
        reasons.append("stale_policy_response")
    if not context.hard_safety_gate_passed:
        reasons.append("hard_safety_gate_failed")
    if not context.ik_reachable:
        reasons.append("ik_unreachable")
    if not context.collision_free:
        reasons.append("collision_preflight_failed")
    if not context.command_limits_passed:
        reasons.append("command_limits_failed")
    if context.minimum_clearance_m < cfg.hard_minimum_clearance_m:
        reasons.append("clearance_below_hard_minimum")
    if context.eef_tracking_error_m > cfg.hard_tracking_error_m:
        reasons.append("tracking_error_above_hard_maximum")
    if context.gripper_tracking_error_rad > cfg.hard_gripper_error_rad:
        reasons.append("gripper_error_above_hard_maximum")
    if min(context.ik_margin_rad, context.joint_limit_margin_rad) < cfg.hard_margin_rad:
        reasons.append("kinematic_margin_below_hard_minimum")
    if context.pelvis_stability < cfg.hard_minimum_stability:
        reasons.append("pelvis_unstable")
    if reasons:
        return SpeedDecision(True, 0.0, False, tuple(reasons))

    phase_caps = {
        "free_space": cfg.maximum_scale,
        "approach": 0.70,
        "grasp": cfg.precision_scale,
        "lift": 0.80,
        "place": cfg.precision_scale,
        "retreat": 1.00,
    }
    target = phase_caps[context.task_phase]
    acceleration_allowed = context.task_phase == "free_space" and not context.contact

    conservative_conditions = {
        "contact": context.contact,
        "low_clearance": context.minimum_clearance_m < cfg.acceleration_clearance_m,
        "tracking_error": context.eef_tracking_error_m > cfg.acceleration_tracking_error_m,
        "gripper_tracking_error": (
            context.gripper_tracking_error_rad > cfg.acceleration_gripper_error_rad
        ),
        "low_kinematic_margin": (
            min(context.ik_margin_rad, context.joint_limit_margin_rad)
            < cfg.acceleration_margin_rad
        ),
        "reduced_pelvis_stability": context.pelvis_stability < cfg.acceleration_stability,
    }
    for reason, active in conservative_conditions.items():
        if active:
            reasons.append(reason)
            acceleration_allowed = False
    if not acceleration_allowed:
        target = min(target, cfg.precision_scale)
    else:
        ratio = (
            (context.distance_to_goal_m - cfg.near_distance_m)
            / (cfg.far_distance_m - cfg.near_distance_m)
        )
        target = cfg.precision_scale + (
            cfg.maximum_scale - cfg.precision_scale
        ) * _smoothstep(ratio)
    return SpeedDecision(False, float(target), acceleration_allowed, tuple(reasons))


def _geodesic_step(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0, axis=1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
    dot = np.clip(np.abs(np.sum(q0 * q1, axis=1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _motion_metrics(actions: np.ndarray, timestamps: np.ndarray) -> dict[str, float]:
    dt = np.diff(timestamps)
    if np.any(dt <= 0) or not np.all(np.isfinite(dt)):
        raise ValueError("retimed timestamps must be finite and strictly increasing")
    side_velocities = []
    translation_speeds = []
    angular_speeds = []
    for position, quaternion in (
        (LEFT_POS, LEFT_QUAT), (RIGHT_POS, RIGHT_QUAT)
    ):
        velocity = np.diff(actions[:, position], axis=0) / dt[:, None]
        side_velocities.append(velocity)
        translation_speeds.append(np.linalg.norm(velocity, axis=1))
        angular_speeds.append(
            _geodesic_step(actions[:-1, quaternion], actions[1:, quaternion]) / dt
        )
    acceleration_values: list[np.ndarray] = []
    jerk_values: list[np.ndarray] = []
    if len(dt) > 1:
        acceleration_dt = 0.5 * (dt[:-1] + dt[1:])
        for velocity in side_velocities:
            acceleration = np.diff(velocity, axis=0) / acceleration_dt[:, None]
            acceleration_values.append(np.linalg.norm(acceleration, axis=1))
            if len(acceleration) > 1:
                jerk_dt = 0.5 * (acceleration_dt[:-1] + acceleration_dt[1:])
                jerk = np.diff(acceleration, axis=0) / jerk_dt[:, None]
                jerk_values.append(np.linalg.norm(jerk, axis=1))
    gripper_speed = np.abs(np.diff(actions[:, 14:16], axis=0)) / dt[:, None]

    def maximum(arrays: list[np.ndarray]) -> float:
        nonempty = [array for array in arrays if array.size]
        return float(max(np.max(array) for array in nonempty)) if nonempty else 0.0

    return {
        "eef_speed_m_s": maximum(translation_speeds),
        "eef_acceleration_m_s2": maximum(acceleration_values),
        "eef_jerk_m_s3": maximum(jerk_values),
        "angular_speed_rad_s": maximum(angular_speeds),
        "gripper_speed_rad_s": float(np.max(gripper_speed)) if gripper_speed.size else 0.0,
        "duration_s": float(timestamps[-1] - timestamps[0]),
    }


def _limit_ratio(metrics: dict[str, float], envelope: MotionEnvelope) -> float:
    ratios = (
        metrics["eef_speed_m_s"] / envelope.max_eef_speed_m_s,
        np.sqrt(metrics["eef_acceleration_m_s2"] / envelope.max_eef_acceleration_m_s2),
        np.cbrt(metrics["eef_jerk_m_s3"] / envelope.max_eef_jerk_m_s3),
        metrics["angular_speed_rad_s"] / envelope.max_angular_speed_rad_s,
        metrics["gripper_speed_rad_s"] / envelope.max_gripper_speed_rad_s,
    )
    return float(max(ratios))


class ContextAwareRetimer:
    """Retimes a semantically verified chunk, or returns a fail-closed hold."""

    def __init__(
        self,
        config: ContextRetimerConfig | None = None,
        envelope: MotionEnvelope | None = None,
    ):
        self.config = config or ContextRetimerConfig()
        self.envelope = envelope or MotionEnvelope()
        self.previous_scale = 1.0

    def reset(self, scale: float = 1.0) -> None:
        self.previous_scale = float(np.clip(scale, 0.0, self.config.maximum_scale))

    def plan(
        self,
        chunk: EEFActionChunk,
        contexts: AdaptiveSafetyContext | Sequence[AdaptiveSafetyContext],
    ) -> ContextRetimingResult:
        if isinstance(contexts, AdaptiveSafetyContext):
            context_list = [contexts] * len(chunk.timestamps)
        else:
            context_list = list(contexts)
        if len(context_list) != len(chunk.timestamps):
            raise ValueError("contexts must contain one entry per action sample")
        decisions = tuple(decide_speed(context, self.config) for context in context_list)
        hold_reasons = tuple(
            f"sample_{index}:{reason}"
            for index, decision in enumerate(decisions)
            if decision.hold
            for reason in decision.reasons
        )
        if hold_reasons:
            return ContextRetimingResult(
                accepted=False,
                hold=True,
                chunk=None,
                scale_profile=np.zeros(len(chunk.timestamps)),
                decisions=decisions,
                metrics={},
                reasons=hold_reasons,
                path_actions_byte_identical=True,
            )

        nominal_dt = np.diff(chunk.timestamps)
        scale = np.empty(len(chunk.timestamps), dtype=np.float64)
        scale[0] = min(self.previous_scale, decisions[0].target_scale)
        for index in range(1, len(scale)):
            maximum_increase = (
                self.config.maximum_scale_increase_per_s * nominal_dt[index - 1]
            )
            scale[index] = min(
                decisions[index].target_scale,
                scale[index - 1] + maximum_increase,
            )
        scale = np.clip(
            scale, self.config.safety_minimum_scale, self.config.maximum_scale
        )

        metrics: dict[str, float] = {}
        timestamps = chunk.timestamps.copy()
        for _ in range(self.config.maximum_limit_iterations):
            segment_scale = np.maximum(
                0.5 * (scale[:-1] + scale[1:]),
                self.config.safety_minimum_scale,
            )
            timestamps = np.concatenate((
                [chunk.timestamps[0]],
                chunk.timestamps[0] + np.cumsum(nominal_dt / segment_scale),
            ))
            metrics = _motion_metrics(chunk.actions, timestamps)
            ratio = _limit_ratio(metrics, self.envelope)
            if ratio <= 1.0 + self.config.numerical_tolerance:
                break
            candidate = scale / (ratio * 1.001)
            if float(np.min(candidate)) < self.config.safety_minimum_scale:
                return ContextRetimingResult(
                    accepted=False,
                    hold=True,
                    chunk=None,
                    scale_profile=np.zeros(len(scale)),
                    decisions=decisions,
                    metrics=metrics,
                    reasons=("motion_envelope_infeasible_at_safety_minimum_scale",),
                    path_actions_byte_identical=True,
                )
            scale = candidate
        else:
            return ContextRetimingResult(
                accepted=False,
                hold=True,
                chunk=None,
                scale_profile=np.zeros(len(scale)),
                decisions=decisions,
                metrics=metrics,
                reasons=("motion_envelope_iteration_limit",),
                path_actions_byte_identical=True,
            )

        retimed = EEFActionChunk(timestamps, chunk.actions.copy())
        path_identical = bool(np.array_equal(retimed.actions, chunk.actions))
        if not path_identical:
            return ContextRetimingResult(
                accepted=False,
                hold=True,
                chunk=None,
                scale_profile=np.zeros(len(scale)),
                decisions=decisions,
                metrics=metrics,
                reasons=("retimer_changed_action_samples",),
                path_actions_byte_identical=False,
            )
        self.previous_scale = float(scale[-1])
        return ContextRetimingResult(
            accepted=True,
            hold=False,
            chunk=retimed,
            scale_profile=scale,
            decisions=decisions,
            metrics=metrics,
            reasons=(),
            path_actions_byte_identical=True,
        )
