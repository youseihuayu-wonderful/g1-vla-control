#!/usr/bin/env python3
"""Mock-only fail-closed decision adapter for G1 Policy Shadow.

There is intentionally no Unitree import, publisher, controller, mode switch,
or hardware transport in this module. It can record suggestions and hold
reasons, but it cannot send a robot command.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ShadowSafetyThresholds:
    lowstate_stale_ms: float = 50.0
    camera_stale_ms: float = 100.0
    policy_timeout_ms: float = 100.0
    maximum_waist_divergence_m: float = 0.005
    maximum_waist_divergence_deg: float = 3.0


@dataclass(frozen=True)
class ShadowSafetyInputs:
    lowstate_age_ms: float
    unique_tick_age_ms: float
    camera_age_ms: dict[str, float]
    camera_frozen: dict[str, bool]
    policy_latency_ms: float
    canonical_action: np.ndarray
    ik_success: bool
    collision_free: bool
    joint_limits_ok: bool
    dds_connected: bool
    waist_divergence_m: float
    waist_divergence_deg: float


class MockSuggestionSink:
    """In-memory evidence sink; it has no command or network method."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        suggested_joint_target: np.ndarray,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        target = np.ascontiguousarray(suggested_joint_target, dtype=np.float64)
        record = {
            "suggested_joint_target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
            "suggested_joint_target_shape": list(target.shape),
            "suggested_joint_target_finite": bool(np.all(np.isfinite(target))),
            "decision": decision,
            "mock_sink_only": True,
            "robot_command_sent": False,
        }
        self.records.append(record)
        return record


class FailClosedMockAdapter:
    """Evaluate safety inputs and record, never publish, a suggestion."""

    hardware_transport_present = False
    publisher_created = False
    robot_command_sent = False

    def __init__(
        self,
        sink: MockSuggestionSink,
        thresholds: ShadowSafetyThresholds = ShadowSafetyThresholds(),
    ) -> None:
        self.sink = sink
        self.thresholds = thresholds

    def evaluate(self, inputs: ShadowSafetyInputs) -> dict[str, Any]:
        reasons: list[str] = []
        if not inputs.dds_connected:
            reasons.append("dds_disconnected")
        if (
            not np.isfinite(inputs.lowstate_age_ms)
            or not np.isfinite(inputs.unique_tick_age_ms)
        ):
            reasons.append("lowstate_age_nonfinite")
        elif (
            inputs.lowstate_age_ms >= self.thresholds.lowstate_stale_ms
            or inputs.unique_tick_age_ms >= self.thresholds.lowstate_stale_ms
        ):
            reasons.append("lowstate_stale")

        expected_cameras = {"head_left", "left_wrist", "right_wrist"}
        if set(inputs.camera_age_ms) != expected_cameras:
            reasons.append("camera_set_invalid")
        else:
            for name in sorted(expected_cameras):
                age = inputs.camera_age_ms[name]
                if not np.isfinite(age) or age >= self.thresholds.camera_stale_ms:
                    reasons.append(f"camera_stale:{name}")
                if inputs.camera_frozen.get(name, True):
                    reasons.append(f"camera_frozen:{name}")

        if (
            not np.isfinite(inputs.policy_latency_ms)
            or inputs.policy_latency_ms >= self.thresholds.policy_timeout_ms
        ):
            reasons.append("policy_timeout")
        action = np.asarray(inputs.canonical_action)
        if action.shape != (32, 16):
            reasons.append("canonical_action_shape_invalid")
        if not np.all(np.isfinite(action)):
            reasons.append("action_nonfinite")
        if not inputs.ik_success:
            reasons.append("ik_failure")
        if not inputs.collision_free:
            reasons.append("collision_failure")
        if not inputs.joint_limits_ok:
            reasons.append("joint_limits_failure")
        if (
            not np.isfinite(inputs.waist_divergence_m)
            or not np.isfinite(inputs.waist_divergence_deg)
            or inputs.waist_divergence_m
            > self.thresholds.maximum_waist_divergence_m
            or inputs.waist_divergence_deg
            > self.thresholds.maximum_waist_divergence_deg
        ):
            reasons.append("waist_divergence")

        safety_gate_passed = not reasons
        # Mock-only is a separate, unconditional barrier. Even a nominal safety
        # result cannot become hardware authority in this adapter.
        publish_allowed = bool(safety_gate_passed and self.hardware_transport_present)
        return {
            "safety_gate_passed": safety_gate_passed,
            "reasons": reasons,
            "hold": not safety_gate_passed,
            "publish_allowed": publish_allowed,
            "hardware_transport_present": self.hardware_transport_present,
            "publisher_created": self.publisher_created,
            "robot_command_sent": self.robot_command_sent,
        }

    def commit_to_mock(
        self,
        suggested_joint_target: np.ndarray,
        inputs: ShadowSafetyInputs,
    ) -> dict[str, Any]:
        decision = self.evaluate(inputs)
        if decision["publish_allowed"]:
            raise RuntimeError("mock-only adapter must never allow publish")
        return self.sink.record(suggested_joint_target, decision)
