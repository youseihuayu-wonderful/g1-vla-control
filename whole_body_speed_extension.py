#!/usr/bin/env python3
"""Whole-body speed-extension gate for squat/reach scenarios.

This module extends the timestamp-only arm scheduler into a whole-body *policy
contract*: UPRIGHT can use the existing arm speed schedule, while LOWER/REACH/
RECOVER/RISE are conservative or hold-only until RL balance, torque margins,
physical EEF parity, DDS freshness, and H1-H6 authorization are complete.

It never imports Unitree SDK, never creates a publisher, and never sends robot
commands. The output is an execution-locked shadow/design artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

BODY_PHASES = ("UPRIGHT", "LOWER", "REACH", "RECOVER", "RISE")
ARM_PHASES = ("free_space", "approach", "grasp", "lift", "place", "retreat")


@dataclass(frozen=True)
class WholeBodySpeedConfig:
    upright_free_space_scale: float = 1.20
    upright_task_scale: float = 1.00
    lower_scale: float = 0.55
    reach_scale: float = 0.45
    recover_scale: float = 0.40
    rise_scale: float = 0.35
    payload_scale_cap: float = 0.30
    minimum_body_height_m: float = 0.60
    maximum_body_pitch_abs_rad: float = 0.25
    hard_minimum_torque_margin: float = 0.20
    hard_minimum_pelvis_stability: float = 0.70


@dataclass(frozen=True)
class WholeBodyContext:
    body_phase: str
    arm_task_phase: str
    rl_controller_active: bool
    support_confirmed: bool
    estop_ready: bool
    h_gates_resolved: int
    h_gates_total: int
    torque_margin: float
    pelvis_stability: float
    body_height_m: float
    body_pitch_rad: float
    payload_estimated_kg: float = 0.0
    contact: bool = False
    body_phase_tracking_error: float = 0.0
    network_timeout: bool = False


@dataclass(frozen=True)
class WholeBodyDecision:
    hold: bool
    target_scale: float
    reasons: tuple[str, ...]
    body_phase: str
    arm_task_phase: str
    execution_authorized: bool


def _finite_context(context: WholeBodyContext) -> bool:
    values = (
        context.torque_margin,
        context.pelvis_stability,
        context.body_height_m,
        context.body_pitch_rad,
        context.payload_estimated_kg,
        context.body_phase_tracking_error,
    )
    return bool(np.all(np.isfinite(values)))


def nominal_phase_scale(
    body_phase: str,
    arm_task_phase: str,
    *,
    payload_estimated_kg: float = 0.0,
    config: WholeBodySpeedConfig = WholeBodySpeedConfig(),
) -> float:
    if body_phase not in BODY_PHASES:
        raise ValueError(f"unknown body phase: {body_phase}")
    if arm_task_phase not in ARM_PHASES:
        raise ValueError(f"unknown arm task phase: {arm_task_phase}")
    if body_phase == "UPRIGHT":
        scale = config.upright_free_space_scale if arm_task_phase == "free_space" else config.upright_task_scale
    elif body_phase == "LOWER":
        scale = config.lower_scale
    elif body_phase == "REACH":
        scale = config.reach_scale
    elif body_phase == "RECOVER":
        scale = config.recover_scale
    elif body_phase == "RISE":
        scale = config.rise_scale
    else:  # pragma: no cover; guarded above.
        raise AssertionError(body_phase)
    if payload_estimated_kg > 0.0:
        scale = min(scale, config.payload_scale_cap)
    return float(scale)


def decide_whole_body_speed(
    context: WholeBodyContext,
    config: WholeBodySpeedConfig = WholeBodySpeedConfig(),
) -> WholeBodyDecision:
    reasons: list[str] = []
    if not _finite_context(context):
        return WholeBodyDecision(
            hold=True,
            target_scale=0.0,
            reasons=("non_finite_context",),
            body_phase=context.body_phase,
            arm_task_phase=context.arm_task_phase,
            execution_authorized=False,
        )
    if context.body_phase not in BODY_PHASES:
        reasons.append("unknown_body_phase")
    if context.arm_task_phase not in ARM_PHASES:
        reasons.append("unknown_arm_phase")
    if context.network_timeout:
        reasons.append("network_timeout")
    if not context.support_confirmed:
        reasons.append("support_not_confirmed")
    if not context.estop_ready:
        reasons.append("estop_not_ready")
    if not context.rl_controller_active:
        reasons.append("rl_controller_not_active")
    if context.h_gates_resolved < context.h_gates_total:
        reasons.append("h_gates_unresolved")
    if context.torque_margin < config.hard_minimum_torque_margin:
        reasons.append("torque_margin_below_minimum")
    if context.pelvis_stability < config.hard_minimum_pelvis_stability:
        reasons.append("pelvis_stability_below_minimum")
    if context.body_height_m < config.minimum_body_height_m:
        reasons.append("body_height_below_minimum")
    if abs(context.body_pitch_rad) > config.maximum_body_pitch_abs_rad:
        reasons.append("body_pitch_outside_limit")
    if context.body_phase_tracking_error > 0.04:
        reasons.append("body_phase_tracking_error_above_limit")
    if context.contact and context.body_phase in {"LOWER", "REACH", "RISE"}:
        reasons.append("contact_during_balance_critical_phase")

    if reasons:
        return WholeBodyDecision(
            hold=True,
            target_scale=0.0,
            reasons=tuple(reasons),
            body_phase=context.body_phase,
            arm_task_phase=context.arm_task_phase,
            execution_authorized=False,
        )

    scale = nominal_phase_scale(
        context.body_phase,
        context.arm_task_phase,
        payload_estimated_kg=context.payload_estimated_kg,
        config=config,
    )
    return WholeBodyDecision(
        hold=False,
        target_scale=scale,
        reasons=(),
        body_phase=context.body_phase,
        arm_task_phase=context.arm_task_phase,
        execution_authorized=True,
    )


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def build_nominal_policy(config: WholeBodySpeedConfig) -> list[dict[str, Any]]:
    sequence = [
        ("UPRIGHT", "free_space", 0.75, 0.00, "stable standing / free-space arm motion"),
        ("LOWER", "approach", 0.63, 0.00, "height transition; no acceleration"),
        ("REACH", "approach", 0.63, 0.20, "forward lean; precision and balance critical"),
        ("REACH", "grasp", 0.63, 0.20, "near-object grasp; gripper/contact critical"),
        ("RECOVER", "lift", 0.63, 0.00, "recover pitch before standing"),
        ("RISE", "lift", 0.75, 0.00, "payload-aware stand-up"),
    ]
    return [
        {
            "body_phase": body_phase,
            "arm_task_phase": arm_phase,
            "nominal_body_height_m": height,
            "nominal_body_pitch_rad": pitch,
            "nominal_scale_no_payload": nominal_phase_scale(
                body_phase, arm_phase, config=config
            ),
            "nominal_scale_with_payload": nominal_phase_scale(
                body_phase, arm_phase, payload_estimated_kg=0.2, config=config
            ),
            "description": description,
        }
        for body_phase, arm_phase, height, pitch, description in sequence
    ]


def build_whole_body_report(
    *,
    low_risk_path: Path = RESULTS / "g1_speed_low_risk_hardware_check.json",
    offline_replay_path: Path = RESULTS / "offline_replay_comparison.json",
    six_gate_path: Path = RESULTS / "g1_six_gate_execution_status_20260824.json",
    output: Path = RESULTS / "whole_body_speed_extension.json",
    config: WholeBodySpeedConfig = WholeBodySpeedConfig(),
) -> dict[str, Any]:
    low_risk = load_json(low_risk_path)
    offline_replay = load_json(offline_replay_path)
    six_gate = load_json(six_gate_path)

    resolved = int(six_gate.get("resolved_gate_count", 0))
    total = int(six_gate.get("total_gate_count", 6))
    support = bool(six_gate.get("safety_baseline", {}).get("mechanical_support_confirmed_by_operator"))
    estop = bool(six_gate.get("safety_baseline", {}).get("estop_ready_confirmed_by_operator"))

    contexts = [
        WholeBodyContext(
            body_phase=item["body_phase"],
            arm_task_phase=item["arm_task_phase"],
            rl_controller_active=False,
            support_confirmed=support,
            estop_ready=estop,
            h_gates_resolved=resolved,
            h_gates_total=total,
            torque_margin=0.0,
            pelvis_stability=0.0,
            body_height_m=float(item["nominal_body_height_m"]),
            body_pitch_rad=float(item["nominal_body_pitch_rad"]),
            payload_estimated_kg=0.2 if item["body_phase"] == "RISE" else 0.0,
            contact=item["arm_task_phase"] == "grasp",
        )
        for item in build_nominal_policy(config)
    ]
    decisions = [asdict(decide_whole_body_speed(context, config)) for context in contexts]
    unresolved_reasons = sorted({
        reason for decision in decisions for reason in decision["reasons"]
    })

    report = {
        "schema_version": "g1_whole_body_speed_extension_v1",
        "scope": (
            "Shadow/design extension from standing dual-arm speed scheduling to "
            "whole-body squat/reach phases. No RL command, no Publisher, no robot motion."
        ),
        "inputs": {
            "low_risk": str(low_risk_path),
            "offline_replay": str(offline_replay_path),
            "six_gate": str(six_gate_path),
        },
        "config": asdict(config),
        "nominal_phase_policy": build_nominal_policy(config),
        "execution_locked_phase_decisions": decisions,
        "speed_shadow_summary": {
            "arm_only_duration_reduction_fraction": offline_replay.get("summary", {}).get(
                "total_duration_reduction_fraction"
            ),
            "arm_only_replay_frame_reduction_fraction": offline_replay.get("summary", {}).get(
                "fixed_rate_replay_frame_reduction_fraction"
            ),
            "risk_accelerated_episode_count": offline_replay.get("summary", {}).get(
                "risk_accelerated_episode_count"
            ),
        },
        "gate_state": {
            "current_gate": six_gate.get("current_gate"),
            "resolved_gate_count": resolved,
            "total_gate_count": total,
            "low_risk_read_only_shadow_ready": low_risk.get("decision", {}).get(
                "read_only_shadow_ready"
            ),
            "publisher_created": six_gate.get("safety_baseline", {}).get("publisher_created"),
            "robot_command_sent": six_gate.get("safety_baseline", {}).get("robot_command_sent"),
            "hardware_execution_performed": six_gate.get("safety_baseline", {}).get(
                "hardware_execution_performed"
            ),
        },
        "missing_whole_body_prerequisites": {
            "rl_balance_controller_live_and_logged": False,
            "torque_margin_stream_available": False,
            "pelvis_stability_metric_live": False,
            "whole_body_collision_model_physical_parity": False,
            "squat_reach_real_shadow_dataset_available": False,
            "h1_to_h5_motion_prerequisites_passed": resolved >= 5,
            "h6_single_motion_authorization_present": False,
        },
        "decision": {
            "whole_body_extension_completed": True,
            "shadow_design_ready": True,
            "whole_body_execution_ready": False,
            "all_execution_decisions_hold": all(decision["hold"] for decision in decisions),
            "execution_lock_reasons": unresolved_reasons,
            "publisher_created": False,
            "robot_command_sent": False,
            "hardware_execution_performed": False,
            "whole_body_motion_allowed": False,
            "robot_motion_allowed": False,
            "reason": (
                "Whole-body speed policy is defined for shadow/design review only. "
                "Execution is locked until live RL stability, torque margins, full H1-H5 gates, "
                "and H6 single-motion authorization exist."
            ),
        },
    }
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-risk", type=Path, default=RESULTS / "g1_speed_low_risk_hardware_check.json")
    parser.add_argument("--offline-replay", type=Path, default=RESULTS / "offline_replay_comparison.json")
    parser.add_argument("--six-gate", type=Path, default=RESULTS / "g1_six_gate_execution_status_20260824.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "whole_body_speed_extension.json")
    args = parser.parse_args()
    report = build_whole_body_report(
        low_risk_path=args.low_risk,
        offline_replay_path=args.offline_replay,
        six_gate_path=args.six_gate,
        output=args.output,
    )
    print(args.output)
    print(json.dumps({
        "shadow_design_ready": report["decision"]["shadow_design_ready"],
        "whole_body_execution_ready": report["decision"]["whole_body_execution_ready"],
        "all_execution_decisions_hold": report["decision"]["all_execution_decisions_hold"],
        "current_gate": report["gate_state"]["current_gate"],
        "robot_motion_allowed": report["decision"]["robot_motion_allowed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
