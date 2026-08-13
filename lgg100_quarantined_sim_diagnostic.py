#!/usr/bin/env python3
"""MuJoCo dynamics diagnostic for preflighted, quarantined LGG100 chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from action_schema import EEFActionChunk
from adaptive_speed_context import ContextAwareRetimer
from dex1_gripper import Dex1Controller
from g1_policy_contract import POLICY_RATE_HZ
from g1_sim_speed_context import build_simulation_speed_context
from neural_action_audit import audit_neural_action_chunk
from retiming_safety_validation import _run_scale
from stack_scene import build_model, reset_to_reference_pose, translate_cubes

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--semantic-report", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--cube-x-offset-m", type=float, default=0.0)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "lgg100_quarantined_sim_diagnostic.json",
    )
    parser.add_argument("--allow-quarantined-sim-diagnostic", action="store_true")
    args = parser.parse_args()
    if not args.allow_quarantined_sim_diagnostic:
        raise SystemExit("Explicit diagnostic acknowledgement is required")
    semantics = json.loads(args.semantic_report.read_text())
    preflight = json.loads(args.preflight_report.read_text())
    binding = preflight.get("observation_binding")
    if binding is None or binding.get("accepted") is not True:
        raise ValueError("Preflight does not contain a passing observation binding")
    cube_translation = np.array([args.cube_x_offset_m, 0.0, 0.0])
    if not np.allclose(
        preflight.get("cube_translation_m", [0.0, 0.0, 0.0]),
        cube_translation,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Preflight scene translation does not match simulation")
    if semantics.get("semantic_identification_supported") is not True:
        raise ValueError("Semantic identification is not passing")
    if semantics.get("g1_contract_verified") is not False:
        raise ValueError("This script must not consume a hardware-authorized report")
    if preflight.get("execution_performed") is not False:
        raise ValueError("Expected diagnostic-only preflight evidence")

    with np.load(args.chunks, allow_pickle=False) as payload:
        raw_chunks = np.asarray(payload["actions"], dtype=np.float64)
    if raw_chunks.ndim == 2:
        raw_chunks = raw_chunks[None]

    model = build_model()
    source = mujoco.MjData(model)
    reset_to_reference_pose(model, source)
    translate_cubes(model, source, cube_translation)
    measured_grippers = Dex1Controller(model).motor_states(source)
    records: list[dict] = []
    for index, raw in enumerate(raw_chunks):
        audit = audit_neural_action_chunk(raw)
        actions = audit.canonicalized_actions_for_analysis
        if actions is None:
            records.append({
                "chunk": index,
                "execution_performed": False,
                "candidate_passed": False,
                "reasons": list(audit.reasons),
            })
            continue
        preflight_chunk = preflight["chunks"][index]
        evidence_schedule = tuple(
            preflight_chunk["inferred_phase_schedule"]["phases"]
        )
        required_preflight_phases = {
            {
                "free_space": "free_space",
                "approach": "free_space",
                "grasp": "grasp",
                "lift": "grasp",
                "place": "place",
                "retreat": "free_space",
            }[phase]
            for phase in evidence_schedule
        }
        target_views_passed = all(
            preflight_chunk["phase_views"][phase]["all_accepted"]
            for phase in required_preflight_phases
        )
        swept_view = preflight_chunk["swept_scheduled_view"]
        if not target_views_passed:
            records.append({
                "chunk": index,
                "execution_performed": False,
                "candidate_passed": False,
                "reasons": ["scheduled_target_preflight_failed"],
            })
            continue
        if swept_view.get("accepted") is not True:
            records.append({
                "chunk": index,
                "execution_performed": False,
                "candidate_passed": False,
                "reasons": [
                    f"scheduled_swept_preflight_failed:{swept_view.get('reason')}"
                ],
                "swept_collision_reasons": swept_view.get(
                    "collision_reasons", []
                ),
            })
            continue
        timestamps = np.arange(len(actions), dtype=np.float64) / POLICY_RATE_HZ
        chunk = EEFActionChunk(timestamps, actions)
        # The context builder uses live geometry/contact and controller state.
        # A new chunk's first target displacement is command jump, not live
        # controller tracking error. The jerk-limited EEF filter owns that jump;
        # this offline shadow starts with no outstanding controller command.
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pelvis_position = source.xpos[pelvis]
        pelvis_rotation = source.xmat[pelvis].reshape(3, 3)
        current_positions = []
        for side in ("left", "right"):
            site = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_eef"
            )
            current_positions.append(
                pelvis_rotation.T @ (source.site_xpos[site] - pelvis_position)
            )
        initial_jump = float(max(
            np.linalg.norm(actions[0, 0:3] - current_positions[0]),
            np.linalg.norm(actions[0, 7:10] - current_positions[1]),
        ))
        contexts = []
        context_evidence = []
        for action_index, action in enumerate(actions):
            context, evidence = build_simulation_speed_context(
                model,
                source,
                commanded_grippers_rad=action[14:16],
                measured_grippers_rad=measured_grippers,
                eef_tracking_error_m=0.0,
                observation_age_ms=20.0,
                policy_response_age_ms=90.0,
                preflight_passed=True,
                collision_free=True,
                command_limits_passed=True,
                gripper_tracking_error_rad=(
                    float(np.max(np.abs(action[14:16] - measured_grippers)))
                    if action_index == 0 else 0.0
                ),
            )
            contexts.append(context)
            context_evidence.append(evidence)
        phase_schedule = tuple(context.task_phase for context in contexts)
        if phase_schedule != evidence_schedule:
            records.append({
                "chunk": index,
                "execution_performed": False,
                "candidate_passed": False,
                "reasons": ["phase_schedule_does_not_match_preflight_evidence"],
            })
            continue
        context = contexts[0]
        evidence = context_evidence[0]
        retiming = ContextAwareRetimer().plan(chunk, contexts)
        if not retiming.accepted or retiming.chunk is None:
            records.append({
                "chunk": index,
                "execution_performed": False,
                "candidate_passed": False,
                "reasons": list(retiming.reasons),
            })
            continue

        baseline = _run_scale(
            chunk, measured_grippers, scale=1.0,
            use_filter=True, use_joint_filter=True,
            phase_schedule=phase_schedule,
            abort_on_phase_aware_contact=True,
            cube_translation_m=cube_translation,
        )
        guarded = _run_scale(
            retiming.chunk, measured_grippers, scale=1.0,
            use_filter=True, use_joint_filter=True,
            phase_schedule=phase_schedule,
            abort_on_phase_aware_contact=True,
            cube_translation_m=cube_translation,
        )
        candidate_passed = bool(
            baseline["hard_command_limits_pass"]
            and guarded["hard_command_limits_pass"]
            and guarded["finite"]
            and not baseline["aborted_on_phase_aware_contact"]
            and not guarded["aborted_on_phase_aware_contact"]
            and guarded["endpoint_error_m"] <= baseline["endpoint_error_m"] + 0.005
            and guarded["phase_aware_contact_step_rate"] == 0.0
            and guarded["maxima"]["actual_joint_jerk_rad_s3"]
            <= baseline["maxima"]["actual_joint_jerk_rad_s3"] + 1e-6
        )
        records.append({
            "chunk": index,
            "raw_sha256": audit.raw_sha256,
            "execution_performed": True,
            "candidate_passed": candidate_passed,
            "initial_target_jump_m": initial_jump,
            "context": {
                "task_phase": context.task_phase,
                "minimum_clearance_m": context.minimum_clearance_m,
                "contact": context.contact,
                "phase_schedule": list(phase_schedule),
                "phase_counts": {
                    phase: phase_schedule.count(phase)
                    for phase in sorted(set(phase_schedule))
                },
                "joint_limit_margin_rad": evidence.joint_limit_margin_rad,
                "pelvis_stability": evidence.pelvis_stability,
            },
            "retiming": {
                "scale_range": [
                    float(retiming.scale_profile.min()),
                    float(retiming.scale_profile.max()),
                ],
                "path_actions_byte_identical": (
                    retiming.path_actions_byte_identical
                ),
                "metrics": retiming.metrics,
            },
            "baseline": baseline,
            "guarded": guarded,
            "comparison": {
                "duration_delta_s": (
                    guarded["simulated_duration_s"]
                    - baseline["simulated_duration_s"]
                ),
                "endpoint_error_delta_m": (
                    guarded["endpoint_error_m"]
                    - baseline["endpoint_error_m"]
                ),
                "free_space_contact_rate_delta": (
                    guarded["free_space_contact_step_rate"]
                    - baseline["free_space_contact_step_rate"]
                ),
                "phase_aware_contact_rate_delta": (
                    guarded["phase_aware_contact_step_rate"]
                    - baseline["phase_aware_contact_step_rate"]
                ),
                "actual_joint_jerk_delta_rad_s3": (
                    guarded["maxima"]["actual_joint_jerk_rad_s3"]
                    - baseline["maxima"]["actual_joint_jerk_rad_s3"]
                ),
            },
        })

    executed = [record for record in records if record["execution_performed"]]
    report = {
        "scope": "Quarantined single-chunk MuJoCo dynamics diagnostic; never G1 hardware and not task success.",
        "cube_translation_m": cube_translation.tolist(),
        "source_chunks_sha256": hashlib.sha256(args.chunks.read_bytes()).hexdigest(),
        "semantic_report_sha256": hashlib.sha256(
            args.semantic_report.read_bytes()
        ).hexdigest(),
        "preflight_report_sha256": hashlib.sha256(
            args.preflight_report.read_bytes()
        ).hexdigest(),
        "semantic_identification_supported": True,
        "g1_contract_verified": False,
        "g1_sim_eligible": False,
        "g1_execution_enabled": False,
        "task_success_claimed": False,
        "records": records,
        "summary": {
            "chunks": len(records),
            "executed_diagnostic_chunks": len(executed),
            "candidate_passes": sum(
                record["candidate_passed"] for record in records
            ),
            "all_executed_candidates_passed": bool(
                executed and all(record["candidate_passed"] for record in executed)
            ),
        },
        "verdict": "This result measures guarded mechanics for local approach chunks only. It does not prove acceleration coverage or block-stacking success.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(json.dumps(report["summary"], indent=2))
    if not report["summary"]["all_executed_candidates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
