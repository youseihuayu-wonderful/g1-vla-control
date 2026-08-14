#!/usr/bin/env python3
"""Diagnostic-only preflight for quarantined real LGG100 chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from action_schema import EEFActionChunk, pelvis_vla_action_to_world_mujoco
from dex1_gripper import Dex1Controller
from g1_mujoco_bridge import policy_state_from_mujoco
from g1_policy_contract import ACTION_DIM, ACTION_HORIZON, POLICY_RATE_HZ
from g1_sim_speed_context import build_simulation_speed_context
from neural_action_audit import audit_neural_action_chunk
from safety_governor import G1TargetPreflight
from stack_scene import build_model, reset_to_reference_pose, translate_cubes
from swept_path_preflight import G1SweptPathPreflight

ROOT = Path(__file__).resolve().parent


def _load_chunks(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        for key in ("actions", "raw_actions"):
            if key in payload:
                chunks = np.asarray(payload[key], dtype=np.float64)
                break
        else:
            raise ValueError("NPZ must contain actions or raw_actions")
    if chunks.ndim == 2:
        chunks = chunks[None, ...]
    if chunks.ndim != 3 or chunks.shape[1:] != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(
            f"Expected [N,{ACTION_HORIZON},{ACTION_DIM}], got {chunks.shape}"
        )
    return chunks


def _chunk_motion(actions: np.ndarray) -> dict[str, float]:
    dt = 1.0 / POLICY_RATE_HZ
    left_speed = np.linalg.norm(np.diff(actions[:, 0:3], axis=0), axis=1) / dt
    right_speed = np.linalg.norm(np.diff(actions[:, 7:10], axis=0), axis=1) / dt
    first_jump = max(
        np.linalg.norm(actions[0, 0:3]),
        np.linalg.norm(actions[0, 7:10]),
    )
    return {
        "maximum_inter_sample_translation_speed_m_s": float(
            max(np.max(left_speed), np.max(right_speed))
        ),
        "maximum_absolute_position_norm_m": float(
            max(
                np.max(np.linalg.norm(actions[:, 0:3], axis=1)),
                np.max(np.linalg.norm(actions[:, 7:10], axis=1)),
            )
        ),
        "first_absolute_position_norm_m": float(first_jump),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--semantic-report", type=Path, required=True)
    parser.add_argument("--cube-x-offset-m", type=float, default=0.0)
    parser.add_argument("--observation", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "lgg100_quarantined_preflight.json",
    )
    args = parser.parse_args()
    semantics = json.loads(args.semantic_report.read_text())
    if semantics.get("semantic_identification_supported") is not True:
        raise ValueError("Semantic identification report is not passing")
    if semantics.get("g1_contract_verified") is not False:
        raise ValueError("This diagnostic expects the contract to remain unverified")

    raw_chunks = _load_chunks(args.chunks)
    model = build_model()
    source = mujoco.MjData(model)
    reset_to_reference_pose(model, source)
    cube_translation = np.array([args.cube_x_offset_m, 0.0, 0.0])
    translate_cubes(model, source, cube_translation)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    gate = G1TargetPreflight(model)
    swept_gate = G1SweptPathPreflight(model)
    measured_grippers = Dex1Controller(model).motor_states(source)
    source_state = policy_state_from_mujoco(
        model, source, measured_grippers
    ).astype(np.float64)
    observation_binding = None
    if args.observation is not None:
        with np.load(args.observation, allow_pickle=False) as payload:
            observation_state = np.asarray(payload["state"], dtype=np.float64)
            observation_translation = np.asarray(
                payload["cube_translation_m"], dtype=np.float64
            )
        state_error = float(np.max(np.abs(observation_state - source_state)))
        translation_matches = bool(np.allclose(
            observation_translation, cube_translation, rtol=0.0, atol=1e-12
        ))
        observation_binding = {
            "path": str(args.observation),
            "sha256": hashlib.sha256(args.observation.read_bytes()).hexdigest(),
            "maximum_state_error": state_error,
            "cube_translation_matches": translation_matches,
            "accepted": bool(state_error <= 1e-6 and translation_matches),
        }
        if not observation_binding["accepted"]:
            raise ValueError("Observation state/scene does not match preflight source")
    chunk_records: list[dict] = []
    for chunk_index, raw in enumerate(raw_chunks):
        audit = audit_neural_action_chunk(raw)
        analysis = audit.canonicalized_actions_for_analysis
        record = {
            "chunk": chunk_index,
            "raw_sha256": audit.raw_sha256,
            "bounded_analysis_available": analysis is not None,
            "normalization_applied_for_analysis": audit.normalization_applied,
            "raw_max_quaternion_norm_error": audit.raw_max_quaternion_norm_error,
            "motion": None,
            "phase_views": {},
            "swept_path_views": {},
            "inferred_phase_schedule": None,
            "swept_scheduled_view": None,
        }
        if analysis is None:
            record["reasons"] = list(audit.reasons)
            chunk_records.append(record)
            continue
        record["motion"] = _chunk_motion(analysis)
        for phase in ("free_space", "grasp", "place"):
            targets = []
            for index, action in enumerate(analysis):
                world = pelvis_vla_action_to_world_mujoco(
                    action, source.xpos[pelvis], source.xquat[pelvis]
                )
                result = gate.check(source, world, phase=phase)
                targets.append({
                    "index": index,
                    "accepted": result.accepted,
                    "reachable": result.reachable,
                    "collision_free": result.collision_free,
                    "joint_limits_ok": result.joint_limits_ok,
                    "position_error_m": result.position_error_m,
                    "orientation_error_rad": result.orientation_error_rad,
                    "reason": result.reason,
                    "collision_reasons": list(result.collision_reasons),
                })
            record["phase_views"][phase] = {
                "accepted": sum(target["accepted"] for target in targets),
                "checked": len(targets),
                "all_accepted": all(target["accepted"] for target in targets),
                "reason_counts": {
                    reason: sum(target["reason"] == reason for target in targets)
                    for reason in sorted({target["reason"] for target in targets})
                },
                "targets": targets,
            }
            timestamps = np.arange(len(analysis), dtype=np.float64) / POLICY_RATE_HZ
            swept = swept_gate.check(
                source, EEFActionChunk(timestamps, analysis), phase=phase
            )
            record["swept_path_views"][phase] = {
                "accepted": swept.accepted,
                "checked_targets": swept.checked_targets,
                "checked_interpolated_configurations": (
                    swept.checked_interpolated_configurations
                ),
                "maximum_position_error_m": swept.maximum_position_error_m,
                "maximum_orientation_error_rad": (
                    swept.maximum_orientation_error_rad
                ),
                "rejection_target_index": swept.rejection_target_index,
                "rejection_substep": swept.rejection_substep,
                "reason": swept.reason,
                "collision_reasons": list(swept.collision_reasons),
            }
        inferred_contexts = [
            build_simulation_speed_context(
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
                # Future gripper targets express phase intent. Tracking error
                # is evaluated only when a sample becomes current at runtime.
                gripper_tracking_error_rad=0.0,
            )[0]
            for action in analysis
        ]
        phase_schedule = tuple(
            context.task_phase for context in inferred_contexts
        )
        scheduled_swept = swept_gate.check(
            source,
            EEFActionChunk(
                np.arange(len(analysis), dtype=np.float64) / POLICY_RATE_HZ,
                analysis,
            ),
            phase=phase_schedule,
        )
        record["inferred_phase_schedule"] = {
            "phases": list(phase_schedule),
            "counts": {
                phase: phase_schedule.count(phase)
                for phase in sorted(set(phase_schedule))
            },
        }
        record["swept_scheduled_view"] = {
            "accepted": scheduled_swept.accepted,
            "checked_targets": scheduled_swept.checked_targets,
            "checked_interpolated_configurations": (
                scheduled_swept.checked_interpolated_configurations
            ),
            "rejection_target_index": scheduled_swept.rejection_target_index,
            "rejection_substep": scheduled_swept.rejection_substep,
            "reason": scheduled_swept.reason,
            "collision_reasons": list(scheduled_swept.collision_reasons),
        }
        chunk_records.append(record)

    report = {
        "scope": "Diagnostic IK/collision preflight of quarantined real LGG100 chunks; no dynamics or hardware execution.",
        "source_chunks": str(args.chunks),
        "cube_translation_m": cube_translation.tolist(),
        "source_policy_state": source_state.tolist(),
        "observation_binding": observation_binding,
        "source_chunks_sha256": hashlib.sha256(args.chunks.read_bytes()).hexdigest(),
        "semantic_report_sha256": hashlib.sha256(
            args.semantic_report.read_bytes()
        ).hexdigest(),
        "semantic_identification_supported": True,
        "g1_contract_verified": False,
        "g1_sim_eligible": False,
        "g1_execution_enabled": False,
        "execution_performed": False,
        "chunks": chunk_records,
        "summary": {
            "chunks": len(chunk_records),
            "bounded_analysis_chunks": sum(
                record["bounded_analysis_available"] for record in chunk_records
            ),
            "all_targets_accepted_by_phase": {
                phase: sum(
                    bool(record["phase_views"].get(phase, {}).get("all_accepted"))
                    for record in chunk_records
                )
                for phase in ("free_space", "grasp", "place")
            },
            "swept_paths_accepted_by_phase": {
                phase: sum(
                    bool(record["swept_path_views"].get(phase, {}).get("accepted"))
                    for record in chunk_records
                )
                for phase in ("free_space", "grasp", "place")
            },
            "inferred_schedule_swept_paths_accepted": sum(
                bool((record.get("swept_scheduled_view") or {}).get("accepted"))
                for record in chunk_records
            ),
        },
        "verdict": "Preflight is diagnostic only. No result in this report grants simulation or hardware execution.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
