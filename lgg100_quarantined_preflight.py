#!/usr/bin/env python3
"""Diagnostic-only preflight for quarantined real LGG100 chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from action_schema import pelvis_vla_action_to_world_mujoco
from g1_policy_contract import POLICY_RATE_HZ
from neural_action_audit import audit_neural_action_chunk
from safety_governor import G1TargetPreflight
from stack_scene import build_model, reset_to_reference_pose

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
    if chunks.ndim != 3 or chunks.shape[1:] != (50, 16):
        raise ValueError(f"Expected [N,50,16], got {chunks.shape}")
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
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    gate = G1TargetPreflight(model)
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
        chunk_records.append(record)

    report = {
        "scope": "Diagnostic IK/collision preflight of quarantined real LGG100 chunks; no dynamics or hardware execution.",
        "source_chunks": str(args.chunks),
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
        },
        "verdict": "Preflight is diagnostic only. No result in this report grants simulation or hardware execution.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
