#!/usr/bin/env python3
"""Apply the fail-closed context retimer to quarantined real LGG100 chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from action_schema import EEFActionChunk
from adaptive_speed_context import AdaptiveSafetyContext, ContextAwareRetimer
from g1_policy_contract import POLICY_RATE_HZ
from neural_action_audit import audit_neural_action_chunk

ROOT = Path(__file__).resolve().parent


def _load_chunks(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "actions" not in payload:
            raise ValueError("Expected actions in the quarantined NPZ")
        chunks = np.asarray(payload["actions"], dtype=np.float64)
    return chunks[None] if chunks.ndim == 2 else chunks


def _context(phase: str, distance: float, **changes) -> AdaptiveSafetyContext:
    values = {
        "task_phase": phase,
        "distance_to_goal_m": distance,
        "minimum_clearance_m": 0.20,
        "eef_tracking_error_m": 0.005,
        "observation_age_ms": 20.0,
        "policy_response_age_ms": 90.0,
        "ik_margin_rad": 0.20,
        "joint_limit_margin_rad": 0.20,
        "pelvis_stability": 1.0,
    }
    values.update(changes)
    return AdaptiveSafetyContext(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--semantic-report", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "lgg100_speed_context_diagnostic.json",
    )
    args = parser.parse_args()
    semantic = json.loads(args.semantic_report.read_text())
    if semantic.get("semantic_identification_supported") is not True:
        raise ValueError("Semantic identification must pass before this diagnostic")
    raw_chunks = _load_chunks(args.chunks)
    with np.load(args.observation, allow_pickle=False) as payload:
        state = np.asarray(payload["state"], dtype=np.float64)

    scenarios = {
        "safe_free_space": lambda distance: _context("free_space", distance),
        "precision_grasp": lambda distance: _context("grasp", distance),
        "precision_place": lambda distance: _context("place", distance),
        "low_clearance": lambda distance: _context(
            "free_space", distance, minimum_clearance_m=0.03
        ),
        "contact": lambda distance: _context(
            "free_space", distance, contact=True
        ),
        "stale": lambda distance: _context(
            "free_space", distance, observation_age_ms=101.0
        ),
        "unknown_phase": lambda distance: _context("unknown", distance),
    }
    records: list[dict] = []
    for index, raw in enumerate(raw_chunks):
        audit = audit_neural_action_chunk(raw)
        if audit.canonicalized_actions_for_analysis is None:
            records.append({
                "chunk": index,
                "available": False,
                "reasons": list(audit.reasons),
            })
            continue
        actions = audit.canonicalized_actions_for_analysis
        timestamps = np.arange(len(actions), dtype=np.float64) / POLICY_RATE_HZ
        chunk = EEFActionChunk(timestamps, actions)
        initial_jump = max(
            np.linalg.norm(actions[0, 0:3] - state[0:3]),
            np.linalg.norm(actions[0, 7:10] - state[7:10]),
        )
        distances = np.maximum(
            np.linalg.norm(actions[:, 0:3] - actions[-1, 0:3], axis=1),
            np.linalg.norm(actions[:, 7:10] - actions[-1, 7:10], axis=1),
        )
        scenario_results: dict[str, dict] = {}
        for name, context_factory in scenarios.items():
            contexts = [context_factory(float(distance)) for distance in distances]
            result = ContextAwareRetimer().plan(chunk, contexts)
            scenario_results[name] = {
                "accepted": result.accepted,
                "hold": result.hold,
                "reasons": list(result.reasons),
                "scale_range": (
                    [float(result.scale_profile.min()), float(result.scale_profile.max())]
                    if len(result.scale_profile) else None
                ),
                "duration_s": result.metrics.get("duration_s"),
                "metrics": result.metrics,
                "path_actions_byte_identical": result.path_actions_byte_identical,
            }
        records.append({
            "chunk": index,
            "available": True,
            "raw_sha256": audit.raw_sha256,
            "raw_max_quaternion_norm_error": audit.raw_max_quaternion_norm_error,
            "initial_target_jump_m": float(initial_jump),
            "maximum_distance_to_chunk_goal_m": float(np.max(distances)),
            "scenarios": scenario_results,
        })

    expected = all(
        record.get("available")
        and record["scenarios"]["safe_free_space"]["accepted"]
        and record["scenarios"]["precision_grasp"]["accepted"]
        and record["scenarios"]["precision_place"]["accepted"]
        and record["scenarios"]["low_clearance"]["accepted"]
        and record["scenarios"]["contact"]["accepted"]
        and record["scenarios"]["stale"]["hold"]
        and record["scenarios"]["unknown_phase"]["hold"]
        for record in records
    )
    report = {
        "scope": "Fail-closed speed-context diagnostic on quarantined real LGG100 chunks; no dynamics or hardware execution.",
        "source_sha256": hashlib.sha256(args.chunks.read_bytes()).hexdigest(),
        "semantic_identification_supported": True,
        "g1_contract_verified": False,
        "g1_sim_eligible": False,
        "g1_execution_enabled": False,
        "execution_performed": False,
        "chunks": records,
        "summary": {
            "chunks": len(records),
            "expected_context_behavior_passed": expected,
            "safe_free_space_accelerated_chunks": sum(
                record.get("scenarios", {}).get("safe_free_space", {}).get(
                    "scale_range", [0, 0]
                )[1] > 1.0
                for record in records
            ),
            "precision_chunks_capped_at_or_below_0_5": sum(
                max(
                    record.get("scenarios", {}).get(phase, {}).get(
                        "scale_range", [0, np.inf]
                    )[1]
                    for phase in ("precision_grasp", "precision_place")
                ) <= 0.5 + 1e-12
                for record in records
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(json.dumps(report["summary"], indent=2))
    if not expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
