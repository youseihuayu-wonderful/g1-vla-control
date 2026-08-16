#!/usr/bin/env python3
"""Validate controlled near/mixed/far phase-aware speed behavior.

This gate is intentionally narrower than task success. It consumes hash-bound,
quarantined ensemble/preflight/dynamics artifacts and cannot authorize G1
simulation eligibility or hardware execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from g1_policy_contract import (
    ACTION_HORIZON,
    CONTRACT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    POLICY_RATE_HZ,
)

ROLES = ("near", "mixed", "far")
NOMINAL_CHUNK_DURATION_S = (ACTION_HORIZON - 1) / POLICY_RATE_HZ


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _false_safety_flags(payload: dict) -> bool:
    return all(
        payload.get(key) is False
        for key in (
            "g1_contract_verified", "g1_sim_eligible",
            "g1_execution_enabled",
        )
    )


def validate_scenario(
    role: str,
    expected_x_offset_m: float,
    ensemble_report_path: Path,
    preflight_path: Path,
    dynamics_path: Path,
) -> dict:
    if role not in ROLES:
        raise ValueError(f"unknown scenario role: {role}")
    ensemble = _load(ensemble_report_path)
    preflight = _load(preflight_path)
    dynamics = _load(dynamics_path)
    reasons: list[str] = []
    expected_binding = {
        "g1_policy_contract_id": CONTRACT_ID,
        "g1_policy_contract_version": CONTRACT_VERSION,
        "g1_policy_contract_sha256": CONTRACT_SHA256,
        "action_horizon": ACTION_HORIZON,
    }
    for label, payload in (
        ("ensemble", ensemble),
        ("preflight", preflight),
        ("dynamics", dynamics),
    ):
        if any(payload.get(key) != value for key, value in expected_binding.items()):
            reasons.append(f"{label}_contract_binding_mismatch")

    output_path = Path(str(ensemble.get("output", "")))
    if not output_path.is_file():
        reasons.append("ensemble_output_missing")
        output_hash = None
    else:
        output_hash = _sha256(output_path)
        if output_hash != ensemble.get("output_sha256"):
            reasons.append("ensemble_output_hash_mismatch")
    if ensemble.get("quarantined") is not True:
        reasons.append("ensemble_not_quarantined")
    if not _false_safety_flags(ensemble):
        reasons.append("ensemble_safety_flags_not_false")

    translation = np.asarray(
        preflight.get("cube_translation_m", []), dtype=np.float64
    )
    expected_translation = np.array([expected_x_offset_m, 0.0, 0.0])
    if translation.shape != (3,) or not np.allclose(
        translation, expected_translation, rtol=0.0, atol=1e-12
    ):
        reasons.append("preflight_scene_translation_mismatch")
    if preflight.get("source_chunks_sha256") != output_hash:
        reasons.append("preflight_source_hash_mismatch")
    observation_binding = preflight.get("observation_binding") or {}
    if observation_binding.get("accepted") is not True:
        reasons.append("preflight_observation_binding_missing_or_failed")
    if observation_binding.get("maximum_state_error", np.inf) > 1e-6:
        reasons.append("preflight_observation_state_mismatch")
    if observation_binding.get("cube_translation_matches") is not True:
        reasons.append("preflight_observation_scene_mismatch")
    if observation_binding.get("contract_matches") is not True:
        reasons.append("preflight_observation_contract_mismatch")
    if not _false_safety_flags(preflight):
        reasons.append("preflight_safety_flags_not_false")
    preflight_summary = preflight.get("summary", {})
    if preflight_summary.get("bounded_analysis_chunks") != 1:
        reasons.append("bounded_ensemble_preflight_missing")
    if preflight_summary.get("inferred_schedule_swept_paths_accepted") != 1:
        reasons.append("scheduled_swept_path_not_accepted")

    dynamics_translation = np.asarray(
        dynamics.get("cube_translation_m", []), dtype=np.float64
    )
    if dynamics_translation.shape != (3,) or not np.allclose(
        dynamics_translation, expected_translation, rtol=0.0, atol=1e-12
    ):
        reasons.append("dynamics_scene_translation_mismatch")
    if dynamics.get("source_chunks_sha256") != output_hash:
        reasons.append("dynamics_source_hash_mismatch")
    if dynamics.get("preflight_report_sha256") != _sha256(preflight_path):
        reasons.append("dynamics_preflight_hash_mismatch")
    if not _false_safety_flags(dynamics):
        reasons.append("dynamics_safety_flags_not_false")
    if dynamics.get("task_success_claimed") is not False:
        reasons.append("dynamics_claims_task_success")
    summary = dynamics.get("summary", {})
    if (
        summary.get("executed_diagnostic_chunks") != 1
        or summary.get("candidate_passes") != 1
        or summary.get("all_executed_candidates_passed") is not True
    ):
        reasons.append("dynamics_candidate_not_passing")

    records = dynamics.get("records", [])
    if len(records) != 1:
        reasons.append("expected_exactly_one_ensemble_record")
        record = {}
    else:
        record = records[0]
    context = record.get("context", {})
    phase_counts = context.get("phase_counts", {})
    retiming = record.get("retiming", {})
    scale_range = retiming.get("scale_range", [np.nan, np.nan])
    metrics = retiming.get("metrics", {})
    baseline = record.get("baseline", {})
    guarded = record.get("guarded", {})
    clearance = float(context.get("minimum_clearance_m", np.nan))
    minimum_scale = float(scale_range[0]) if len(scale_range) == 2 else np.nan
    maximum_scale = float(scale_range[1]) if len(scale_range) == 2 else np.nan
    duration = float(metrics.get("duration_s", np.nan))
    if not record.get("execution_performed") or not record.get("candidate_passed"):
        reasons.append("ensemble_dynamics_record_not_passing")
    if retiming.get("path_actions_byte_identical") is not True:
        reasons.append("retimer_changed_action_samples")
    for branch_name, branch in (("baseline", baseline), ("guarded", guarded)):
        if branch.get("hard_command_limits_pass") is not True:
            reasons.append(f"{branch_name}_command_limits_failed")
        if branch.get("finite") is not True:
            reasons.append(f"{branch_name}_non_finite")
        if branch.get("aborted_on_phase_aware_contact") is not False:
            reasons.append(f"{branch_name}_contact_abort")
    if guarded.get("phase_aware_contact_step_rate") != 0.0:
        reasons.append("guarded_phase_invalid_contact")
    if sum(int(value) for value in phase_counts.values()) != ACTION_HORIZON:
        reasons.append("phase_schedule_length_mismatch")
    if not np.all(np.isfinite((clearance, minimum_scale, maximum_scale, duration))):
        reasons.append("non_finite_speed_evidence")

    tolerance = 1e-9
    if role == "near":
        if not clearance < 0.060:
            reasons.append("near_clearance_not_near")
        if phase_counts.get("free_space", 0) != 0:
            reasons.append("near_contains_free_space_phase")
        if maximum_scale > 0.5 + tolerance:
            reasons.append("near_precision_cap_exceeded")
        if duration <= NOMINAL_CHUNK_DURATION_S:
            reasons.append("near_chunk_not_slowed")
    elif role == "mixed":
        if not 0.10 < clearance < 0.16:
            reasons.append("mixed_clearance_outside_band")
        if phase_counts.get("free_space", 0) <= 0:
            reasons.append("mixed_missing_free_space_phase")
        if phase_counts.get("grasp", 0) <= 0:
            reasons.append("mixed_missing_grasp_phase")
        if not (minimum_scale < 1.0 < maximum_scale):
            reasons.append("mixed_missing_accelerate_then_slow_behavior")
    elif role == "far":
        if clearance < 0.16:
            reasons.append("far_clearance_too_low")
        if phase_counts != {"free_space": ACTION_HORIZON}:
            reasons.append("far_not_all_free_space")
        if maximum_scale <= 1.0:
            reasons.append("far_acceleration_not_exercised")
        if duration >= NOMINAL_CHUNK_DURATION_S:
            reasons.append("far_chunk_not_faster_than_nominal")

    return {
        "role": role,
        "expected_cube_translation_m": expected_translation.tolist(),
        "ensemble_report": str(ensemble_report_path),
        "ensemble_report_sha256": _sha256(ensemble_report_path),
        "preflight_report": str(preflight_path),
        "preflight_report_sha256": _sha256(preflight_path),
        "dynamics_report": str(dynamics_path),
        "dynamics_report_sha256": _sha256(dynamics_path),
        "clearance_m": clearance,
        "phase_counts": phase_counts,
        "scale_range": [minimum_scale, maximum_scale],
        "retimed_chunk_duration_s": duration,
        "nominal_chunk_duration_s": NOMINAL_CHUNK_DURATION_S,
        "duration_change_percent": (
            100.0 * (duration - NOMINAL_CHUNK_DURATION_S)
            / NOMINAL_CHUNK_DURATION_S
        ),
        "accepted": not reasons,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", action="append", nargs=5, required=True,
        metavar=("ROLE", "X_OFFSET_M", "ENSEMBLE", "PREFLIGHT", "DYNAMICS"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.scenario) != len(ROLES):
        raise SystemExit("exactly one near, mixed, and far scenario is required")
    records = [
        validate_scenario(
            values[0], float(values[1]), Path(values[2]),
            Path(values[3]), Path(values[4]),
        )
        for values in args.scenario
    ]
    if sorted(record["role"] for record in records) != sorted(ROLES):
        raise SystemExit("scenario roles must be exactly near, mixed, and far")
    ordered = sorted(records, key=lambda record: ROLES.index(record["role"]))
    clearances = [record["clearance_m"] for record in ordered]
    monotonic_clearance = bool(
        np.all(np.isfinite(clearances)) and np.all(np.diff(clearances) > 0.0)
    )
    passed = bool(monotonic_clearance and all(record["accepted"] for record in ordered))
    report = {
        "scope": "Controlled hash-bound phase-aware speed sweep over quarantined single-chunk LGG100 ensembles.",
        "scenarios": ordered,
        "monotonic_clearance": monotonic_clearance,
        "controlled_phase_speed_behavior_passed": passed,
        "semantic_interpretation_supported": True,
        "observation_execution_initial_state_bound": passed,
        "physical_scene_calibration_verified": False,
        "policy_task_quality_passed": False,
        "task_success_claimed": False,
        "g1_contract_verified": False,
        "g1_sim_eligible": False,
        "g1_execution_enabled": False,
        "hardware_execution_performed": False,
        "verdict": (
            "Controlled phase-aware speed behavior passed. This does not validate scene calibration, closed-loop stacking, or hardware execution."
            if passed else
            "Controlled phase-aware speed behavior failed; keep all neural actions quarantined."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(json.dumps({
        "controlled_phase_speed_behavior_passed": passed,
        "monotonic_clearance": monotonic_clearance,
        "scenarios": [
            {
                "role": record["role"],
                "accepted": record["accepted"],
                "clearance_m": record["clearance_m"],
                "scale_range": record["scale_range"],
                "duration_change_percent": record["duration_change_percent"],
                "reasons": record["reasons"],
            }
            for record in ordered
        ],
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
