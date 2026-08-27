#!/usr/bin/env python3
"""Paired offline benchmark of legacy and tolerance-aware G1 swept preflight.

Legacy here means this project's fixed-250-iteration MuJoCo implementation; it
is not Yuhao's Pinocchio implementation. Both branches use the same source,
action bytes, registered 5 mm / 3 degree acceptance Gate, collision classifier,
and interpolation resolution. The fast branch stops inside that Gate at
4 mm / 2.5 degrees.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import mujoco
import numpy as np

from g1_adaptive_phase_validation import FAR_LIFT_OFFSET_M
from g1_fast_sequential_ik import FastIKConfig
from g1_fast_swept_path_preflight import G1FastSweptPathPreflight
from g1_policy_contract import CONTRACT_ID, CONTRACT_SHA256, CONTRACT_VERSION, POLICY_RATE_HZ
from run_simulation import build_contract_fixture
from stack_scene import build_model, reset_to_reference_pose
from swept_path_preflight import G1SweptPathPreflight


PREFETCH_WINDOW_MS = 5.0 / POLICY_RATE_HZ * 1000.0
REQUIRED_SPEEDUP = 4.1


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def _bootstrap_median_ci(
    values: list[float], *, seed: int = 20260827, samples: int = 5000
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    medians = np.median(array[indices], axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def _run_timed(check: Callable[[], Any]) -> tuple[float, Any]:
    started = time.perf_counter_ns()
    result = check()
    return (time.perf_counter_ns() - started) / 1_000_000.0, result


def run_benchmark(*, pairs: int = 50, warmups: int = 5) -> dict[str, Any]:
    if not 5 <= pairs <= 500:
        raise ValueError("pairs must be in [5,500]")
    if not 1 <= warmups <= 20:
        raise ValueError("warmups must be in [1,20]")
    model = build_model()
    source = mujoco.MjData(model)
    reset_to_reference_pose(model, source)
    chunk, _, _ = build_contract_fixture(FAR_LIFT_OFFSET_M)
    fast_config = FastIKConfig()

    def legacy_check():
        return G1SweptPathPreflight(
            model,
            position_tolerance_m=0.005,
            orientation_tolerance_rad=np.deg2rad(3.0),
            ik_iterations=250,
            maximum_arm_interpolation_step_rad=0.01,
            maximum_finger_interpolation_step_m=0.001,
        ).check(source, chunk, phase="free_space")

    def fast_check():
        return G1FastSweptPathPreflight(
            model,
            ik_config=fast_config,
            maximum_arm_interpolation_step_rad=0.01,
            maximum_finger_interpolation_step_m=0.001,
        ).check(source, chunk, phase="free_space")

    for _ in range(warmups):
        legacy_check()
        fast_check()

    rng = np.random.default_rng(20260827)
    records = []
    previous_gc = gc.isenabled()
    gc.disable()
    try:
        for pair in range(pairs):
            order = ["legacy", "fast"]
            rng.shuffle(order)
            pair_record: dict[str, Any] = {"pair": pair, "order": order}
            for branch in order:
                elapsed_ms, result = _run_timed(
                    legacy_check if branch == "legacy" else fast_check
                )
                payload = {
                    "elapsed_ms": elapsed_ms,
                    "accepted": bool(result.accepted),
                    "checked_targets": int(result.checked_targets),
                    "checked_interpolated_configurations": int(
                        result.checked_interpolated_configurations
                    ),
                    "maximum_position_error_m": float(
                        result.maximum_position_error_m
                    ),
                    "maximum_orientation_error_rad": float(
                        result.maximum_orientation_error_rad
                    ),
                    "reason": result.reason,
                    "collision_reasons": list(result.collision_reasons),
                }
                if branch == "fast":
                    payload.update({
                        "total_ik_iterations": int(result.total_ik_iterations),
                        "ik_elapsed_ms": float(result.ik_elapsed_ms),
                        "collision_elapsed_ms": float(result.collision_elapsed_ms),
                        "internally_measured_total_ms": float(result.total_elapsed_ms),
                    })
                pair_record[branch] = payload
            pair_record["speedup_ratio"] = (
                pair_record["legacy"]["elapsed_ms"]
                / pair_record["fast"]["elapsed_ms"]
            )
            records.append(pair_record)
    finally:
        if previous_gc:
            gc.enable()

    legacy_ms = [record["legacy"]["elapsed_ms"] for record in records]
    fast_ms = [record["fast"]["elapsed_ms"] for record in records]
    speedups = [record["speedup_ratio"] for record in records]
    legacy_summary = _summary(legacy_ms)
    fast_summary = _summary(fast_ms)
    speedup_summary = _summary(speedups)
    speedup_ci = _bootstrap_median_ci(speedups)
    all_legacy_accepted = all(record["legacy"]["accepted"] for record in records)
    all_fast_accepted = all(record["fast"]["accepted"] for record in records)
    all_counts_equal = all(
        record["legacy"]["checked_targets"]
        == record["fast"]["checked_targets"] == len(chunk.actions)
        and record["legacy"]["checked_interpolated_configurations"]
        == record["fast"]["checked_interpolated_configurations"]
        for record in records
    )
    criteria = {
        "all_legacy_runs_accepted": all_legacy_accepted,
        "all_fast_runs_accepted": all_fast_accepted,
        "all_target_and_interpolation_counts_equal": all_counts_equal,
        "fast_position_residual_within_internal_4mm": all(
            record["fast"]["maximum_position_error_m"] <= 0.004
            for record in records
        ),
        "fast_orientation_residual_within_internal_2_5deg": all(
            record["fast"]["maximum_orientation_error_rad"] <= np.deg2rad(2.5)
            for record in records
        ),
        "fast_p95_within_333ms_prefetch_window": (
            fast_summary["p95"] <= PREFETCH_WINDOW_MS
        ),
        "paired_speedup_lower_95ci_at_least_4_1x": speedup_ci[0] >= REQUIRED_SPEEDUP,
    }
    return {
        "schema_version": "g1_fast_preflight_paired_benchmark_v1",
        "scope": (
            "Paired local MuJoCo microbenchmark on one deterministic 32-step "
            "fixture; not Yuhao runtime, neural task speed, or hardware evidence."
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.system(),
            "machine": platform.machine(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "garbage_collection_disabled_in_timed_region": True,
        },
        "contract": {
            "id": CONTRACT_ID,
            "version": CONTRACT_VERSION,
            "sha256": CONTRACT_SHA256,
            "policy_rate_hz": POLICY_RATE_HZ,
            "prefetch_lead_steps": 5,
            "prefetch_window_ms": PREFETCH_WINDOW_MS,
        },
        "input": {
            "source_qpos_sha256": _hash_array(source.qpos),
            "action_sha256": _hash_array(chunk.actions),
            "timestamp_sha256": _hash_array(chunk.timestamps),
            "action_shape": list(chunk.actions.shape),
            "phase": "free_space",
            "warmups_per_branch": warmups,
            "formal_pairs": pairs,
            "order_seed": 20260827,
        },
        "comparison_contract": {
            "legacy_definition": "project MuJoCo fixed 250 iterations per target",
            "legacy_is_yuhao_pinocchio": False,
            "registered_position_tolerance_m": 0.005,
            "registered_orientation_tolerance_deg": 3.0,
            "fast_internal_position_stop_m": fast_config.position_tolerance_m,
            "fast_internal_orientation_stop_deg": float(
                np.rad2deg(fast_config.orientation_tolerance_rad)
            ),
            "maximum_arm_interpolation_step_rad": 0.01,
            "maximum_finger_interpolation_step_m": 0.001,
            "collision_classifier_identical": True,
            "threshold_relaxed": False,
        },
        "summary": {
            "legacy_total_ms": legacy_summary,
            "fast_total_ms": fast_summary,
            "paired_speedup_ratio": speedup_summary,
            "paired_median_speedup_95ci": speedup_ci,
            "median_of_paired_ratios": speedup_summary["p50"],
            "ratio_of_median_latencies": legacy_summary["p50"] / fast_summary["p50"],
            "criteria": criteria,
            "benchmark_passed": all(criteria.values()),
        },
        "runs": records,
        "decision": {
            "offline_single_fixture_preflight_speed_gate_passed": all(criteria.values()),
            "multi_scenario_correctness_corpus_passed": False,
            "yuhao_pinocchio_comparison_completed": False,
            "real_robot_compute_benchmark_completed": False,
            "task_level_speedup_passed": False,
            "robot_motion_allowed": False,
        },
        "safety": {
            "offline_only": True,
            "publisher_created": False,
            "robot_command_sent": False,
        },
        "g1_execution_enabled": False,
        "hardware_execution_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(pairs=args.pairs, warmups=args.warmups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["summary"]["benchmark_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
