#!/usr/bin/env python3
"""Layered offline benchmark of pinned Yuhao and project IK/preflight paths."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import mujoco
import numpy as np

from g1_adaptive_phase_validation import FAR_LIFT_OFFSET_M
from g1_dual_arm_ik import LEFT_JOINTS, RIGHT_JOINTS
from g1_fast_sequential_ik import solve_sequential_ik
from g1_yuhao_fk_source_parity import (
    PINNED_EEF_KINEMATICS_SHA256,
    PINNED_URDF_SHA256,
    PINNED_YUHAO_COMMIT,
)
from run_simulation import build_contract_fixture
from stack_scene import build_model, reset_to_reference_pose


POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_RAD = np.deg2rad(3.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def _orientation_error_rad(left: np.ndarray, right: np.ndarray) -> float:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return float(2.0 * np.arccos(np.clip(abs(left @ right), 0.0, 1.0)))


def _load_yuhao(yuhao_root: Path):
    module_path = yuhao_root / "openpi" / "eef_kinematics.py"
    urdf_path = yuhao_root / "assets" / "g1" / "g1_body29_hand14.urdf"
    if _sha256(module_path) != PINNED_EEF_KINEMATICS_SHA256:
        raise RuntimeError("pinned Yuhao eef_kinematics.py hash mismatch")
    if _sha256(urdf_path) != PINNED_URDF_SHA256:
        raise RuntimeError("pinned Yuhao URDF hash mismatch")
    spec = importlib.util.spec_from_file_location("pinned_yuhao_layered_benchmark", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned Yuhao kinematics")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module_path, urdf_path


def _author_run(kinematics, actions: np.ndarray, initial_q14: np.ndarray) -> dict[str, Any]:
    q = initial_q14.copy()
    maximum_position_error = 0.0
    maximum_orientation_error = 0.0
    started = time.perf_counter_ns()
    for action in actions:
        q, position_error = kinematics.solve_ik(
            action[:7], action[7:14], q,
            max_iters=20, tol=1e-5, damping=1e-8,
        )
        left, right = kinematics.fk(q)
        orientation_error = max(
            _orientation_error_rad(left[3:7], action[3:7]),
            _orientation_error_rad(right[3:7], action[10:14]),
        )
        maximum_position_error = max(maximum_position_error, float(position_error))
        maximum_orientation_error = max(maximum_orientation_error, orientation_error)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return {
        "elapsed_ms": elapsed_ms,
        "accepted_by_project_outer_gate": bool(
            maximum_position_error <= POSITION_TOLERANCE_M
            and maximum_orientation_error <= ORIENTATION_TOLERANCE_RAD
        ),
        "maximum_position_error_m": maximum_position_error,
        "maximum_orientation_error_rad": maximum_orientation_error,
        "checked_targets": len(actions),
    }


def _fast_run(model, source, actions: np.ndarray) -> dict[str, Any]:
    started = time.perf_counter_ns()
    result = solve_sequential_ik(model, source, actions)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return {
        "elapsed_ms": elapsed_ms,
        "accepted_by_project_outer_gate": result.accepted,
        "maximum_position_error_m": max(
            target.maximum_position_error_m for target in result.targets
        ),
        "maximum_orientation_error_rad": max(
            target.maximum_orientation_error_rad for target in result.targets
        ),
        "checked_targets": len(result.targets),
        "total_iterations": result.total_iterations,
    }


def run_benchmark(
    *, yuhao_root: Path, preflight_report_path: Path, pairs: int = 50, warmups: int = 5
) -> dict[str, Any]:
    if not 5 <= pairs <= 500:
        raise ValueError("pairs must be in [5,500]")
    if not 1 <= warmups <= 20:
        raise ValueError("warmups must be in [1,20]")
    module, module_path, urdf_path = _load_yuhao(yuhao_root.resolve())
    model = build_model()
    source = mujoco.MjData(model)
    reset_to_reference_pose(model, source)
    chunk, _, _ = build_contract_fixture(FAR_LIFT_OFFSET_M)
    arm_names = LEFT_JOINTS + RIGHT_JOINTS
    initial_q14 = np.asarray([
        source.qpos[model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]]
        for name in arm_names
    ], dtype=np.float64)
    kinematics = module.G1DualArmKinematics()

    for _ in range(warmups):
        _author_run(kinematics, chunk.actions, initial_q14)
        _fast_run(model, source, chunk.actions)

    records = []
    rng = np.random.default_rng(20260827)
    previous_gc = gc.isenabled()
    gc.disable()
    try:
        for pair in range(pairs):
            order = ["yuhao_pinocchio_ik_only", "project_fast_mujoco_ik_only"]
            rng.shuffle(order)
            record: dict[str, Any] = {"pair": pair, "order": order}
            for branch in order:
                if branch == "yuhao_pinocchio_ik_only":
                    record[branch] = _author_run(
                        kinematics, chunk.actions, initial_q14
                    )
                else:
                    record[branch] = _fast_run(model, source, chunk.actions)
            records.append(record)
    finally:
        if previous_gc:
            gc.enable()

    preflight_bytes = preflight_report_path.read_bytes()
    preflight = json.loads(preflight_bytes)
    if preflight["input"]["action_sha256"] != _hash_array(chunk.actions):
        raise ValueError("preflight benchmark action hash does not match layered input")
    author_times = [r["yuhao_pinocchio_ik_only"]["elapsed_ms"] for r in records]
    fast_times = [r["project_fast_mujoco_ik_only"]["elapsed_ms"] for r in records]
    all_author_pass = all(
        r["yuhao_pinocchio_ik_only"]["accepted_by_project_outer_gate"]
        for r in records
    )
    all_fast_pass = all(
        r["project_fast_mujoco_ik_only"]["accepted_by_project_outer_gate"]
        for r in records
    )
    return {
        "schema_version": "g1_yuhao_fast_layered_benchmark_v1",
        "scope": (
            "Same-input 32-target offline solver throughput plus linked complete "
            "preflight evidence. Yuhao production solves selected ticks and has no "
            "project swept collision Gate; no cross-layer total-speed ratio is claimed."
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.system(),
            "machine": platform.machine(),
            "pinocchio": module.pin.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
        "source": {
            "yuhao_repository": "https://github.com/leihao100/g1-client",
            "yuhao_commit": PINNED_YUHAO_COMMIT,
            "eef_kinematics_sha256": _sha256(module_path),
            "urdf_sha256": _sha256(urdf_path),
            "project_preflight_report_sha256": hashlib.sha256(preflight_bytes).hexdigest(),
        },
        "input": {
            "action_sha256": _hash_array(chunk.actions),
            "action_shape": list(chunk.actions.shape),
            "source_q14_sha256": _hash_array(initial_q14),
            "pairs": pairs,
            "warmups_per_branch": warmups,
            "order_seed": 20260827,
            "waist_assumption": "zero_for_same_input_software_comparison",
        },
        "layers": {
            "yuhao_pinocchio_ik_only": {
                "official_max_iterations_per_target": 20,
                "swept_collision_included": False,
                "latency_ms": _summary(author_times),
                "all_runs_pass_project_5mm_3deg_outer_gate": all_author_pass,
                "maximum_position_error_m": max(
                    r["yuhao_pinocchio_ik_only"]["maximum_position_error_m"]
                    for r in records
                ),
                "maximum_orientation_error_rad": max(
                    r["yuhao_pinocchio_ik_only"]["maximum_orientation_error_rad"]
                    for r in records
                ),
            },
            "project_fast_mujoco_ik_only": {
                "maximum_iterations_per_target": 60,
                "internal_stop": "4 mm / 2.5 deg",
                "swept_collision_included": False,
                "latency_ms": _summary(fast_times),
                "all_runs_pass_project_5mm_3deg_outer_gate": all_fast_pass,
                "maximum_position_error_m": max(
                    r["project_fast_mujoco_ik_only"]["maximum_position_error_m"]
                    for r in records
                ),
                "maximum_orientation_error_rad": max(
                    r["project_fast_mujoco_ik_only"]["maximum_orientation_error_rad"]
                    for r in records
                ),
                "total_iterations_per_run": sorted({
                    r["project_fast_mujoco_ik_only"]["total_iterations"]
                    for r in records
                }),
            },
            "project_legacy_mujoco_ik_plus_swept": {
                "linked_not_rerun": True,
                "fixed_iterations_per_target": 250,
                "swept_collision_included": True,
                "latency_ms": preflight["summary"]["legacy_total_ms"],
            },
            "project_fast_mujoco_ik_plus_swept": {
                "linked_not_rerun": True,
                "swept_collision_included": True,
                "latency_ms": preflight["summary"]["fast_total_ms"],
                "prefetch_gate_passed": preflight["summary"]["criteria"][
                    "fast_p95_within_333ms_prefetch_window"
                ],
            },
        },
        "runs": records,
        "decision": {
            "same_input_solver_layer_comparison_completed": all_author_pass and all_fast_pass,
            "complete_preflight_layer_linked": True,
            "yuhao_ik_only_to_project_complete_preflight_ratio_claimed": False,
            "yuhao_production_loop_benchmarked": False,
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
    parser.add_argument("--yuhao-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        yuhao_root=args.yuhao_root,
        preflight_report_path=args.preflight_report,
        pairs=args.pairs,
        warmups=args.warmups,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["decision"]["same_input_solver_layer_comparison_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
