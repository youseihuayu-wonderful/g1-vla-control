#!/usr/bin/env python3
"""Trace quarantined LGG100 target convergence in MuJoCo without execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from action_schema import pelvis_vla_action_to_world_mujoco
from g1_dual_arm_ik import G1DualArmIK, orientation_error
from g1_policy_contract import ACTION_HORIZON
from stack_scene import build_model, reset_to_reference_pose, translate_cubes

POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_RAD = np.deg2rad(3.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_checkpoints(value: str) -> tuple[int, ...]:
    try:
        checkpoints = tuple(sorted({int(item) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoints must be comma-separated integers") from exc
    if not checkpoints or checkpoints[0] != 0 or checkpoints[-1] > 1000:
        raise argparse.ArgumentTypeError("checkpoints must include 0 and be at most 1000")
    return checkpoints


def _errors(solver: G1DualArmIK, target: np.ndarray, iteration: int) -> dict:
    left_position, left_quaternion = solver.pose("left")
    right_position, right_quaternion = solver.pose("right")
    left_position_error = float(np.linalg.norm(left_position - target[:3]))
    right_position_error = float(np.linalg.norm(right_position - target[7:10]))
    left_orientation_error = float(np.linalg.norm(
        orientation_error(target[3:7], left_quaternion)
    ))
    right_orientation_error = float(np.linalg.norm(
        orientation_error(target[10:14], right_quaternion)
    ))
    maximum_position_error = max(left_position_error, right_position_error)
    maximum_orientation_error = max(
        left_orientation_error, right_orientation_error
    )
    return {
        "iteration": iteration,
        "left_position_error_m": left_position_error,
        "right_position_error_m": right_position_error,
        "maximum_position_error_m": maximum_position_error,
        "left_orientation_error_rad": left_orientation_error,
        "right_orientation_error_rad": right_orientation_error,
        "maximum_orientation_error_rad": maximum_orientation_error,
        "within_registered_tolerance": bool(
            maximum_position_error <= POSITION_TOLERANCE_M
            and maximum_orientation_error <= ORIENTATION_TOLERANCE_RAD
        ),
    }


def _joints_at_limits(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    solver: G1DualArmIK,
) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for side, info in (("left", solver.left), ("right", solver.right)):
        values = []
        for joint_id, qpos_id in zip(
            info["joint"], info["qpos"], strict=True
        ):
            low, high = model.jnt_range[joint_id]
            qpos = float(data.qpos[qpos_id])
            if model.jnt_limited[joint_id] and (
                abs(qpos - low) < 1e-6 or abs(qpos - high) < 1e-6
            ):
                values.append({
                    "joint": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)
                    ),
                    "qpos": qpos,
                    "range": [float(low), float(high)],
                })
        result[side] = values
    return result


def _load_action(
    path: Path,
    *,
    cycle: int,
    target_index: int,
    expected_cube_translation: np.ndarray,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if bool(archive["executable"]) or not bool(archive["quarantined"]):
            raise ValueError(f"artifact is not quarantined and non-executable: {path}")
        actions = np.asarray(
            archive["canonicalized_actions_for_analysis"], dtype=np.float64
        )
        translation = np.asarray(archive["cube_translation_m"], dtype=np.float64)
    if actions.ndim != 3 or actions.shape[1:] != (ACTION_HORIZON, 16):
        raise ValueError(f"unexpected action shape {actions.shape}: {path}")
    if not 0 <= cycle < len(actions) or not 0 <= target_index < ACTION_HORIZON:
        raise ValueError(f"cycle or target index is outside artifact bounds: {path}")
    if translation.shape != (3,) or not np.array_equal(
        translation, expected_cube_translation
    ):
        raise ValueError(f"cube translation does not match diagnostic scene: {path}")
    return actions[cycle, target_index].copy()


def _trace_trial(
    path: Path,
    *,
    cycle: int,
    target_index: int,
    cube_translation: np.ndarray,
    checkpoints: tuple[int, ...],
) -> dict:
    action = _load_action(
        path,
        cycle=cycle,
        target_index=target_index,
        expected_cube_translation=cube_translation,
    )
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    translate_cubes(model, data, cube_translation)
    pelvis = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )
    target = pelvis_vla_action_to_world_mujoco(
        action, data.xpos[pelvis], data.xquat[pelvis]
    )
    solver = G1DualArmIK(
        model, data, damping=0.06, max_joint_speed=2.0
    )
    solver.reset()
    arm_qpos = np.concatenate((solver.left["qpos"], solver.right["qpos"]))
    checkpoint_set = set(checkpoints)
    trace = []
    start = time.monotonic()
    for iteration in range(checkpoints[-1] + 1):
        if iteration in checkpoint_set:
            trace.append(_errors(solver, target, iteration))
        if iteration == checkpoints[-1]:
            break
        solver.step(target[:7], target[7:14], 0.02)
        data.qpos[arm_qpos] = solver.q_target[arm_qpos]
        data.qvel[:] = 0.0
        mujoco.mj_fwdPosition(model, data)
    passing = [
        entry["iteration"] for entry in trace
        if entry["within_registered_tolerance"]
    ]
    return {
        "artifact": str(path),
        "artifact_sha256": _sha256(path),
        "cycle": cycle,
        "target_index": target_index,
        "target_world": {
            "left_position_m": target[:3].tolist(),
            "right_position_m": target[7:10].tolist(),
        },
        "trace": trace,
        "first_passing_checkpoint": passing[0] if passing else None,
        "joints_at_limits_after_final_checkpoint": _joints_at_limits(
            model, data, solver
        ),
        "elapsed_ms": (time.monotonic() - start) * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trial", action="append", nargs=2, metavar=("NAME", "NPZ"),
        required=True,
    )
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--cube-x-offset-m", type=float, default=0.08)
    parser.add_argument(
        "--checkpoints", type=_parse_checkpoints,
        default=_parse_checkpoints("0,1,5,10,20,30,40,60,100,150,250"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cube_translation = np.array([args.cube_x_offset_m, 0.0, 0.0])
    trials = {
        name: _trace_trial(
            Path(path),
            cycle=args.cycle,
            target_index=args.target_index,
            cube_translation=cube_translation,
            checkpoints=args.checkpoints,
        )
        for name, path in args.trial
    }
    report = {
        "scope": (
            "Read-only convergence trace for quarantined LGG100 targets; "
            "MuJoCo only, never hardware."
        ),
        "position_tolerance_m": POSITION_TOLERANCE_M,
        "orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
        "cube_translation_m": cube_translation.tolist(),
        "checkpoints": list(args.checkpoints),
        "trials": trials,
        "g1_execution_enabled": False,
        "hardware_execution_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(json.dumps({
        name: {
            "first_passing_checkpoint": trial["first_passing_checkpoint"],
            "final_errors": trial["trace"][-1],
            "joints_at_limits": trial[
                "joints_at_limits_after_final_checkpoint"
            ],
            "elapsed_ms": trial["elapsed_ms"],
        }
        for name, trial in trials.items()
    }, indent=2))


if __name__ == "__main__":
    main()
