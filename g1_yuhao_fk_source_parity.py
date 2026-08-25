#!/usr/bin/env python3
"""Offline Yuhao Pinocchio vs project MuJoCo FK source-parity check.

The input is a previously captured subscriber-only G1 LowState sample. This
module has no robot network, publisher, controller, or command path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib.metadata import version as distribution_version
import json
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np


PINNED_YUHAO_COMMIT = "1422e8d6ef674aa047cfb2878bc7dae54b118fbe"
PINNED_EEF_KINEMATICS_SHA256 = (
    "1653c374535505738755872c64899b42ae99c576be3838487f8dc222137980e2"
)
PINNED_URDF_SHA256 = (
    "8bbf006633fc50b616f665c7a970780cc296577a0adfd7d28b049e751c238735"
)
POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_DEG = 3.0
SOURCE_PARITY_POSITION_TOLERANCE_M = 0.001
SOURCE_PARITY_ORIENTATION_TOLERANCE_DEG = 0.1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quaternion_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return float(np.rad2deg(2.0 * np.arccos(np.clip(abs(left @ right), 0.0, 1.0))))


def _pose_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    return {
        "left_position_error_m": float(np.linalg.norm(reference[:3] - candidate[:3])),
        "right_position_error_m": float(np.linalg.norm(reference[7:10] - candidate[7:10])),
        "left_orientation_error_deg": _quaternion_error_deg(reference[3:7], candidate[3:7]),
        "right_orientation_error_deg": _quaternion_error_deg(reference[10:14], candidate[10:14]),
    }


def _maximum_errors(comparison: dict[str, float]) -> tuple[float, float]:
    return (
        max(comparison["left_position_error_m"], comparison["right_position_error_m"]),
        max(
            comparison["left_orientation_error_deg"],
            comparison["right_orientation_error_deg"],
        ),
    )


def run_source_parity(
    project_root: Path, yuhao_root: Path, sample_path: Path
) -> dict[str, Any]:
    project_root = project_root.resolve()
    yuhao_root = yuhao_root.resolve()
    sample_path = sample_path.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from g1_dual_arm_ik import LEFT_JOINTS, RIGHT_JOINTS
    from g1_mujoco_bridge import policy_state_from_mujoco
    from stack_scene import build_model, reset_to_reference_pose

    module_path = yuhao_root / "openpi" / "eef_kinematics.py"
    urdf_path = yuhao_root / "assets" / "g1" / "g1_body29_hand14.urdf"
    eef_hash = _sha256(module_path)
    urdf_hash = _sha256(urdf_path)
    if eef_hash != PINNED_EEF_KINEMATICS_SHA256:
        raise RuntimeError(f"unexpected Yuhao eef_kinematics.py hash: {eef_hash}")
    if urdf_hash != PINNED_URDF_SHA256:
        raise RuntimeError(f"unexpected Yuhao URDF hash: {urdf_hash}")

    spec = importlib.util.spec_from_file_location(
        "pinned_yuhao_eef_kinematics", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned Yuhao eef_kinematics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sample = json.loads(sample_path.read_text())
    arm_names = LEFT_JOINTS + RIGHT_JOINTS
    q14 = np.asarray([sample["arm_q_rad"][name] for name in arm_names], dtype=np.float64)
    kinematics = module.G1DualArmKinematics()
    author_left, author_right = kinematics.fk(q14)
    author = np.concatenate([author_left, author_right]).astype(np.float64)

    def project_fk(use_measured_waist: bool) -> np.ndarray:
        model = build_model()
        data = mujoco.MjData(model)
        reset_to_reference_pose(model, data)
        for name in arm_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint_id]] = sample["arm_q_rad"][name]
        for name, value in sample["waist_q_rad"].items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint_id]] = value if use_measured_waist else 0.0
        mujoco.mj_forward(model, data)
        return policy_state_from_mujoco(model, data, np.zeros(2)).astype(np.float64)[:14]

    zero_waist = project_fk(False)
    measured_waist = project_fk(True)
    source_comparison = _pose_difference(author, zero_waist)
    waist_comparison = _pose_difference(zero_waist, measured_waist)
    source_position, source_orientation = _maximum_errors(source_comparison)
    waist_position, waist_orientation = _maximum_errors(waist_comparison)
    source_parity_passed = bool(
        source_position <= SOURCE_PARITY_POSITION_TOLERANCE_M
        and source_orientation <= SOURCE_PARITY_ORIENTATION_TOLERANCE_DEG
    )
    zero_waist_safe_approximation = bool(
        waist_position <= POSITION_TOLERANCE_M
        and waist_orientation <= ORIENTATION_TOLERANCE_DEG
    )

    return {
        "schema_version": "g1_yuhao_pinocchio_mujoco_fk_source_parity_v2",
        "scope": (
            "Offline source-parity and measured-waist impact analysis using one "
            "previously captured subscriber-only real G1 joint sample."
        ),
        "runtime": {
            "pinocchio_version": str(module.pin.__version__),
            "pin_distribution_version": distribution_version("pin"),
            "cmeel_urdfdom_version": distribution_version("cmeel-urdfdom"),
            "cmeel_tinyxml2_version": distribution_version("cmeel-tinyxml2"),
        },
        "source": {
            "repository": "https://github.com/leihao100/g1-client",
            "commit": PINNED_YUHAO_COMMIT,
            "eef_kinematics_sha256": eef_hash,
            "urdf_sha256": urdf_hash,
            "waist_assumption": "locked_at_zero",
            "eef_offset_local_x_m": float(module.EE_OFFSET),
        },
        "input": {
            "sample_sha256": _sha256(sample_path),
            "source_raw_capture_sha256": sample["source_raw_capture_sha256"],
            "tick": sample["tick"],
            "measured_waist_q_rad": sample["waist_q_rad"],
        },
        "thresholds": {
            "source_parity_position_m": SOURCE_PARITY_POSITION_TOLERANCE_M,
            "source_parity_orientation_deg": SOURCE_PARITY_ORIENTATION_TOLERANCE_DEG,
            "project_fk_position_m": POSITION_TOLERANCE_M,
            "project_fk_orientation_deg": ORIENTATION_TOLERANCE_DEG,
        },
        "poses_xyzw": {
            "yuhao_pinocchio_zero_waist": author.tolist(),
            "project_mujoco_zero_waist": zero_waist.tolist(),
            "project_mujoco_measured_waist": measured_waist.tolist(),
        },
        "comparison": {
            "yuhao_vs_project_zero_waist": source_comparison,
            "project_zero_vs_measured_waist": waist_comparison,
        },
        "decision": {
            "yuhao_project_zero_waist_source_parity_passed": source_parity_passed,
            "measured_waist_is_nonzero": bool(
                max(abs(value) for value in sample["waist_q_rad"].values()) > 1e-6
            ),
            "zero_waist_is_safe_approximation_at_current_pose": zero_waist_safe_approximation,
            "maximum_waist_induced_position_difference_m": waist_position,
            "maximum_waist_induced_orientation_difference_deg": waist_orientation,
            "shadow_policy_observation_fk": "yuhao_zero_waist_for_contract_parity",
            "shadow_safety_and_ik_fk": "measured_waist_for_physical_geometry",
            "dual_view_required": True,
            "waist_divergence_requires_hold": not zero_waist_safe_approximation,
            "offline_h2_core_completed": source_parity_passed,
            "physical_eef_parity_verified": False,
            "robot_motion_allowed": False,
            "reason": (
                "The two software models agree under the author's zero-waist "
                "assumption, but the measured waist changes EEF position beyond "
                "the registered 5 mm tolerance. Shadow must preserve zero-waist "
                "policy-contract state while independently using measured waist "
                "for safety/IK and holding on excessive divergence."
            ),
        },
        "safety": {
            "offline_only": True,
            "network_accessed": False,
            "publisher_created": False,
            "robot_command_sent": False,
            "mode_change_requested": False,
        },
        "g1_execution_enabled": False,
        "hardware_execution_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--yuhao-root", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_source_parity(args.project_root, args.yuhao_root, args.sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["decision"]["offline_h2_core_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
