#!/usr/bin/env python3
"""Build an explicitly experimental camera-calibrated G1 observation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from camera_calibration_search import _look_at_quaternion
from camera_calibration_search_v2 import _lighten_skybox, _wrist_quaternion
from dex1_gripper import Dex1Controller
from g1_mujoco_bridge import policy_state_from_mujoco
from g1_policy_contract import IMAGE_KEYS, preprocess_rgb_image, validate_observation
from stack_scene import CAMERA_NAMES, TASK_PROMPT, build_model, reset_to_reference_pose

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration_report.read_text())
    best = {
        camera: calibration["search"][camera]["best"]
        for camera in CAMERA_NAMES
    }
    model = build_model()
    _lighten_skybox(model)
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    hold = data.ctrl.copy()
    for _ in range(250):
        data.ctrl[:] = hold
        mujoco.mj_step(model, data)

    high = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_left_high"
    )
    high_position = np.asarray(best["cam_left_high"]["position"])
    high_target = np.asarray(best["cam_left_high"]["target"])
    model.cam_pos[high] = high_position
    model.cam_quat[high] = _look_at_quaternion(high_position, high_target)
    model.cam_fovy[high] = best["cam_left_high"]["fovy"]
    for camera in ("cam_left_wrist", "cam_right_wrist"):
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera
        )
        entry = best[camera]
        model.cam_pos[camera_id] = np.asarray(entry["position"])
        model.cam_quat[camera_id] = _wrist_quaternion(
            entry["pitch_deg"], entry["yaw_deg"]
        )
        model.cam_fovy[camera_id] = entry["fovy"]

    renderer = mujoco.Renderer(model, height=480, width=640)
    source_images = {}
    try:
        for camera in CAMERA_NAMES:
            renderer.update_scene(data, camera=camera)
            image = renderer.render().copy()
            if best[camera].get("vertical_flip"):
                image = image[::-1].copy()
            source_images[camera] = image
    finally:
        renderer.close()
    images = {
        camera: preprocess_rgb_image(image)
        for camera, image in source_images.items()
    }
    state = policy_state_from_mujoco(
        model, data, Dex1Controller(model).motor_states(data)
    )
    observation = {
        IMAGE_KEYS[0]: images["cam_left_high"],
        IMAGE_KEYS[1]: images["cam_left_wrist"],
        IMAGE_KEYS[2]: images["cam_right_wrist"],
        "observation/state": state,
        "prompt": TASK_PROMPT,
    }
    validate_observation(observation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        cam_left_high=observation[IMAGE_KEYS[0]],
        cam_left_wrist=observation[IMAGE_KEYS[1]],
        cam_right_wrist=observation[IMAGE_KEYS[2]],
        state=state,
        prompt=np.asarray(TASK_PROMPT),
        calibration_report_sha256=np.asarray(
            hashlib.sha256(args.calibration_report.read_bytes()).hexdigest()
        ),
        experimental=np.asarray(True),
        contract_eligible=np.asarray(False),
    )
    evidence = {
        "scope": "Experimental visual calibration candidate; not the frozen production contract.",
        "calibration_report": str(args.calibration_report),
        "calibration_report_sha256": hashlib.sha256(
            args.calibration_report.read_bytes()
        ).hexdigest(),
        "observation_npz": str(args.output),
        "observation_npz_sha256": hashlib.sha256(
            args.output.read_bytes()
        ).hexdigest(),
        "camera_parameters": best,
        "experimental": True,
        "g1_contract_verified": False,
        "g1_sim_eligible": False,
        "g1_execution_enabled": False,
        "image_sha256": {
            camera: hashlib.sha256(images[camera].tobytes()).hexdigest()
            for camera in CAMERA_NAMES
        },
        "state": state.tolist(),
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
    print(args.output)
    print(args.evidence)
    print(json.dumps({
        "observation_npz_sha256": evidence["observation_npz_sha256"],
        "image_sha256": evidence["image_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
