"""Experimental calibrated MuJoCo observation helpers.

These camera candidates are useful only for quarantined diagnostics. They are
not a calibrated physical sensor contract and cannot enable simulation or G1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from camera_calibration_search import _look_at_quaternion
from camera_calibration_search_v2 import _lighten_skybox, _wrist_quaternion
from dex1_gripper import Dex1Controller
from g1_mujoco_bridge import policy_state_from_mujoco
from g1_policy_contract import IMAGE_KEYS, preprocess_rgb_image, validate_observation
from stack_scene import CAMERA_NAMES, TASK_PROMPT


def apply_experimental_camera_calibration(
    model: mujoco.MjModel,
    calibration: dict | Path,
) -> dict[str, dict]:
    if isinstance(calibration, Path):
        calibration = json.loads(calibration.read_text())
    best = {
        camera: dict(calibration["search"][camera]["best"])
        for camera in CAMERA_NAMES
    }
    _lighten_skybox(model)
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
    return best


def render_experimental_policy_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera_parameters: dict[str, dict],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source_images: dict[str, np.ndarray] = {}
    for camera in CAMERA_NAMES:
        renderer.update_scene(data, camera=camera)
        image = renderer.render().copy()
        if camera_parameters[camera].get("vertical_flip"):
            image = image[::-1].copy()
        source_images[camera] = image
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
    return observation, source_images
