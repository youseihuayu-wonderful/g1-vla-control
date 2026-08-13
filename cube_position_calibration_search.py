#!/usr/bin/env python3
"""Calibrate table cube XY positions against three synchronized public views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from camera_calibration_search import _look_at_quaternion
from camera_calibration_search_v2 import (
    _features, _lighten_skybox, _wrist_quaternion,
)
from stack_scene import CAMERA_NAMES, build_model, reset_to_reference_pose


def _color_score(sim: dict | None, ref: dict | None) -> tuple[float, dict]:
    if ref is None:
        if sim is None:
            return 0.0, {"both_not_visible": True}
        score = 50.0 * sim["area"]
        return score, {"unexpected_visibility": True, "area": sim["area"], "score": score}
    if sim is None:
        return 25.0, {"missing": True, "score": 25.0}
    center_error = float(np.linalg.norm(
        np.asarray(sim["center"]) - np.asarray(ref["center"])
    ))
    log_area_error = float(np.log(
        (sim["area"] + 1e-8) / (ref["area"] + 1e-8)
    ))
    score = 30.0 * center_error**2 + log_area_error**2
    return score, {
        "center_error": center_error,
        "area_ratio": sim["area"] / ref["area"],
        "score": score,
    }


def _apply_cameras(model: mujoco.MjModel, calibration: dict) -> None:
    best = calibration["search"]
    high = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_left_high"
    )
    entry = best["cam_left_high"]["best"]
    position = np.asarray(entry["position"])
    target = np.asarray(entry["target"])
    model.cam_pos[high] = position
    model.cam_quat[high] = _look_at_quaternion(position, target)
    model.cam_fovy[high] = entry["fovy"]
    for camera in ("cam_left_wrist", "cam_right_wrist"):
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera
        )
        entry = best[camera]["best"]
        model.cam_pos[camera_id] = np.asarray(entry["position"])
        model.cam_quat[camera_id] = _wrist_quaternion(
            entry["pitch_deg"], entry["yaw_deg"]
        )
        model.cam_fovy[camera_id] = entry["fovy"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--camera-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_features = {
        camera: _features(np.asarray(
            Image.open(args.reference_dir / f"{camera}.png").convert("RGB")
        ))
        for camera in CAMERA_NAMES
    }
    calibration = json.loads(args.camera_calibration.read_text())
    model = build_model()
    _lighten_skybox(model)
    _apply_cameras(model, calibration)
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    hold = data.ctrl.copy()
    for _ in range(250):
        data.ctrl[:] = hold
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)

    camera_flip = {
        camera: bool(
            calibration["search"][camera]["best"].get("vertical_flip")
        )
        for camera in CAMERA_NAMES
    }

    def render_features() -> tuple[dict, dict]:
        images, features = {}, {}
        for camera in CAMERA_NAMES:
            renderer.update_scene(data, camera=camera)
            image = renderer.render().copy()
            if camera_flip[camera]:
                image = image[::-1].copy()
            images[camera] = image
            features[camera] = _features(image)
        return images, features

    results = {}
    try:
        for color in ("red", "blue", "yellow"):
            body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"{color}_cube"
            )
            joint = int(model.body_jntadr[body])
            qpos = int(model.jnt_qposadr[joint])
            original = data.qpos[qpos:qpos + 3].copy()
            candidates = []
            for x in np.linspace(0.38, 0.70, 17):
                for y in np.linspace(-0.32, 0.32, 17):
                    data.qpos[qpos:qpos + 3] = [x, y, original[2]]
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    images, features = render_features()
                    score = 0.0
                    evidence = {}
                    for camera in CAMERA_NAMES:
                        value, comparison = _color_score(
                            features[camera]["objects"][color],
                            reference_features[camera]["objects"][color],
                        )
                        score += value
                        evidence[camera] = comparison
                    candidates.append({
                        "score": float(score),
                        "position": [float(x), float(y), float(original[2])],
                        "evidence": evidence,
                        "images": images,
                    })
            candidates.sort(key=lambda item: item["score"])
            best = candidates[0]
            data.qpos[qpos:qpos + 3] = best["position"]
            mujoco.mj_forward(model, data)
            for camera, image in best["images"].items():
                Image.fromarray(image).save(
                    args.output_dir / f"{color}_{camera}_best.png"
                )
            results[color] = {
                "best": {k: v for k, v in best.items() if k != "images"},
                "top10": [
                    {k: v for k, v in candidate.items() if k != "images"}
                    for candidate in candidates[:10]
                ],
            }
    finally:
        renderer.close()

    report = {
        "scope": "Alternating coarse cube-position calibration with fixed experimental camera candidate.",
        "camera_calibration": str(args.camera_calibration),
        "search_bounds": {
            "x_m": [0.38, 0.70],
            "y_m": [-0.32, 0.32],
            "grid": [17, 17],
        },
        "results": results,
        "g1_contract_verified": False,
        "g1_sim_eligible": False,
        "g1_execution_enabled": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(json.dumps({
        color: results[color]["best"] for color in results
    }, indent=2))


if __name__ == "__main__":
    main()
