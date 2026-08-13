#!/usr/bin/env python3
"""Second-pass camera search with gripper silhouettes and independent wrists."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from camera_calibration_search import (
    _look_at_quaternion,
    _mask_components,
    _objective as color_objective,
    _quat_from_xyaxes,
)
from stack_scene import CAMERA_NAMES, build_model, reset_to_reference_pose


def _features(image: np.ndarray) -> dict:
    gray = image.astype(np.float32).mean(axis=2)
    dark = gray < 65.0
    row_profile = [
        float(np.mean(part)) for part in np.array_split(dark, 8, axis=0)
    ]
    column_profile = [
        float(np.mean(part)) for part in np.array_split(dark, 8, axis=1)
    ]
    return {
        "objects": _mask_components(image),
        "dark_row_profile": row_profile,
        "dark_column_profile": column_profile,
        "luminance_mean": float(gray.mean()),
    }


def _objective(simulation: dict, reference: dict, *, high: bool) -> tuple[float, dict]:
    color_score, color_comparison = color_objective(
        simulation["objects"], reference["objects"]
    )
    row_error = float(np.mean(np.square(
        np.asarray(simulation["dark_row_profile"])
        - np.asarray(reference["dark_row_profile"])
    )))
    column_error = float(np.mean(np.square(
        np.asarray(simulation["dark_column_profile"])
        - np.asarray(reference["dark_column_profile"])
    )))
    luminance_error = (
        (simulation["luminance_mean"] - reference["luminance_mean"]) / 100.0
    ) ** 2
    # Silhouette matters most for wrist cameras; high view also receives a
    # center/bottom torso penalty through row/column profiles.
    silhouette_weight = 80.0 if high else 120.0
    total = (
        color_score
        + silhouette_weight * row_error
        + 30.0 * column_error
        + 0.5 * luminance_error
    )
    return float(total), {
        "color_score": color_score,
        "color_comparison": color_comparison,
        "dark_row_mse": row_error,
        "dark_column_mse": column_error,
        "luminance_error": luminance_error,
    }


def _wrist_quaternion(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    forward = np.array([
        np.sin(yaw) * np.cos(pitch),
        np.cos(yaw) * np.cos(pitch),
        -np.sin(pitch),
    ])
    forward /= np.linalg.norm(forward)
    xaxis = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    zaxis = -forward
    yaxis = np.cross(zaxis, xaxis)
    return _quat_from_xyaxes(xaxis, yaxis)


def _lighten_skybox(model: mujoco.MjModel) -> None:
    for texture in range(model.ntex):
        if model.tex_type[texture] != mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            continue
        start = int(model.tex_adr[texture])
        pixel_count = int(
            model.tex_width[texture]
            * model.tex_height[texture]
            * model.tex_nchannel[texture]
        )
        pixels = model.tex_data[start:start + pixel_count]
        pixels[:] = 225


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    references = {
        name: np.asarray(Image.open(args.reference_dir / f"{name}.png").convert("RGB"))
        for name in CAMERA_NAMES
    }
    reference_features = {name: _features(image) for name, image in references.items()}

    model = build_model()
    _lighten_skybox(model)
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    hold = data.ctrl.copy()
    for _ in range(250):
        data.ctrl[:] = hold
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)

    def render(name: str) -> np.ndarray:
        renderer.update_scene(data, camera=name)
        return renderer.render().copy()

    report = {"reference_features": reference_features, "search": {}}
    try:
        high_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_left_high"
        )
        high_results = []
        for x, z, target_x, target_z, fovy in itertools.product(
            (0.16, 0.22, 0.28, 0.34),
            (1.18, 1.28, 1.38, 1.48),
            (0.42, 0.48, 0.54),
            (0.80, 0.86, 0.92),
            (45.0, 55.0, 65.0, 75.0),
        ):
            position = np.array([x, 0.0, z])
            target = np.array([target_x, 0.0, target_z])
            model.cam_pos[high_id] = position
            model.cam_quat[high_id] = _look_at_quaternion(position, target)
            model.cam_fovy[high_id] = fovy
            image = render("cam_left_high")
            features = _features(image)
            score, evidence = _objective(
                features, reference_features["cam_left_high"], high=True
            )
            high_results.append({
                "score": score,
                "position": position.tolist(),
                "target": target.tolist(),
                "fovy": fovy,
                "features": features,
                "evidence": evidence,
                "image": image,
            })
        high_results.sort(key=lambda item: item["score"])
        best_high = high_results[0]
        Image.fromarray(best_high["image"]).save(
            args.output_dir / "cam_left_high_best.png"
        )
        report["search"]["cam_left_high"] = {
            "best": {k: v for k, v in best_high.items() if k != "image"},
            "top5": [
                {k: v for k, v in item.items() if k != "image"}
                for item in high_results[:5]
            ],
        }

        for camera in ("cam_left_wrist", "cam_right_wrist"):
            camera_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, camera
            )
            results = []
            for lateral, forward, vertical, pitch, yaw, fovy in itertools.product(
                (-0.05, 0.0, 0.05),
                (-0.20, -0.10, 0.0),
                (-0.05, 0.0, 0.05),
                (-10.0, 10.0, 30.0),
                (-10.0, 0.0, 10.0),
                (110.0, 130.0, 150.0),
            ):
                position = np.array([lateral, forward, vertical])
                model.cam_pos[camera_id] = position
                model.cam_quat[camera_id] = _wrist_quaternion(pitch, yaw)
                model.cam_fovy[camera_id] = fovy
                raw = render(camera)
                for vertical_flip in (False, True):
                    image = raw[::-1].copy() if vertical_flip else raw
                    features = _features(image)
                    score, evidence = _objective(
                        features, reference_features[camera], high=False
                    )
                    results.append({
                        "score": score,
                        "position": position.tolist(),
                        "pitch_deg": pitch,
                        "yaw_deg": yaw,
                        "fovy": fovy,
                        "vertical_flip": vertical_flip,
                        "features": features,
                        "evidence": evidence,
                        "image": image,
                    })
            results.sort(key=lambda item: item["score"])
            best = results[0]
            Image.fromarray(best["image"]).save(
                args.output_dir / f"{camera}_best.png"
            )
            report["search"][camera] = {
                "best": {k: v for k, v in best.items() if k != "image"},
                "top5": [
                    {k: v for k, v in item.items() if k != "image"}
                    for item in results[:5]
                ],
            }
    finally:
        renderer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(json.dumps({
        name: report["search"][name]["best"]
        for name in ("cam_left_high", "cam_left_wrist", "cam_right_wrist")
    }, indent=2))


if __name__ == "__main__":
    main()
