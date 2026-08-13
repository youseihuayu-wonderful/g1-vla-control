#!/usr/bin/env python3
"""Search coarse G1 camera geometry against synchronized public frame 0."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy import ndimage

from stack_scene import CAMERA_NAMES, build_model, reset_to_reference_pose

ROOT = Path(__file__).resolve().parent


def _mask_components(image: np.ndarray) -> dict[str, dict | None]:
    rgb = image.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    masks = {
        "red": (r > 90) & (r > 1.30 * g) & (r > 1.25 * b),
        "blue": (b > 70) & (b > 1.25 * r) & (b > 1.12 * g),
        "yellow": (
            (r > 110) & (g > 60) & (r > 1.30 * b)
            & (g > 1.10 * b) & (r > 0.90 * g)
        ),
    }
    output: dict[str, dict | None] = {}
    height, width = image.shape[:2]
    for color, mask in masks.items():
        labels, count = ndimage.label(mask)
        candidates = []
        for label in range(1, count + 1):
            y, x = np.where(labels == label)
            if not 150 <= len(x) <= 80_000:
                continue
            area = len(x) / mask.size
            bbox_width = (x.max() - x.min() + 1) / width
            bbox_height = (y.max() - y.min() + 1) / height
            if bbox_width > 0.8 or bbox_height > 0.8:
                continue
            candidates.append({
                "pixels": int(len(x)),
                "area": float(area),
                "center": [float(x.mean() / width), float(y.mean() / height)],
                "bbox": [
                    float(x.min() / width), float(y.min() / height),
                    float((x.max() + 1) / width),
                    float((y.max() + 1) / height),
                ],
            })
        output[color] = max(candidates, key=lambda item: item["pixels"]) if candidates else None
    return output


def _objective(simulation: dict, reference: dict) -> tuple[float, dict]:
    total = 0.0
    comparisons = {}
    for color in ("red", "blue", "yellow"):
        sim = simulation.get(color)
        ref = reference.get(color)
        if ref is None:
            continue
        if sim is None:
            total += 25.0
            comparisons[color] = {"missing": True}
            continue
        center_error = float(np.linalg.norm(
            np.asarray(sim["center"]) - np.asarray(ref["center"])
        ))
        log_area_error = float(np.log((sim["area"] + 1e-8) / (ref["area"] + 1e-8)))
        score = 20.0 * center_error**2 + log_area_error**2
        total += score
        comparisons[color] = {
            "center_error": center_error,
            "area_ratio": sim["area"] / ref["area"],
            "score": score,
        }
    return total, comparisons


def _quat_from_xyaxes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x / np.linalg.norm(x)
    y = y - x * np.dot(x, y)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    matrix = np.column_stack((x, y, z))
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, matrix.ravel())
    return quaternion


def _look_at_quaternion(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    look = target - position
    look /= np.linalg.norm(look)
    x = np.cross(look, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(x) < 1e-8:
        x = np.array([0.0, -1.0, 0.0])
    x /= np.linalg.norm(x)
    z = -look
    y = np.cross(z, x)
    return _quat_from_xyaxes(x, y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    references = {
        camera: np.asarray(Image.open(args.reference_dir / f"{camera}.png").convert("RGB"))
        for camera in CAMERA_NAMES
    }
    reference_features = {
        camera: _mask_components(image) for camera, image in references.items()
    }

    model = build_model()
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    hold = data.ctrl.copy()
    for _ in range(250):
        data.ctrl[:] = hold
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)

    def render(camera: str) -> np.ndarray:
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    report = {"reference_features": reference_features, "search": {}}
    try:
        # World-fixed high camera: move it in front of the torso and vary view.
        high = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_left_high")
        high_results = []
        for x, z, target_x, fovy in itertools.product(
            np.linspace(0.14, 0.30, 5),
            np.linspace(1.25, 1.49, 5),
            (0.44, 0.50, 0.56),
            (48.0, 56.0, 64.0, 72.0),
        ):
            position = np.array([x, 0.0, z])
            target = np.array([target_x, 0.0, 0.84])
            model.cam_pos[high] = position
            model.cam_quat[high] = _look_at_quaternion(position, target)
            model.cam_fovy[high] = fovy
            image = render("cam_left_high")
            features = _mask_components(image)
            score, comparison = _objective(
                features, reference_features["cam_left_high"]
            )
            high_results.append({
                "score": score,
                "position": position.tolist(),
                "target": target.tolist(),
                "fovy": fovy,
                "features": features,
                "comparison": comparison,
                "image": image,
            })
        high_results.sort(key=lambda item: item["score"])
        best_high = high_results[0]
        Image.fromarray(best_high.pop("image")).save(
            args.output_dir / "cam_left_high_best.png"
        )
        report["search"]["cam_left_high"] = {
            "best": best_high,
            "top5": [
                {key: value for key, value in item.items() if key != "image"}
                for item in high_results[:5]
            ],
        }

        # Body-fixed wrist cameras share mounting geometry.
        wrist_results = []
        left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_left_wrist")
        right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_right_wrist")
        for forward, vertical, pitch_deg, fovy in itertools.product(
            np.linspace(-0.08, 0.04, 7),
            (-0.03, 0.0, 0.03),
            (10.0, 20.0, 30.0, 40.0),
            (75.0, 90.0, 105.0, 120.0),
        ):
            pitch = np.deg2rad(pitch_deg)
            position = np.array([vertical, forward, 0.0])
            xaxis = np.array([1.0, 0.0, 0.0])
            yaxis = np.array([0.0, np.sin(pitch), np.cos(pitch)])
            quaternion = _quat_from_xyaxes(xaxis, yaxis)
            for camera_id in (left, right):
                model.cam_pos[camera_id] = position
                model.cam_quat[camera_id] = quaternion
                model.cam_fovy[camera_id] = fovy
            left_image = render("cam_left_wrist")
            right_image = render("cam_right_wrist")
            left_features = _mask_components(left_image)
            right_features = _mask_components(right_image)
            left_score, left_comparison = _objective(
                left_features, reference_features["cam_left_wrist"]
            )
            right_score, right_comparison = _objective(
                right_features, reference_features["cam_right_wrist"]
            )
            wrist_results.append({
                "score": left_score + right_score,
                "position": position.tolist(),
                "pitch_deg": pitch_deg,
                "fovy": fovy,
                "left_features": left_features,
                "right_features": right_features,
                "left_comparison": left_comparison,
                "right_comparison": right_comparison,
                "left_image": left_image,
                "right_image": right_image,
            })
        wrist_results.sort(key=lambda item: item["score"])
        best_wrist = wrist_results[0]
        Image.fromarray(best_wrist.pop("left_image")).save(
            args.output_dir / "cam_left_wrist_best.png"
        )
        Image.fromarray(best_wrist.pop("right_image")).save(
            args.output_dir / "cam_right_wrist_best.png"
        )
        report["search"]["wrist_shared"] = {
            "best": best_wrist,
            "top5": [
                {
                    key: value for key, value in item.items()
                    if key not in {"left_image", "right_image"}
                }
                for item in wrist_results[:5]
            ],
        }
    finally:
        renderer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(json.dumps({
        "high": report["search"]["cam_left_high"]["best"],
        "wrist": report["search"]["wrist_shared"]["best"],
    }, indent=2))


if __name__ == "__main__":
    main()
