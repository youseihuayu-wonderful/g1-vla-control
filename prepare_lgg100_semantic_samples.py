#!/usr/bin/env python3
"""Prepare hash-bound public episode samples for LGG100 semantic inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
import pyarrow.parquet as pq

from dataset_contract_audit import DATASET, _fk_transform
from g1_policy_contract import ACTION_HORIZON, preprocess_rgb_image
from stack_scene import CAMERA_NAMES, TASK_PROMPT

DATASET_REVISION = "c468ef259ff8bfad1b3b0e7b2c1f45efbb0b30ba"
ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = Path(os.environ.get(
    "G1_VLA_DATA_CACHE", ROOT / "results" / "semantic_cache"
))


def _download(relative_path: str, cache: Path) -> Path:
    destination = cache / relative_path
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://huggingface.co/datasets/{DATASET}/resolve/"
        f"{DATASET_REVISION}/{relative_path}"
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)
    return destination


def _decode_selected_frames(path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Install PyAV in the G1 simulation environment") from exc
    selected: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in indices:
                selected[index] = frame.to_ndarray(format="rgb24")
            if len(selected) == len(indices):
                break
    missing = sorted(indices - set(selected))
    if missing:
        raise RuntimeError(f"Video {path} is missing frames {missing}")
    return selected


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--samples-per-episode", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "lgg100_semantic_samples.npz",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "results" / "lgg100_semantic_samples_manifest.json",
    )
    args = parser.parse_args()
    if not args.episodes or args.samples_per_episode < 1:
        raise SystemExit("At least one episode and one sample per episode are required")

    images = {camera: [] for camera in CAMERA_NAMES}
    states: list[np.ndarray] = []
    reference_actions: list[np.ndarray] = []
    episode_ids: list[int] = []
    frame_ids: list[int] = []
    sample_timestamps: list[float] = []
    records: list[dict] = []

    for episode in args.episodes:
        parquet_path = _download(
            f"data/chunk-000/episode_{episode:06d}.parquet", args.cache_dir
        )
        table = pq.read_table(
            parquet_path, columns=["observation.state", "action", "timestamp"]
        )
        raw_state = np.asarray(table.column("observation.state").to_pylist())
        raw_action = np.asarray(table.column("action").to_pylist())
        timestamps = np.asarray(table.column("timestamp").to_numpy(), dtype=np.float64)
        if raw_state.shape != raw_action.shape or raw_state.shape[1] != 16:
            raise RuntimeError(f"Unexpected episode arrays {raw_state.shape} {raw_action.shape}")
        maximum_start = len(raw_action) - ACTION_HORIZON
        if maximum_start < 1:
            raise RuntimeError(f"Episode {episode} is shorter than the action horizon")
        indices = np.linspace(
            0, maximum_start, args.samples_per_episode + 2, dtype=int
        )[1:-1]
        indices = np.unique(indices)
        if len(indices) != args.samples_per_episode:
            raise RuntimeError("Sampling produced duplicate frame indices")
        index_set = set(int(index) for index in indices)

        camera_frames: dict[str, dict[int, np.ndarray]] = {}
        for camera in CAMERA_NAMES:
            video_path = _download(
                f"videos/chunk-000/observation.images.{camera}/"
                f"episode_{episode:06d}.mp4",
                args.cache_dir,
            )
            camera_frames[camera] = _decode_selected_frames(video_path, index_set)

        eef_state = _fk_transform(raw_state)
        eef_action = _fk_transform(raw_action)
        for frame in indices:
            frame = int(frame)
            processed = {
                camera: preprocess_rgb_image(camera_frames[camera][frame])
                for camera in CAMERA_NAMES
            }
            state = eef_state[frame].astype(np.float32)
            action_chunk = eef_action[frame:frame + ACTION_HORIZON].astype(np.float64)
            for camera in CAMERA_NAMES:
                images[camera].append(processed[camera])
            states.append(state)
            reference_actions.append(action_chunk)
            episode_ids.append(episode)
            frame_ids.append(frame)
            sample_timestamps.append(float(timestamps[frame]))
            records.append({
                "episode": episode,
                "frame": frame,
                "timestamp_s": float(timestamps[frame]),
                "state_sha256": _array_sha256(state),
                "reference_action_sha256": _array_sha256(action_chunk),
                "image_sha256": {
                    camera: _array_sha256(processed[camera])
                    for camera in CAMERA_NAMES
                },
            })

    payload = {
        "cam_left_high": np.stack(images["cam_left_high"]),
        "cam_left_wrist": np.stack(images["cam_left_wrist"]),
        "cam_right_wrist": np.stack(images["cam_right_wrist"]),
        "state": np.stack(states),
        "reference_actions": np.stack(reference_actions),
        "episode": np.asarray(episode_ids, dtype=np.int32),
        "frame": np.asarray(frame_ids, dtype=np.int32),
        "timestamp": np.asarray(sample_timestamps, dtype=np.float64),
        "prompt": np.asarray(TASK_PROMPT),
        "dataset_revision": np.asarray(DATASET_REVISION),
        "candidate_eef_offset_m": np.asarray(0.050),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    manifest = {
        "scope": "Public real-episode observations and candidate FK references; no neural inference.",
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "episodes": list(args.episodes),
        "samples_per_episode": args.samples_per_episode,
        "sample_count": len(states),
        "action_horizon": ACTION_HORIZON,
        "candidate_transform": {
            "frame": "pelvis",
            "eef_offset_m": [0.050, 0.0, 0.0],
            "quaternion": "xyzw",
            "exactly_author_confirmed": False,
        },
        "samples": records,
        "npz": str(args.output),
        "npz_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)
    print(args.manifest)
    print(json.dumps({
        "samples": len(states),
        "episodes": list(args.episodes),
        "npz_sha256": manifest["npz_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
