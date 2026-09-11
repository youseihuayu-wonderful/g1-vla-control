#!/usr/bin/env python3
"""Offline trajectory quality analyzer for cached G1/LGG100 demonstrations.

This tool is read-only: it loads cached parquet episodes, reconstructs the
pelvis-frame EEF actions used by the frozen contract, and marks idle, coarse,
fine, high-speed, and jitter/anomaly intervals. It never opens a robot command
transport and never authorizes hardware motion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEFAULT_CACHE = Path.home() / ".cache" / "g1_vla_control" / "stack_the_cubes"

RAW_JOINTS = slice(0, 14)
GRIPPERS = slice(14, 16)
LEFT_EEF_POS = slice(0, 3)
RIGHT_EEF_POS = slice(7, 10)


@dataclass(frozen=True)
class AnalyzerConfig:
    idle_joint_speed_rad_s: float = 0.02
    idle_gripper_speed_rad_s: float = 0.02
    gripper_activity_speed_rad_s: float = 0.25
    minimum_segment_s: float = 0.20
    high_quantile: float = 0.99
    spike_quantile: float = 0.995
    maximum_reported_segments_per_kind: int = 20
    maximum_reported_spikes_per_kind: int = 20


def _require_strict_timestamps(timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 4:
        raise ValueError("timestamps must be a 1-D array with at least four samples")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps must be finite")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return timestamps


def _require_action_matrix(values: np.ndarray, *, rows: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (rows, 16):
        raise ValueError(f"{name} must have shape ({rows}, 16), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


def _safe_quantile(values: Sequence[np.ndarray], quantile: float, fallback: float = 0.0) -> float:
    arrays = [np.ravel(array) for array in values if np.size(array)]
    if not arrays:
        return float(fallback)
    joined = np.concatenate(arrays)
    if not joined.size:
        return float(fallback)
    return float(np.quantile(joined, quantile))


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _top_indices(values: np.ndarray, threshold: float, limit: int) -> list[int]:
    if values.size == 0:
        return []
    indices = np.flatnonzero(values > threshold)
    if len(indices) <= limit:
        return [int(index) for index in indices]
    ranked = indices[np.argsort(values[indices])[-limit:]][::-1]
    return [int(index) for index in ranked]


def _segments_from_interval_mask(
    mask: np.ndarray,
    timestamps: np.ndarray,
    *,
    minimum_segment_s: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Convert a T-1 interval mask into compact frame-index segments."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != len(timestamps) - 1:
        raise ValueError("interval mask must have length len(timestamps)-1")

    segments: list[dict[str, Any]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            _append_segment(segments, timestamps, start, index, minimum_segment_s)
            start = None
    if start is not None:
        _append_segment(segments, timestamps, start, len(mask), minimum_segment_s)

    segments.sort(key=lambda item: item["duration_s"], reverse=True)
    return segments[:limit]


def _append_segment(
    segments: list[dict[str, Any]],
    timestamps: np.ndarray,
    start_interval: int,
    end_interval_exclusive: int,
    minimum_segment_s: float,
) -> None:
    start_frame = start_interval
    end_frame = end_interval_exclusive
    duration = float(timestamps[end_frame] - timestamps[start_frame])
    if duration + 1e-12 < minimum_segment_s:
        return
    segments.append({
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "start_s": float(timestamps[start_frame]),
        "end_s": float(timestamps[end_frame]),
        "duration_s": duration,
    })


def compute_signals(
    actions: np.ndarray,
    timestamps: np.ndarray,
    *,
    eef_actions: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return interval-level speed/acceleration/jerk signals for one episode."""
    timestamps = _require_strict_timestamps(timestamps)
    actions = _require_action_matrix(actions, rows=len(timestamps), name="actions")
    if eef_actions is not None:
        eef_actions = _require_action_matrix(eef_actions, rows=len(timestamps), name="eef_actions")

    dt = np.diff(timestamps)
    joint_velocity = np.diff(actions[:, RAW_JOINTS], axis=0) / dt[:, None]
    gripper_velocity = np.diff(actions[:, GRIPPERS], axis=0) / dt[:, None]
    joint_speed = np.max(np.abs(joint_velocity), axis=1)
    gripper_speed = np.max(np.abs(gripper_velocity), axis=1)

    if len(joint_velocity) >= 2:
        accel_dt = dt[1:, None]
        joint_acceleration = np.diff(joint_velocity, axis=0) / accel_dt
        joint_acceleration_max = np.max(np.abs(joint_acceleration), axis=1)
    else:
        joint_acceleration = np.empty((0, 14), dtype=np.float64)
        joint_acceleration_max = np.empty(0, dtype=np.float64)

    if len(joint_acceleration) >= 2:
        jerk_dt = dt[2:, None]
        joint_jerk = np.diff(joint_acceleration, axis=0) / jerk_dt
        joint_jerk_max = np.max(np.abs(joint_jerk), axis=1)
    else:
        joint_jerk_max = np.empty(0, dtype=np.float64)

    if eef_actions is None:
        eef_speed = np.zeros(len(dt), dtype=np.float64)
        eef_acceleration_max = np.empty(0, dtype=np.float64)
        eef_jerk_max = np.empty(0, dtype=np.float64)
    else:
        left_velocity = np.diff(eef_actions[:, LEFT_EEF_POS], axis=0) / dt[:, None]
        right_velocity = np.diff(eef_actions[:, RIGHT_EEF_POS], axis=0) / dt[:, None]
        eef_speed = np.maximum(
            np.linalg.norm(left_velocity, axis=1),
            np.linalg.norm(right_velocity, axis=1),
        )
        if len(left_velocity) >= 2:
            eef_acceleration = np.maximum(
                np.linalg.norm(np.diff(left_velocity, axis=0) / dt[1:, None], axis=1),
                np.linalg.norm(np.diff(right_velocity, axis=0) / dt[1:, None], axis=1),
            )
            eef_acceleration_max = eef_acceleration
        else:
            eef_acceleration_max = np.empty(0, dtype=np.float64)
        if len(eef_acceleration_max) >= 2:
            # Jerk magnitude from EEF acceleration scalar deltas; sufficient for
            # anomaly localization without changing control geometry.
            eef_jerk_max = np.abs(np.diff(eef_acceleration_max) / dt[2:])
        else:
            eef_jerk_max = np.empty(0, dtype=np.float64)

    return {
        "dt": dt,
        "joint_speed": joint_speed,
        "gripper_speed": gripper_speed,
        "joint_acceleration": joint_acceleration_max,
        "joint_jerk": joint_jerk_max,
        "eef_speed": eef_speed,
        "eef_acceleration": eef_acceleration_max,
        "eef_jerk": eef_jerk_max,
    }


def derive_thresholds(
    episode_signals: Sequence[dict[str, np.ndarray]],
    config: AnalyzerConfig = AnalyzerConfig(),
) -> dict[str, float]:
    return {
        "idle_joint_speed_rad_s": float(config.idle_joint_speed_rad_s),
        "idle_gripper_speed_rad_s": float(config.idle_gripper_speed_rad_s),
        "gripper_activity_speed_rad_s": float(config.gripper_activity_speed_rad_s),
        "high_joint_speed_rad_s": _safe_quantile(
            [signals["joint_speed"] for signals in episode_signals], config.high_quantile
        ),
        "high_eef_speed_m_s": _safe_quantile(
            [signals["eef_speed"] for signals in episode_signals], config.high_quantile
        ),
        "high_gripper_speed_rad_s": _safe_quantile(
            [signals["gripper_speed"] for signals in episode_signals], config.high_quantile
        ),
        "joint_acceleration_spike_rad_s2": _safe_quantile(
            [signals["joint_acceleration"] for signals in episode_signals], config.spike_quantile
        ),
        "joint_jerk_spike_rad_s3": _safe_quantile(
            [signals["joint_jerk"] for signals in episode_signals], config.spike_quantile
        ),
        "eef_acceleration_spike_m_s2": _safe_quantile(
            [signals["eef_acceleration"] for signals in episode_signals], config.spike_quantile
        ),
        "eef_jerk_spike_m_s3": _safe_quantile(
            [signals["eef_jerk"] for signals in episode_signals], config.spike_quantile
        ),
        "fine_eef_speed_m_s": _safe_quantile(
            [signals["eef_speed"] for signals in episode_signals], 0.50
        ),
        "coarse_eef_speed_m_s": _safe_quantile(
            [signals["eef_speed"] for signals in episode_signals], 0.75
        ),
    }


def analyze_episode(
    name: str,
    actions: np.ndarray,
    timestamps: np.ndarray,
    *,
    eef_actions: np.ndarray | None,
    thresholds: dict[str, float],
    config: AnalyzerConfig = AnalyzerConfig(),
) -> dict[str, Any]:
    signals = compute_signals(actions, timestamps, eef_actions=eef_actions)
    timestamps = _require_strict_timestamps(timestamps)
    duration = float(timestamps[-1] - timestamps[0])

    idle_mask = (
        (signals["joint_speed"] <= thresholds["idle_joint_speed_rad_s"])
        & (signals["gripper_speed"] <= thresholds["idle_gripper_speed_rad_s"])
    )
    high_speed_mask = (
        (signals["joint_speed"] > thresholds["high_joint_speed_rad_s"])
        | (signals["eef_speed"] > thresholds["high_eef_speed_m_s"])
        | (signals["gripper_speed"] > thresholds["high_gripper_speed_rad_s"])
    )
    gripper_active = signals["gripper_speed"] > thresholds["gripper_activity_speed_rad_s"]
    fine_mask = (
        ~idle_mask
        & (
            gripper_active
            | (signals["eef_speed"] <= thresholds["fine_eef_speed_m_s"])
        )
    )
    coarse_mask = (
        ~idle_mask
        & ~gripper_active
        & (signals["eef_speed"] >= thresholds["coarse_eef_speed_m_s"])
    )

    joint_jerk_spikes = _top_indices(
        signals["joint_jerk"],
        thresholds["joint_jerk_spike_rad_s3"],
        config.maximum_reported_spikes_per_kind,
    )
    eef_jerk_spikes = _top_indices(
        signals["eef_jerk"],
        thresholds["eef_jerk_spike_m_s3"],
        config.maximum_reported_spikes_per_kind,
    )

    return {
        "name": name,
        "frames": int(len(timestamps)),
        "duration_s": duration,
        "median_dt_s": float(np.median(signals["dt"])),
        "effective_fps": float(1.0 / np.median(signals["dt"])),
        "motion_stats": {
            "joint_speed_rad_s": _stats(signals["joint_speed"]),
            "gripper_speed_rad_s": _stats(signals["gripper_speed"]),
            "eef_speed_m_s": _stats(signals["eef_speed"]),
            "joint_acceleration_rad_s2": _stats(signals["joint_acceleration"]),
            "joint_jerk_rad_s3": _stats(signals["joint_jerk"]),
            "eef_acceleration_m_s2": _stats(signals["eef_acceleration"]),
            "eef_jerk_m_s3": _stats(signals["eef_jerk"]),
        },
        "segments": {
            "idle_wait": _segments_from_interval_mask(
                idle_mask,
                timestamps,
                minimum_segment_s=config.minimum_segment_s,
                limit=config.maximum_reported_segments_per_kind,
            ),
            "fine_or_gripper_active": _segments_from_interval_mask(
                fine_mask,
                timestamps,
                minimum_segment_s=config.minimum_segment_s,
                limit=config.maximum_reported_segments_per_kind,
            ),
            "coarse_motion": _segments_from_interval_mask(
                coarse_mask,
                timestamps,
                minimum_segment_s=config.minimum_segment_s,
                limit=config.maximum_reported_segments_per_kind,
            ),
            "high_speed": _segments_from_interval_mask(
                high_speed_mask,
                timestamps,
                minimum_segment_s=0.0,
                limit=config.maximum_reported_segments_per_kind,
            ),
        },
        "spikes": {
            "joint_jerk": [
                {
                    "frame": int(index + 2),
                    "time_s": float(timestamps[index + 2]),
                    "value_rad_s3": float(signals["joint_jerk"][index]),
                }
                for index in joint_jerk_spikes
            ],
            "eef_jerk": [
                {
                    "frame": int(index + 2),
                    "time_s": float(timestamps[index + 2]),
                    "value_m_s3": float(signals["eef_jerk"][index]),
                }
                for index in eef_jerk_spikes
            ],
        },
        "counts": {
            "idle_interval_count": int(np.count_nonzero(idle_mask)),
            "fine_interval_count": int(np.count_nonzero(fine_mask)),
            "coarse_interval_count": int(np.count_nonzero(coarse_mask)),
            "high_speed_interval_count": int(np.count_nonzero(high_speed_mask)),
            "joint_jerk_spike_count": int(
                np.count_nonzero(signals["joint_jerk"] > thresholds["joint_jerk_spike_rad_s3"])
            ),
            "eef_jerk_spike_count": int(
                np.count_nonzero(signals["eef_jerk"] > thresholds["eef_jerk_spike_m_s3"])
            ),
        },
    }


def _episode_paths(cache: Path, max_episodes: int | None) -> list[Path]:
    paths = sorted(cache.glob("episode_*.parquet"))
    if max_episodes is not None:
        paths = paths[:max_episodes]
    if not paths:
        raise FileNotFoundError(
            f"No cached episode_*.parquet files found in {cache}. "
            "Run dataset_contract_audit.py first to populate the cache."
        )
    return paths


def _load_raw_episodes(paths: Sequence[Path]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    episodes: list[dict[str, Any]] = []
    for path in paths:
        table = pq.read_table(path, columns=["observation.state", "action", "timestamp"])
        episodes.append({
            "name": path.stem,
            "state": np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64),
            "action": np.asarray(table.column("action").to_pylist(), dtype=np.float64),
            "timestamp": np.asarray(table.column("timestamp"), dtype=np.float64),
        })
    return episodes


def analyze_cached_dataset(
    *,
    cache: Path = DEFAULT_CACHE,
    output: Path = RESULTS / "trajectory_analysis.json",
    max_episodes: int | None = None,
    config: AnalyzerConfig = AnalyzerConfig(),
) -> dict[str, Any]:
    paths = _episode_paths(cache, max_episodes)
    raw_episodes = _load_raw_episodes(paths)

    from dataset_contract_audit import _fk_transform

    eef_actions = _fk_transform(np.concatenate([episode["action"] for episode in raw_episodes]))
    cursor = 0
    episode_signals = []
    for episode in raw_episodes:
        count = len(episode["action"])
        episode["eef_action"] = eef_actions[cursor : cursor + count]
        cursor += count
        episode_signals.append(
            compute_signals(
                episode["action"],
                episode["timestamp"],
                eef_actions=episode["eef_action"],
            )
        )

    thresholds = derive_thresholds(episode_signals, config)
    episodes = [
        analyze_episode(
            episode["name"],
            episode["action"],
            episode["timestamp"],
            eef_actions=episode["eef_action"],
            thresholds=thresholds,
            config=config,
        )
        for episode in raw_episodes
    ]

    def sum_count(key: str) -> int:
        return int(sum(episode["counts"][key] for episode in episodes))

    report = {
        "schema_version": "g1_trajectory_analysis_v1",
        "scope": (
            "Offline analysis of cached LGG100 Stack-the-cubes demonstrations. "
            "Segments are labels for data cleaning and retiming design only; no robot motion."
        ),
        "cache": str(cache),
        "episode_count": len(episodes),
        "frame_count": int(sum(episode["frames"] for episode in episodes)),
        "thresholds": thresholds,
        "summary": {
            "idle_interval_count": sum_count("idle_interval_count"),
            "fine_interval_count": sum_count("fine_interval_count"),
            "coarse_interval_count": sum_count("coarse_interval_count"),
            "high_speed_interval_count": sum_count("high_speed_interval_count"),
            "joint_jerk_spike_count": sum_count("joint_jerk_spike_count"),
            "eef_jerk_spike_count": sum_count("eef_jerk_spike_count"),
            "episodes_with_idle_wait": int(sum(
                bool(episode["segments"]["idle_wait"]) for episode in episodes
            )),
            "episodes_with_high_speed": int(sum(
                bool(episode["segments"]["high_speed"]) for episode in episodes
            )),
            "episodes_with_joint_jerk_spikes": int(sum(
                bool(episode["spikes"]["joint_jerk"]) for episode in episodes
            )),
        },
        "episodes": episodes,
        "decision": {
            "trajectory_analyzer_completed": True,
            "retiming_scheduler_completed": False,
            "hardware_execution_performed": False,
            "robot_motion_allowed": False,
        },
    }
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=RESULTS / "trajectory_analysis.json")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--idle-joint-speed", type=_positive_float, default=0.02)
    parser.add_argument("--idle-gripper-speed", type=_positive_float, default=0.02)
    parser.add_argument("--gripper-activity-speed", type=_positive_float, default=0.25)
    parser.add_argument("--minimum-segment-s", type=_positive_float, default=0.20)
    args = parser.parse_args()

    config = AnalyzerConfig(
        idle_joint_speed_rad_s=args.idle_joint_speed,
        idle_gripper_speed_rad_s=args.idle_gripper_speed,
        gripper_activity_speed_rad_s=args.gripper_activity_speed,
        minimum_segment_s=args.minimum_segment_s,
    )
    report = analyze_cached_dataset(
        cache=args.cache,
        output=args.output,
        max_episodes=args.max_episodes,
        config=config,
    )
    print(args.output)
    print(json.dumps({
        "episodes": report["episode_count"],
        "frames": report["frame_count"],
        "idle_intervals": report["summary"]["idle_interval_count"],
        "high_speed_intervals": report["summary"]["high_speed_interval_count"],
        "joint_jerk_spikes": report["summary"]["joint_jerk_spike_count"],
        "robot_motion_allowed": report["decision"]["robot_motion_allowed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
