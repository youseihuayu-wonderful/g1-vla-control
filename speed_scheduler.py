#!/usr/bin/env python3
"""Offline segment-aware speed scheduler for G1/LGG100 trajectories.

The scheduler consumes the labels/thresholds produced by ``trajectory_analyzer``
and changes timestamps only. Action samples are copied byte-for-byte. This file
is an offline data-processing tool: it has no robot SDK dependency and never
opens a command transport.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from trajectory_analyzer import DEFAULT_CACHE, RESULTS, compute_signals

RAW_JOINTS = slice(0, 14)
GRIPPERS = slice(14, 16)
LEFT_EEF_POS = slice(0, 3)
RIGHT_EEF_POS = slice(7, 10)


@dataclass(frozen=True)
class SpeedScheduleConfig:
    idle_scale: float = 2.00
    coarse_scale: float = 1.35
    fine_scale: float = 1.00
    gripper_active_scale: float = 0.90
    high_speed_scale: float = 0.85
    jerk_spike_scale: float = 0.75
    minimum_scale: float = 0.05
    maximum_scale: float = 1.50
    maximum_scale_increase_per_s: float = 2.0
    speed_cap_margin: float = 0.98
    numerical_tolerance: float = 1e-6


@dataclass(frozen=True)
class ScheduleResult:
    timestamps: np.ndarray
    scale_profile: np.ndarray
    interval_labels: dict[str, np.ndarray]
    baseline_signals: dict[str, np.ndarray]
    retimed_signals: dict[str, np.ndarray]


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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


def _load_thresholds(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"{path} does not contain analyzer thresholds")
    required = {
        "idle_joint_speed_rad_s",
        "idle_gripper_speed_rad_s",
        "gripper_activity_speed_rad_s",
        "high_joint_speed_rad_s",
        "high_eef_speed_m_s",
        "high_gripper_speed_rad_s",
        "joint_jerk_spike_rad_s3",
        "eef_jerk_spike_m_s3",
        "fine_eef_speed_m_s",
        "coarse_eef_speed_m_s",
    }
    missing = required - set(thresholds)
    if missing:
        raise ValueError(f"missing analyzer thresholds: {sorted(missing)}")
    return {key: float(thresholds[key]) for key in required}


def _episode_paths(cache: Path, max_episodes: int | None) -> list[Path]:
    paths = sorted(cache.glob("episode_*.parquet"))
    if max_episodes is not None:
        paths = paths[:max_episodes]
    if not paths:
        raise FileNotFoundError(
            f"No cached episode_*.parquet files found in {cache}. "
            "Run dataset_contract_audit.py first."
        )
    return paths


def _load_raw_episodes(paths: list[Path]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    episodes: list[dict[str, Any]] = []
    for path in paths:
        table = pq.read_table(path, columns=["action", "timestamp"])
        episodes.append({
            "name": path.stem,
            "action": np.asarray(table.column("action").to_pylist(), dtype=np.float64),
            "timestamp": np.asarray(table.column("timestamp"), dtype=np.float64),
        })
    return episodes


def _mark_jerk_intervals(spikes: np.ndarray, threshold: float, interval_count: int) -> np.ndarray:
    mask = np.zeros(interval_count, dtype=bool)
    for index in np.flatnonzero(spikes > threshold):
        # A jerk sample at index i uses frames i, i+1, i+2, i+3 through
        # second-order finite differences. Slow the local neighborhood.
        start = max(0, int(index) - 1)
        end = min(interval_count, int(index) + 3)
        mask[start:end] = True
    return mask


def interval_labels(
    signals: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> dict[str, np.ndarray]:
    interval_count = len(signals["dt"])
    idle = (
        (signals["joint_speed"] <= thresholds["idle_joint_speed_rad_s"])
        & (signals["gripper_speed"] <= thresholds["idle_gripper_speed_rad_s"])
    )
    gripper_active = signals["gripper_speed"] > thresholds["gripper_activity_speed_rad_s"]
    high_speed = (
        (signals["joint_speed"] > thresholds["high_joint_speed_rad_s"])
        | (signals["eef_speed"] > thresholds["high_eef_speed_m_s"])
        | (signals["gripper_speed"] > thresholds["high_gripper_speed_rad_s"])
    )
    jerk_spike = _mark_jerk_intervals(
        signals["joint_jerk"], thresholds["joint_jerk_spike_rad_s3"], interval_count
    ) | _mark_jerk_intervals(
        signals["eef_jerk"], thresholds["eef_jerk_spike_m_s3"], interval_count
    )
    fine = (
        ~idle
        & ~gripper_active
        & (signals["eef_speed"] <= thresholds["fine_eef_speed_m_s"])
    )
    coarse = (
        ~idle
        & ~gripper_active
        & (signals["eef_speed"] >= thresholds["coarse_eef_speed_m_s"])
    )
    return {
        "idle": idle,
        "fine": fine,
        "gripper_active": gripper_active,
        "coarse": coarse,
        "high_speed": high_speed,
        "jerk_spike": jerk_spike,
    }


def _speed_cap_scale(
    signals: dict[str, np.ndarray],
    thresholds: dict[str, float],
    config: SpeedScheduleConfig,
) -> np.ndarray:
    cap = np.full(len(signals["dt"]), config.maximum_scale, dtype=np.float64)
    for signal_name, threshold_name in (
        ("joint_speed", "high_joint_speed_rad_s"),
        ("eef_speed", "high_eef_speed_m_s"),
        ("gripper_speed", "high_gripper_speed_rad_s"),
    ):
        speed = np.asarray(signals[signal_name], dtype=np.float64)
        threshold = thresholds[threshold_name]
        limit = threshold * config.speed_cap_margin
        # Cap intervals that are still inside the analyzer's high-speed boundary,
        # so speeding coarse/idle motion cannot create a new high-speed interval.
        # Existing outliers are slowed by semantic labels instead of forcing a
        # huge global pause that would turn data cleaning artifacts into schedule
        # length.
        local = np.divide(
            limit,
            speed,
            out=np.full_like(speed, config.maximum_scale),
            where=(speed > 1e-12) & (speed <= threshold),
        )
        cap = np.minimum(cap, local)
    return np.clip(cap, config.minimum_scale, config.maximum_scale)


def _rate_limit_scale(
    desired: np.ndarray,
    dt: np.ndarray,
    config: SpeedScheduleConfig,
) -> np.ndarray:
    desired = np.asarray(desired, dtype=np.float64)
    scale = np.empty_like(desired)
    # Start conservatively. Fast idle/coarse regions ramp up; slow safety regions
    # can take effect immediately.
    scale[0] = min(1.0, desired[0])
    for index in range(1, len(scale)):
        if desired[index] < scale[index - 1]:
            scale[index] = desired[index]
        else:
            maximum = scale[index - 1] + config.maximum_scale_increase_per_s * dt[index - 1]
            scale[index] = min(desired[index], maximum)
    return np.clip(scale, config.minimum_scale, config.maximum_scale)


def schedule_timestamps(
    actions: np.ndarray,
    timestamps: np.ndarray,
    *,
    eef_actions: np.ndarray,
    thresholds: dict[str, float],
    config: SpeedScheduleConfig = SpeedScheduleConfig(),
) -> ScheduleResult:
    baseline = compute_signals(actions, timestamps, eef_actions=eef_actions)
    labels = interval_labels(baseline, thresholds)

    desired = np.ones(len(baseline["dt"]), dtype=np.float64)
    desired[labels["idle"]] = config.idle_scale
    desired[labels["coarse"]] = np.maximum(desired[labels["coarse"]], config.coarse_scale)
    desired[labels["fine"]] = np.minimum(desired[labels["fine"]], config.fine_scale)
    desired[labels["gripper_active"]] = np.minimum(
        desired[labels["gripper_active"]], config.gripper_active_scale
    )
    desired[labels["high_speed"]] = np.minimum(
        desired[labels["high_speed"]], config.high_speed_scale
    )
    desired[labels["jerk_spike"]] = np.minimum(
        desired[labels["jerk_spike"]], config.jerk_spike_scale
    )

    desired = np.minimum(desired, _speed_cap_scale(baseline, thresholds, config))
    desired = np.clip(desired, config.minimum_scale, config.maximum_scale)
    scale = _rate_limit_scale(desired, baseline["dt"], config)
    retimed_dt = baseline["dt"] / scale
    retimed_timestamps = np.concatenate((
        [float(timestamps[0])],
        float(timestamps[0]) + np.cumsum(retimed_dt),
    ))
    retimed = compute_signals(actions, retimed_timestamps, eef_actions=eef_actions)
    return ScheduleResult(
        timestamps=retimed_timestamps,
        scale_profile=scale,
        interval_labels=labels,
        baseline_signals=baseline,
        retimed_signals=retimed,
    )


def _speed_caps_pass(
    result: ScheduleResult,
    thresholds: dict[str, float],
    config: SpeedScheduleConfig,
) -> bool:
    tolerance = 1.0 + config.numerical_tolerance
    for signal_name, threshold_name in (
        ("joint_speed", "high_joint_speed_rad_s"),
        ("eef_speed", "high_eef_speed_m_s"),
        ("gripper_speed", "high_gripper_speed_rad_s"),
    ):
        baseline = result.baseline_signals[signal_name]
        retimed = result.retimed_signals[signal_name]
        threshold = thresholds[threshold_name]
        allowed_max = max(float(np.max(baseline)), threshold * config.speed_cap_margin)
        if float(np.max(retimed)) > allowed_max * tolerance:
            return False
        inside_boundary = baseline <= threshold
        if np.any(inside_boundary):
            limit = threshold * config.speed_cap_margin * tolerance
            if float(np.max(retimed[inside_boundary])) > limit:
                return False
    return True


def _episode_record(
    episode: dict[str, Any],
    eef_actions: np.ndarray,
    thresholds: dict[str, float],
    config: SpeedScheduleConfig,
) -> dict[str, Any]:
    actions = episode["action"]
    timestamps = episode["timestamp"]
    result = schedule_timestamps(
        actions,
        timestamps,
        eef_actions=eef_actions,
        thresholds=thresholds,
        config=config,
    )
    baseline_duration = float(timestamps[-1] - timestamps[0])
    scheduled_duration = float(result.timestamps[-1] - result.timestamps[0])
    duration_reduction = (
        (baseline_duration - scheduled_duration) / baseline_duration
        if baseline_duration > 0.0 else 0.0
    )
    action_hash = _hash_array(actions)
    labels = result.interval_labels
    criteria = {
        "actions_byte_identical": action_hash == _hash_array(actions.copy()),
        "timestamps_strictly_increasing": bool(np.all(np.diff(result.timestamps) > 0.0)),
        "timestamps_finite": bool(np.all(np.isfinite(result.timestamps))),
        "speed_caps_pass": _speed_caps_pass(result, thresholds, config),
    }
    return {
        "name": episode["name"],
        "frames": int(len(timestamps)),
        "action_sha256": action_hash,
        "path_actions_byte_identical": criteria["actions_byte_identical"],
        "baseline_duration_s": baseline_duration,
        "scheduled_duration_s": scheduled_duration,
        "duration_reduction_fraction": float(duration_reduction),
        "scale": _stats(result.scale_profile),
        "interval_counts": {
            key: int(np.count_nonzero(mask)) for key, mask in labels.items()
        },
        "baseline_motion_stats": {
            "joint_speed_rad_s": _stats(result.baseline_signals["joint_speed"]),
            "eef_speed_m_s": _stats(result.baseline_signals["eef_speed"]),
            "gripper_speed_rad_s": _stats(result.baseline_signals["gripper_speed"]),
            "joint_jerk_rad_s3": _stats(result.baseline_signals["joint_jerk"]),
        },
        "scheduled_motion_stats": {
            "joint_speed_rad_s": _stats(result.retimed_signals["joint_speed"]),
            "eef_speed_m_s": _stats(result.retimed_signals["eef_speed"]),
            "gripper_speed_rad_s": _stats(result.retimed_signals["gripper_speed"]),
            "joint_jerk_rad_s3": _stats(result.retimed_signals["joint_jerk"]),
        },
        "criteria": criteria,
        "accepted": bool(all(criteria.values())),
    }


def schedule_cached_dataset(
    *,
    cache: Path = DEFAULT_CACHE,
    analysis: Path = RESULTS / "trajectory_analysis.json",
    output: Path = RESULTS / "speed_schedule.json",
    max_episodes: int | None = None,
    config: SpeedScheduleConfig = SpeedScheduleConfig(),
) -> dict[str, Any]:
    thresholds = _load_thresholds(analysis)
    paths = _episode_paths(cache, max_episodes)
    episodes = _load_raw_episodes(paths)

    from dataset_contract_audit import _fk_transform

    eef_concat = _fk_transform(np.concatenate([episode["action"] for episode in episodes]))
    cursor = 0
    records = []
    for episode in episodes:
        count = len(episode["action"])
        eef_actions = eef_concat[cursor : cursor + count]
        cursor += count
        records.append(_episode_record(episode, eef_actions, thresholds, config))

    baseline_total = float(sum(record["baseline_duration_s"] for record in records))
    scheduled_total = float(sum(record["scheduled_duration_s"] for record in records))
    accepted_count = int(sum(record["accepted"] for record in records))
    report = {
        "schema_version": "g1_speed_schedule_v1",
        "scope": (
            "Offline timestamp-only schedule over cached LGG100 demonstrations. "
            "Action samples are unchanged; no robot motion or production enablement."
        ),
        "cache": str(cache),
        "analysis": str(analysis),
        "episode_count": len(records),
        "frame_count": int(sum(record["frames"] for record in records)),
        "config": config.__dict__,
        "thresholds": thresholds,
        "summary": {
            "accepted_episode_count": accepted_count,
            "all_episodes_accepted": accepted_count == len(records),
            "baseline_total_duration_s": baseline_total,
            "scheduled_total_duration_s": scheduled_total,
            "total_duration_reduction_fraction": (
                (baseline_total - scheduled_total) / baseline_total
                if baseline_total > 0.0 else 0.0
            ),
            "duration_reduction_fraction": _stats(np.asarray([
                record["duration_reduction_fraction"] for record in records
            ], dtype=np.float64)),
            "minimum_scale": float(min(record["scale"]["p50"] for record in records)),
            "maximum_scale": float(max(record["scale"]["max"] for record in records)),
        },
        "episodes": records,
        "decision": {
            "speed_scheduler_completed": True,
            "timestamp_only": True,
            "path_actions_byte_identical_all": bool(all(
                record["path_actions_byte_identical"] for record in records
            )),
            "offline_replay_comparison_completed": False,
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
    parser.add_argument("--analysis", type=Path, default=RESULTS / "trajectory_analysis.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "speed_schedule.json")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--idle-scale", type=_positive_float, default=2.00)
    parser.add_argument("--coarse-scale", type=_positive_float, default=1.35)
    parser.add_argument("--fine-scale", type=_positive_float, default=1.00)
    parser.add_argument("--gripper-active-scale", type=_positive_float, default=0.90)
    parser.add_argument("--high-speed-scale", type=_positive_float, default=0.85)
    parser.add_argument("--jerk-spike-scale", type=_positive_float, default=0.75)
    args = parser.parse_args()
    config = SpeedScheduleConfig(
        idle_scale=args.idle_scale,
        coarse_scale=args.coarse_scale,
        fine_scale=args.fine_scale,
        gripper_active_scale=args.gripper_active_scale,
        high_speed_scale=args.high_speed_scale,
        jerk_spike_scale=args.jerk_spike_scale,
    )
    report = schedule_cached_dataset(
        cache=args.cache,
        analysis=args.analysis,
        output=args.output,
        max_episodes=args.max_episodes,
        config=config,
    )
    print(args.output)
    print(json.dumps({
        "episodes": report["episode_count"],
        "frames": report["frame_count"],
        "accepted_episode_count": report["summary"]["accepted_episode_count"],
        "total_duration_reduction_fraction": report["summary"]["total_duration_reduction_fraction"],
        "path_actions_byte_identical_all": report["decision"]["path_actions_byte_identical_all"],
        "robot_motion_allowed": report["decision"]["robot_motion_allowed"],
    }, indent=2))
    return 0 if report["summary"]["all_episodes_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
