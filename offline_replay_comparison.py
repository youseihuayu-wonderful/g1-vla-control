#!/usr/bin/env python3
"""Offline replay comparison for baseline vs scheduled G1 trajectories.

This replays cached demonstrations through a fixed-rate shadow sampler and the
segment-aware timestamp schedule. It proves the scheduler is a time warp only:
indexed action samples and endpoints are unchanged, risky intervals are not
accelerated, and scheduled replay uses fewer fixed-rate control observations in
aggregate. No robot transport or hardware controller is imported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from speed_scheduler import (
    SpeedScheduleConfig,
    _episode_paths,
    _hash_array,
    _load_raw_episodes,
    _load_thresholds,
    schedule_timestamps,
)
from trajectory_analyzer import DEFAULT_CACHE, RESULTS


@dataclass(frozen=True)
class ReplayConfig:
    replay_rate_hz: float = 30.0
    endpoint_tolerance: float = 1e-12
    scale_tolerance: float = 1e-9


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


def fixed_rate_times(timestamps: np.ndarray, replay_rate_hz: float) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("timestamps must contain at least two samples")
    if replay_rate_hz <= 0.0 or not np.isfinite(replay_rate_hz):
        raise ValueError("replay_rate_hz must be positive and finite")
    start = float(timestamps[0])
    end = float(timestamps[-1])
    if end <= start:
        raise ValueError("timestamps must have positive duration")
    step = 1.0 / float(replay_rate_hz)
    query = np.arange(start, end, step, dtype=np.float64)
    if not len(query) or query[-1] < end:
        query = np.concatenate((query, [end]))
    else:
        query[-1] = end
    return query


def sample_actions(
    timestamps: np.ndarray,
    actions: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    query_times = np.asarray(query_times, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] != len(timestamps):
        raise ValueError("actions must be [T,D] and match timestamps")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    sampled = np.empty((len(query_times), actions.shape[1]), dtype=np.float64)
    for column in range(actions.shape[1]):
        sampled[:, column] = np.interp(query_times, timestamps, actions[:, column])
    return sampled


def compare_episode(
    name: str,
    actions: np.ndarray,
    timestamps: np.ndarray,
    *,
    eef_actions: np.ndarray,
    thresholds: dict[str, float],
    schedule_config: SpeedScheduleConfig = SpeedScheduleConfig(),
    replay_config: ReplayConfig = ReplayConfig(),
) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float64)
    original_actions = actions.copy()
    timestamps = np.asarray(timestamps, dtype=np.float64)
    result = schedule_timestamps(
        actions,
        timestamps,
        eef_actions=eef_actions,
        thresholds=thresholds,
        config=schedule_config,
    )

    baseline_query = fixed_rate_times(timestamps, replay_config.replay_rate_hz)
    scheduled_query = fixed_rate_times(result.timestamps, replay_config.replay_rate_hz)
    baseline_replay = sample_actions(timestamps, actions, baseline_query)
    scheduled_replay = sample_actions(result.timestamps, actions, scheduled_query)

    baseline_duration = float(timestamps[-1] - timestamps[0])
    scheduled_duration = float(result.timestamps[-1] - result.timestamps[0])
    endpoint_delta = scheduled_replay[-1] - baseline_replay[-1]
    start_delta = scheduled_replay[0] - baseline_replay[0]

    risk_mask = result.interval_labels["high_speed"] | result.interval_labels["jerk_spike"]
    coarse_or_idle = result.interval_labels["coarse"] | result.interval_labels["idle"]
    risk_scale = result.scale_profile[risk_mask]
    coarse_idle_scale = result.scale_profile[coarse_or_idle]
    risky_intervals_accelerated = bool(
        risk_scale.size
        and float(np.max(risk_scale)) > 1.0 + replay_config.scale_tolerance
    )

    criteria = {
        "schedule_did_not_mutate_actions": bool(np.array_equal(actions, original_actions)),
        "indexed_action_samples_byte_identical": _hash_array(actions) == _hash_array(original_actions),
        "scheduled_timestamps_strictly_increasing": bool(np.all(np.diff(result.timestamps) > 0.0)),
        "scheduled_timestamps_finite": bool(np.all(np.isfinite(result.timestamps))),
        "fixed_rate_replay_finite": bool(
            np.all(np.isfinite(baseline_replay)) and np.all(np.isfinite(scheduled_replay))
        ),
        "start_endpoint_preserved": bool(
            np.max(np.abs(start_delta)) <= replay_config.endpoint_tolerance
            and np.max(np.abs(endpoint_delta)) <= replay_config.endpoint_tolerance
        ),
        "risky_intervals_not_accelerated": not risky_intervals_accelerated,
    }

    return {
        "name": name,
        "frames": int(len(timestamps)),
        "action_sha256": _hash_array(actions),
        "baseline_duration_s": baseline_duration,
        "scheduled_duration_s": scheduled_duration,
        "duration_reduction_fraction": (
            (baseline_duration - scheduled_duration) / baseline_duration
            if baseline_duration > 0.0 else 0.0
        ),
        "baseline_replay_frames": int(len(baseline_query)),
        "scheduled_replay_frames": int(len(scheduled_query)),
        "replay_frame_reduction_fraction": (
            (len(baseline_query) - len(scheduled_query)) / len(baseline_query)
            if len(baseline_query) else 0.0
        ),
        "start_max_abs_delta": float(np.max(np.abs(start_delta))),
        "endpoint_max_abs_delta": float(np.max(np.abs(endpoint_delta))),
        "risk_interval_count": int(np.count_nonzero(risk_mask)),
        "risk_scale_max": float(np.max(risk_scale)) if risk_scale.size else 0.0,
        "coarse_or_idle_interval_count": int(np.count_nonzero(coarse_or_idle)),
        "coarse_or_idle_scale_mean": (
            float(np.mean(coarse_idle_scale)) if coarse_idle_scale.size else 0.0
        ),
        "scale": _stats(result.scale_profile),
        "criteria": criteria,
        "accepted": bool(all(criteria.values())),
    }


def compare_cached_dataset(
    *,
    cache: Path = DEFAULT_CACHE,
    analysis: Path = RESULTS / "trajectory_analysis.json",
    speed_schedule: Path = RESULTS / "speed_schedule.json",
    output: Path = RESULTS / "offline_replay_comparison.json",
    max_episodes: int | None = None,
    schedule_config: SpeedScheduleConfig = SpeedScheduleConfig(),
    replay_config: ReplayConfig = ReplayConfig(),
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
        records.append(compare_episode(
            episode["name"],
            episode["action"],
            episode["timestamp"],
            eef_actions=eef_actions,
            thresholds=thresholds,
            schedule_config=schedule_config,
            replay_config=replay_config,
        ))

    accepted_count = int(sum(record["accepted"] for record in records))
    baseline_duration = float(sum(record["baseline_duration_s"] for record in records))
    scheduled_duration = float(sum(record["scheduled_duration_s"] for record in records))
    baseline_frames = int(sum(record["baseline_replay_frames"] for record in records))
    scheduled_frames = int(sum(record["scheduled_replay_frames"] for record in records))
    max_endpoint_delta = float(max(record["endpoint_max_abs_delta"] for record in records))
    max_start_delta = float(max(record["start_max_abs_delta"] for record in records))
    risk_accelerated_count = int(sum(
        not record["criteria"]["risky_intervals_not_accelerated"] for record in records
    ))

    schedule_payload = json.loads(speed_schedule.read_text()) if speed_schedule.exists() else {}
    report = {
        "schema_version": "g1_offline_replay_comparison_v1",
        "scope": (
            "Fixed-rate offline replay comparison of baseline timestamps and "
            "segment-aware scheduled timestamps over cached demonstrations. No robot motion."
        ),
        "cache": str(cache),
        "analysis": str(analysis),
        "speed_schedule": str(speed_schedule),
        "episode_count": len(records),
        "frame_count": int(sum(record["frames"] for record in records)),
        "replay_rate_hz": replay_config.replay_rate_hz,
        "schedule_summary_source": {
            "schema_version": schedule_payload.get("schema_version"),
            "total_duration_reduction_fraction": schedule_payload.get("summary", {}).get(
                "total_duration_reduction_fraction"
            ),
            "path_actions_byte_identical_all": schedule_payload.get("decision", {}).get(
                "path_actions_byte_identical_all"
            ),
        },
        "summary": {
            "accepted_episode_count": accepted_count,
            "all_episodes_accepted": accepted_count == len(records),
            "baseline_total_duration_s": baseline_duration,
            "scheduled_total_duration_s": scheduled_duration,
            "total_duration_reduction_fraction": (
                (baseline_duration - scheduled_duration) / baseline_duration
                if baseline_duration > 0.0 else 0.0
            ),
            "baseline_fixed_rate_replay_frames": baseline_frames,
            "scheduled_fixed_rate_replay_frames": scheduled_frames,
            "fixed_rate_replay_frame_reduction_fraction": (
                (baseline_frames - scheduled_frames) / baseline_frames
                if baseline_frames else 0.0
            ),
            "duration_reduction_fraction": _stats(np.asarray([
                record["duration_reduction_fraction"] for record in records
            ], dtype=np.float64)),
            "replay_frame_reduction_fraction": _stats(np.asarray([
                record["replay_frame_reduction_fraction"] for record in records
            ], dtype=np.float64)),
            "max_start_abs_delta": max_start_delta,
            "max_endpoint_abs_delta": max_endpoint_delta,
            "risk_accelerated_episode_count": risk_accelerated_count,
        },
        "episodes": records,
        "decision": {
            "offline_replay_comparison_completed": True,
            "timestamp_only": True,
            "path_actions_byte_identical_all": bool(all(
                record["criteria"]["indexed_action_samples_byte_identical"] for record in records
            )),
            "endpoints_preserved_all": bool(max_endpoint_delta <= replay_config.endpoint_tolerance),
            "risky_intervals_not_accelerated_all": risk_accelerated_count == 0,
            "hardware_execution_performed": False,
            "robot_motion_allowed": False,
        },
    }
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0 or not np.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--analysis", type=Path, default=RESULTS / "trajectory_analysis.json")
    parser.add_argument("--speed-schedule", type=Path, default=RESULTS / "speed_schedule.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "offline_replay_comparison.json")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--replay-rate-hz", type=_positive_float, default=30.0)
    args = parser.parse_args()
    report = compare_cached_dataset(
        cache=args.cache,
        analysis=args.analysis,
        speed_schedule=args.speed_schedule,
        output=args.output,
        max_episodes=args.max_episodes,
        replay_config=ReplayConfig(replay_rate_hz=args.replay_rate_hz),
    )
    print(args.output)
    print(json.dumps({
        "episodes": report["episode_count"],
        "accepted_episode_count": report["summary"]["accepted_episode_count"],
        "total_duration_reduction_fraction": report["summary"]["total_duration_reduction_fraction"],
        "fixed_rate_replay_frame_reduction_fraction": report["summary"]["fixed_rate_replay_frame_reduction_fraction"],
        "max_endpoint_abs_delta": report["summary"]["max_endpoint_abs_delta"],
        "risk_accelerated_episode_count": report["summary"]["risk_accelerated_episode_count"],
        "robot_motion_allowed": report["decision"]["robot_motion_allowed"],
    }, indent=2))
    return 0 if report["summary"]["all_episodes_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
