#!/usr/bin/env python3
"""Bounded multi-scenario endurance probe for a frozen LGG100 policy server.

This tool is output-only: it never retimes, executes MuJoCo dynamics, or sends a
hardware command. It records raw neural output and Yuhao's documented bounded
quaternion-only consumer post-processing without claiming simulation eligibility.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np

from g1_policy_contract import (
    ACTION_HORIZON,
    CONTRACT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
)
from lgg100_candidate_server import HF_REVISION
from local_websocket_policy_client import LocalWebsocketPolicyClient
from neural_action_audit import audit_neural_action_chunk


_STOP_REQUESTED = False


def _request_stop(_signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_observation(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        binding = {
            "g1_policy_contract_id": str(payload["g1_policy_contract_id"].item()),
            "g1_policy_contract_version": str(payload["g1_policy_contract_version"].item()),
            "g1_policy_contract_sha256": str(payload["g1_policy_contract_sha256"].item()),
            "action_horizon": int(payload["action_horizon"].item()),
        }
        expected = {
            "g1_policy_contract_id": CONTRACT_ID,
            "g1_policy_contract_version": CONTRACT_VERSION,
            "g1_policy_contract_sha256": CONTRACT_SHA256,
            "action_horizon": ACTION_HORIZON,
        }
        if binding != expected:
            raise ValueError(f"observation contract mismatch: {binding}")
        return {
            "observation/cam_left_high": np.asarray(payload["cam_left_high"]),
            "observation/cam_left_wrist": np.asarray(payload["cam_left_wrist"]),
            "observation/cam_right_wrist": np.asarray(payload["cam_right_wrist"]),
            "observation/state": np.asarray(payload["state"]),
            "prompt": str(payload["prompt"].item()),
        }


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--scenario", nargs=2, action="append", metavar=("NAME", "NPZ"), required=True)
    parser.add_argument("--duration-s", type=float, default=14_400.0)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--heartbeat-s", type=float, default=60.0)
    parser.add_argument("--sample-every", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.duration_s < 60:
        raise SystemExit("duration must be at least 60 seconds")
    if not 0 < args.rate_hz <= 10:
        raise SystemExit("rate must be in (0, 10] Hz")
    if args.heartbeat_s <= 0 or args.sample_every < 1:
        raise SystemExit("heartbeat and sample cadence must be positive")

    names = [name for name, _ in args.scenario]
    if len(names) != len(set(names)):
        raise SystemExit("scenario names must be unique")
    scenarios = [
        {
            "name": name,
            "path": Path(path).resolve(),
            "observation": _load_observation(Path(path).resolve()),
        }
        for name, path in args.scenario
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "soak_status.json"
    records_path = args.output_dir / "inference_records.jsonl.gz"
    samples_path = args.output_dir / "sampled_actions_quarantined.npz"
    summary_path = args.output_dir / "soak_summary.json"

    client = LocalWebsocketPolicyClient(args.host, args.port)
    metadata = client.get_server_metadata()
    if not (
        metadata.get("strict_parameter_tree_restore") is True
        and metadata.get("hf_revision") == HF_REVISION
        and metadata.get("g1_policy_contract_id") == CONTRACT_ID
        and metadata.get("g1_policy_contract_sha256") == CONTRACT_SHA256
        and metadata.get("action_horizon_author_confirmed") is True
        and metadata.get("safe_for_g1_hardware") is False
    ):
        client.close()
        raise ValueError("policy server metadata is not a strict quarantined restore")

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    started_wall = dt.datetime.now(dt.timezone.utc)
    started = time.monotonic()
    deadline = started + args.duration_s
    next_call = started
    next_heartbeat = started
    total = finite = raw_exact_unit = official_postprocessed = missed_deadlines = 0
    latencies: dict[str, list[float]] = {scenario["name"]: [] for scenario in scenarios}
    qerrors: list[float] = []
    unique_hashes: set[str] = set()
    sample_raw_actions: list[np.ndarray] = []
    sample_official_actions: list[np.ndarray] = []
    sample_scenarios: list[str] = []
    sample_iterations: list[int] = []
    failure_reason: str | None = None

    try:
        with gzip.open(records_path, "wt", encoding="utf-8") as records:
            while time.monotonic() < deadline and not _STOP_REQUESTED:
                scenario = scenarios[total % len(scenarios)]
                call_start = time.monotonic()
                response = client.infer(scenario["observation"])
                latency_ms = (time.monotonic() - call_start) * 1000.0
                actions = np.asarray(response["actions"], dtype=np.float64)
                audit = audit_neural_action_chunk(actions)
                total += 1
                finite += int(audit.finite_shape_passed)
                raw_exact_unit += int(audit.raw_quaternion_exact_unit_passed)
                official_postprocessed += int(audit.official_consumer_postprocess_passed)
                latencies[scenario["name"]].append(latency_ms)
                if audit.raw_max_quaternion_norm_error is not None:
                    qerrors.append(audit.raw_max_quaternion_norm_error)
                if audit.raw_sha256 is not None:
                    unique_hashes.add(audit.raw_sha256)
                record = {
                    "iteration": total,
                    "scenario": scenario["name"],
                    "latency_ms": latency_ms,
                    "finite_shape_passed": audit.finite_shape_passed,
                    "raw_quaternion_exact_unit_passed": audit.raw_quaternion_exact_unit_passed,
                    "official_consumer_postprocess_passed": audit.official_consumer_postprocess_passed,
                    "official_consumer_normalization_applied": audit.normalization_applied,
                    "official_postprocessed_sha256": audit.official_postprocessed_sha256,
                    "raw_max_quaternion_norm_error": audit.raw_max_quaternion_norm_error,
                    "raw_sha256": audit.raw_sha256,
                    "reasons": list(audit.reasons),
                    "execution_performed": False,
                }
                records.write(json.dumps(record, separators=(",", ":")) + "\n")
                if not audit.official_consumer_postprocess_passed:
                    failure_reason = "official_consumer_postprocessing_rejected"
                    break
                if total == 1 or total % args.sample_every == 0 or total <= len(scenarios):
                    sample_raw_actions.append(actions.copy())
                    sample_official_actions.append(audit.official_postprocessed_actions.copy())
                    sample_scenarios.append(scenario["name"])
                    sample_iterations.append(total)
                if not audit.finite_shape_passed:
                    failure_reason = "non_finite_or_shape_invalid_output"
                    break

                next_call += 1.0 / args.rate_hz
                sleep_s = next_call - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    missed_deadlines += 1
                    next_call = time.monotonic()

                now = time.monotonic()
                if now >= next_heartbeat:
                    _atomic_json(status_path, {
                        "schema_version": "lgg100_output_only_soak_status_v2",
                        "state": "RUNNING",
                        "elapsed_s": now - started,
                        "target_duration_s": args.duration_s,
                        "completed_calls": total,
                        "finite_shape_passes": finite,
                        "raw_quaternion_exact_unit_passes": raw_exact_unit,
                        "official_consumer_postprocess_passes": official_postprocessed,
                        "missed_rate_deadlines": missed_deadlines,
                        "neural_training": False,
                        "mujoco_dynamics_executed": False,
                        "hardware_execution_performed": False,
                    })
                    next_heartbeat = now + args.heartbeat_s
    except Exception as exc:  # retain a bounded, explicit failure artifact
        failure_reason = f"inference_exception:{type(exc).__name__}"
        raise
    finally:
        client.close()
        elapsed = time.monotonic() - started
        completed_duration = elapsed >= args.duration_s and not _STOP_REQUESTED and failure_reason is None
        if sample_raw_actions:
            np.savez_compressed(
                samples_path,
                raw_actions=np.stack(sample_raw_actions),
                official_postprocessed_actions=np.stack(sample_official_actions),
                scenarios=np.asarray(sample_scenarios),
                iterations=np.asarray(sample_iterations, dtype=np.int64),
                checkpoint_revision=np.asarray(HF_REVISION),
                g1_policy_contract_sha256=np.asarray(CONTRACT_SHA256),
                executable=np.asarray(False),
                quarantined=np.asarray(True),
            )
        all_latency = [value for group in latencies.values() for value in group]
        summary = {
            "schema_version": "lgg100_output_only_soak_summary_v2",
            "scope": "Frozen LGG100 multi-scenario output-only endurance inference.",
            "started_at_utc": started_wall.isoformat(),
            "target_duration_s": args.duration_s,
            "actual_duration_s": elapsed,
            "target_rate_hz": args.rate_hz,
            "completed_target_duration": completed_duration,
            "stop_requested": _STOP_REQUESTED,
            "failure_reason": failure_reason,
            "scenario_names": names,
            "completed_calls": total,
            "finite_shape_passes": finite,
            "raw_quaternion_exact_unit_passes": raw_exact_unit,
            "official_consumer_postprocess_passes": official_postprocessed,
            "official_consumer_postprocess_qualification_passed": bool(
                total > 0 and official_postprocessed == total
            ),
            "unique_raw_chunk_hashes": len(unique_hashes),
            "missed_rate_deadlines": missed_deadlines,
            "latency_ms": {
                "overall": _quantiles(all_latency),
                "by_scenario": {name: _quantiles(values) for name, values in latencies.items()},
            },
            "raw_quaternion_norm_error": _quantiles(qerrors),
            "records_sha256": _sha256(records_path) if records_path.exists() else None,
            "samples_sha256": _sha256(samples_path) if samples_path.exists() else None,
            "neural_training": False,
            "weights_modified": False,
            "mujoco_dynamics_executed": False,
            "g1_contract_verified": bool(
                total > 0 and official_postprocessed == total
            ),
            "g1_sim_eligible": False,
            "g1_execution_enabled": False,
            "hardware_execution_performed": False,
        }
        _atomic_json(summary_path, summary)
        _atomic_json(status_path, {
            "schema_version": "lgg100_output_only_soak_status_v2",
            "state": "COMPLETE" if completed_duration else "FAILED",
            "elapsed_s": elapsed,
            "completed_calls": total,
            "failure_reason": failure_reason or (None if completed_duration else "duration_not_completed"),
            "summary_sha256": _sha256(summary_path),
            "neural_training": False,
            "mujoco_dynamics_executed": False,
            "hardware_execution_performed": False,
        })

    if not completed_duration:
        raise SystemExit("soak did not complete its target duration")
    print(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
