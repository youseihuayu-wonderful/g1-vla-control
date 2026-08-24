#!/usr/bin/env python3
"""Run real LGG100 output-only inference on prepared public episode samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from g1_policy_contract import ACTION_HORIZON
from lgg100_candidate_server import (
    AUTHOR_DISCRETE_STATE_INPUT,
    AUTHOR_MODEL_CONFIG_NAME,
    DEFAULT_PROMPT,
    HF_REVISION,
    OPENPI_AUDITED_COMMIT,
    build_policy,
)
from neural_action_audit import audit_neural_action_chunk

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "lgg100_semantic_outputs_quarantined.npz",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "results" / "lgg100_semantic_inference.json",
    )
    args = parser.parse_args()

    with np.load(args.samples, allow_pickle=False) as payload:
        cameras = {
            "observation/cam_left_high": np.asarray(payload["cam_left_high"]),
            "observation/cam_left_wrist": np.asarray(payload["cam_left_wrist"]),
            "observation/cam_right_wrist": np.asarray(payload["cam_right_wrist"]),
        }
        states = np.asarray(payload["state"])
        episodes = np.asarray(payload["episode"])
        frames = np.asarray(payload["frame"])
        prompt = str(payload["prompt"].item())
        dataset_revision = str(payload["dataset_revision"].item())
    count = len(states)
    if any(len(value) != count for value in cameras.values()):
        raise ValueError("Camera/state sample counts do not match")

    policy = build_policy(
        args.checkpoint_dir.resolve(), ACTION_HORIZON, DEFAULT_PROMPT
    )
    raw_chunks: list[np.ndarray] = []
    official_chunks: list[np.ndarray] = []
    official_available: list[bool] = []
    records: list[dict] = []
    for index in range(count):
        observation = {
            key: value[index] for key, value in cameras.items()
        }
        observation["observation/state"] = states[index]
        observation["prompt"] = prompt
        start = time.monotonic()
        response = policy.infer(observation)
        latency_ms = (time.monotonic() - start) * 1000.0
        actions = np.asarray(response["actions"], dtype=np.float64)
        audit = audit_neural_action_chunk(actions)
        raw_chunks.append(actions)
        available = audit.official_postprocessed_actions is not None
        official_available.append(available)
        official_chunks.append(
            audit.official_postprocessed_actions
            if available else np.full_like(actions, np.nan)
        )
        record = {
            "sample": index,
            "episode": int(episodes[index]),
            "frame": int(frames[index]),
            "latency_ms": latency_ms,
            "shape": list(actions.shape),
            "finite_shape_passed": audit.finite_shape_passed,
            "raw_quaternion_exact_unit_passed": audit.raw_quaternion_exact_unit_passed,
            "official_consumer_postprocess_passed": (
                audit.official_consumer_postprocess_passed
            ),
            "official_consumer_normalization_applied": audit.normalization_applied,
            "raw_max_quaternion_norm_error": audit.raw_max_quaternion_norm_error,
            "maximum_quaternion_component_adjustment": (
                audit.maximum_quaternion_component_adjustment
            ),
            "raw_sha256": audit.raw_sha256,
            "official_postprocessed_sha256": audit.official_postprocessed_sha256,
            "reasons": list(audit.reasons),
            "policy_timing": response.get("policy_timing", {}),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    raw_array = np.stack(raw_chunks)
    official_array = np.stack(official_chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        raw_actions=raw_array,
        official_postprocessed_actions=official_array,
        official_postprocess_available=np.asarray(official_available),
        canonicalized_actions_for_analysis=official_array,
        canonicalized_available=np.asarray(official_available),
        episode=episodes,
        frame=frames,
        checkpoint_revision=np.asarray(HF_REVISION),
        openpi_commit=np.asarray(OPENPI_AUDITED_COMMIT),
        dataset_revision=np.asarray(dataset_revision),
        executable=np.asarray(False),
        quarantined=np.asarray(True),
    )
    latencies = np.asarray([record["latency_ms"] for record in records])
    report = {
        "scope": "Real LGG100 inference on public episode observations; raw output plus pinned official consumer post-processing, no dynamics.",
        "checkpoint_revision": HF_REVISION,
        "openpi_commit": OPENPI_AUDITED_COMMIT,
        "dataset_revision": dataset_revision,
        "strict_parameter_tree_restore": True,
        "author_core_config_directly_confirmed": True,
        "author_model_config_name": AUTHOR_MODEL_CONFIG_NAME,
        "action_horizon": ACTION_HORIZON,
        "discrete_state_input": AUTHOR_DISCRETE_STATE_INPUT,
        "complete_author_train_config_available": False,
        "g1_contract_verified": bool(all(official_available)),
        "g1_sim_eligible": False,
        "g1_execution_enabled": False,
        "execution_performed": False,
        "sample_count": count,
        "summary": {
            "finite_shape_passes": sum(r["finite_shape_passed"] for r in records),
            "raw_quaternion_exact_unit_passes": sum(
                r["raw_quaternion_exact_unit_passed"] for r in records
            ),
            "official_consumer_postprocess_passes": sum(
                r["official_consumer_postprocess_passed"] for r in records
            ),
            "latency_ms": {
                "p50": float(np.quantile(latencies, 0.50)),
                "p95": float(np.quantile(latencies, 0.95)),
                "p99": float(np.quantile(latencies, 0.99)),
                "max": float(np.max(latencies)),
            },
            "unique_raw_hashes": len(set(r["raw_sha256"] for r in records)),
        },
        "records": records,
        "output_npz": str(args.output),
        "output_npz_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(args.report)
    print(json.dumps(report["summary"], indent=2))
    if report["summary"]["finite_shape_passes"] != count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
