#!/usr/bin/env python3
"""Run repeated quarantined LGG100 draws for one saved observation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from lgg100_candidate_server import DEFAULT_PROMPT, HF_REVISION, build_policy
from local_websocket_policy_client import LocalWebsocketPolicyClient
from neural_action_audit import audit_neural_action_chunk

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.draws < 1:
        raise SystemExit("draws must be positive")
    with np.load(args.observation, allow_pickle=False) as payload:
        observation = {
            "observation/cam_left_high": np.asarray(payload["cam_left_high"]),
            "observation/cam_left_wrist": np.asarray(payload["cam_left_wrist"]),
            "observation/cam_right_wrist": np.asarray(payload["cam_right_wrist"]),
            "observation/state": np.asarray(payload["state"]),
            "prompt": str(payload["prompt"].item()),
        }
    if (args.checkpoint_dir is None) == (args.host is None):
        raise SystemExit("specify exactly one of --checkpoint-dir or --host")
    if args.host is None:
        policy = build_policy(
            args.checkpoint_dir.resolve(), 50, DEFAULT_PROMPT
        )
        policy_source = "in_process_strict_restore"
    else:
        policy = LocalWebsocketPolicyClient(args.host, args.port)
        metadata = policy.get_server_metadata()
        if not (
            metadata.get("strict_parameter_tree_restore") is True
            and metadata.get("hf_revision") == HF_REVISION
            and metadata.get("safe_for_g1_hardware") is False
        ):
            raise ValueError("policy server metadata is not a strict quarantined restore")
        policy_source = f"loopback_websocket:{args.host}:{args.port}"
    raw_chunks = []
    analysis_chunks = []
    records = []
    for draw in range(args.draws):
        start = time.monotonic()
        response = policy.infer(observation)
        latency = (time.monotonic() - start) * 1000.0
        actions = np.asarray(response["actions"], dtype=np.float64)
        audit = audit_neural_action_chunk(actions)
        raw_chunks.append(actions)
        analysis_chunks.append(
            audit.canonicalized_actions_for_analysis
            if audit.canonicalized_actions_for_analysis is not None
            else np.full_like(actions, np.nan)
        )
        closing = np.any(
            actions[:, 14:16]
            < observation["observation/state"][None, 14:16] - 0.15,
            axis=1,
        )
        records.append({
            "draw": draw,
            "latency_ms": latency,
            "finite_shape_passed": audit.finite_shape_passed,
            "raw_contract_passed": audit.raw_contract_passed,
            "bounded_analysis_available": (
                audit.canonicalized_actions_for_analysis is not None
            ),
            "raw_max_quaternion_norm_error": (
                audit.raw_max_quaternion_norm_error
            ),
            "raw_sha256": audit.raw_sha256,
            "gripper_first": actions[0, 14:16].tolist(),
            "gripper_min": actions[:, 14:16].min(axis=0).tolist(),
            "gripper_max": actions[:, 14:16].max(axis=0).tolist(),
            "closing_indices": np.flatnonzero(closing).tolist(),
        })
        print(json.dumps(records[-1]), flush=True)
    if isinstance(policy, LocalWebsocketPolicyClient):
        policy.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        actions=np.stack(raw_chunks),
        canonicalized_actions_for_analysis=np.stack(analysis_chunks),
        observation_state=observation["observation/state"],
        checkpoint_revision=np.asarray(HF_REVISION),
        executable=np.asarray(False),
        quarantined=np.asarray(True),
    )
    latency = np.asarray([record["latency_ms"] for record in records])
    report = {
        "scope": "Repeated LGG100 output-only observation probe; no execution.",
        "observation": str(args.observation),
        "observation_sha256": hashlib.sha256(
            args.observation.read_bytes()
        ).hexdigest(),
        "checkpoint_revision": HF_REVISION,
        "policy_source": policy_source,
        "draws": args.draws,
        "records": records,
        "summary": {
            "finite_shape_passes": sum(
                record["finite_shape_passed"] for record in records
            ),
            "bounded_analysis_available": sum(
                record["bounded_analysis_available"] for record in records
            ),
            "draws_with_grasp_phase": sum(
                bool(record["closing_indices"]) for record in records
            ),
            "latency_ms": {
                "p50": float(np.quantile(latency, 0.50)),
                "p95": float(np.quantile(latency, 0.95)),
                "max": float(np.max(latency)),
            },
        },
        "g1_contract_verified": False,
        "g1_sim_eligible": False,
        "g1_execution_enabled": False,
        "execution_performed": False,
    }
    report["output_npz"] = str(args.output)
    report["output_npz_sha256"] = hashlib.sha256(
        args.output.read_bytes()
    ).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(args.report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
