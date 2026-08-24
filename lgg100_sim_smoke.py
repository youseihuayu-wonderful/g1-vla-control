#!/usr/bin/env python3
"""Output-only LGG100 audit using the frozen G1 EDU observation contract.

A real neural restore may pass while G1 simulation eligibility remains false.
No action is executed here; shape alone never establishes G1 compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import socket
import time
from typing import Any

import numpy as np

from g1_mujoco_bridge import build_sim_observation
from g1_policy_contract import (
    ACTION_HORIZON,
    CONTRACT_ID,
    CONTRACT_SHA256,
    POLICY_RATE_HZ,
    contract_metadata,
)
from local_websocket_policy_client import LocalWebsocketPolicyClient
from neural_action_audit import audit_neural_action_chunk

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
HF_REPO = "LGG100/stack-cube-eef-24k"
HF_REVISION = "cced7a7ff7b454fdcac555457a1a2a3dc262ac77"
EXPECTED_ACTION_DIM = 16


def _infer_with_timeout(client, observation: dict[str, Any], timeout_ms: float):
    def handler(signum, frame):
        del signum, frame
        raise TimeoutError(f"LGG100 inference exceeded {timeout_ms:.0f} ms")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_ms / 1000.0)
    try:
        return client.infer(observation)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--warmup-calls", type=int, default=1)
    parser.add_argument("--call-timeout-ms", type=float, default=60_000.0)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--action-rate-hz", type=float, default=POLICY_RATE_HZ)
    parser.add_argument("--output", type=Path, default=RESULTS / "lgg100_vla_smoke_real.json")
    parser.add_argument("--chunk-output", type=Path, default=RESULTS / "lgg100_action_chunk_real.npz")
    args = parser.parse_args()
    if args.calls < 1 or args.warmup_calls < 0:
        raise SystemExit("calls must be positive and warmup-calls non-negative")
    if not np.isclose(args.action_rate_hz, POLICY_RATE_HZ):
        raise SystemExit(f"G1 contract requires action rate {POLICY_RATE_HZ:g} Hz")

    with socket.create_connection((args.host, args.port), timeout=args.connect_timeout_s):
        pass
    client = LocalWebsocketPolicyClient(host=args.host, port=args.port)
    metadata = _json_safe(client.get_server_metadata())
    strict_neural_restore = bool(
        metadata.get("neural_checkpoint_loaded") is True
        and metadata.get("strict_parameter_tree_restore") is True
        and metadata.get("hf_repo") == HF_REPO
        and metadata.get("hf_revision") == HF_REVISION
    )
    observation, observation_evidence = build_sim_observation()

    warmup_errors = []
    for _ in range(args.warmup_calls):
        try:
            _infer_with_timeout(client, observation, args.call_timeout_ms)
        except Exception as exc:
            warmup_errors.append(f"{type(exc).__name__}: {exc}")
            break

    records: list[dict[str, Any]] = []
    chunks: list[np.ndarray] = []
    shapes: set[tuple[int, ...]] = set()
    latencies: list[float] = []
    for index in range(args.calls):
        start = time.monotonic_ns()
        try:
            response = _infer_with_timeout(client, observation, args.call_timeout_ms)
            latency_ms = (time.monotonic_ns() - start) / 1e6
            actions = np.asarray(response.get("actions"), dtype=np.float64)
            finite = bool(actions.size and np.all(np.isfinite(actions)))
            shape_ok = actions.ndim == 2 and actions.shape[1] == EXPECTED_ACTION_DIM
            shape = tuple(int(x) for x in actions.shape)
            shapes.add(shape)
            latencies.append(latency_ms)
            contract_valid = False
            contract_error = None
            raw_quaternion_exact_unit = False
            normalization_applied = False
            maximum_quaternion_adjustment = None
            official_postprocessed_sha256 = None
            if finite and shape_ok:
                chunks.append(actions.copy())
                audit = audit_neural_action_chunk(
                    actions, expected_horizon=ACTION_HORIZON
                )
                raw_quaternion_exact_unit = audit.raw_quaternion_exact_unit_passed
                contract_valid = audit.official_consumer_postprocess_passed
                normalization_applied = audit.normalization_applied
                maximum_quaternion_adjustment = (
                    audit.maximum_quaternion_component_adjustment
                )
                official_postprocessed_sha256 = audit.official_postprocessed_sha256
                if not contract_valid:
                    contract_error = "; ".join(audit.reasons)
            left_norm = np.linalg.norm(actions[:, 3:7], axis=1) if shape_ok else np.array([])
            right_norm = np.linalg.norm(actions[:, 10:14], axis=1) if shape_ok else np.array([])
            records.append({
                "call": index,
                "latency_ms": latency_ms,
                "action_shape": list(shape),
                "finite": finite,
                "valid_16d_chunk": shape_ok,
                "g1_action_contract_valid": contract_valid,
                "g1_action_contract_error": contract_error,
                "raw_quaternion_exact_unit_passed": raw_quaternion_exact_unit,
                "official_consumer_postprocess_passed": contract_valid,
                "official_consumer_normalization_applied": normalization_applied,
                "maximum_quaternion_component_adjustment": maximum_quaternion_adjustment,
                "official_postprocessed_action_sha256": official_postprocessed_sha256,
                "action_min": float(actions.min()) if actions.size else None,
                "action_max": float(actions.max()) if actions.size else None,
                "left_quaternion_norm_range": [float(left_norm.min()), float(left_norm.max())] if left_norm.size else None,
                "right_quaternion_norm_range": [float(right_norm.min()), float(right_norm.max())] if right_norm.size else None,
                "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest() if actions.size else None,
                "policy_timing": _json_safe(response.get("policy_timing", {})),
                "error": None,
            })
        except Exception as exc:
            latency_ms = (time.monotonic_ns() - start) / 1e6
            latencies.append(latency_ms)
            records.append({
                "call": index,
                "latency_ms": latency_ms,
                "action_shape": [],
                "finite": False,
                "valid_16d_chunk": False,
                "g1_action_contract_valid": False,
                "g1_action_contract_error": None,
                "error": f"{type(exc).__name__}: {exc}",
            })

    valid_calls = sum(
        bool(record["finite"] and record["valid_16d_chunk"] and record["error"] is None)
        for record in records
    )
    output_contract_valid_calls = sum(
        bool(record.get("g1_action_contract_valid")) for record in records
    )
    raw_quaternion_exact_unit_calls = sum(
        bool(record.get("raw_quaternion_exact_unit_passed"))
        for record in records
    )
    neural_output_passed = bool(
        strict_neural_restore
        and not warmup_errors
        and valid_calls == args.calls
        and len(shapes) == 1
    )
    structural_output_passed = bool(
        neural_output_passed and output_contract_valid_calls == args.calls
    )
    expected_contract = contract_metadata(verified=True)
    g1_contract_verified = all(
        metadata.get(key) == value for key, value in expected_contract.items()
    )
    ready_for_sequential_preflight = bool(
        structural_output_passed and g1_contract_verified
    )
    g1_sim_eligible = False
    latency_array = np.asarray(latencies, dtype=np.float64)
    latency_summary = {
        "p50": float(np.quantile(latency_array, 0.50)),
        "p95": float(np.quantile(latency_array, 0.95)),
        "p99": float(np.quantile(latency_array, 0.99)),
        "max": float(latency_array.max()),
    } if len(latency_array) else {}
    report = {
        "scope": "Real LGG100 neural output-only audit using the frozen G1 observation contract.",
        "evidence_mode": "lgg100_real_weights_candidate_semantics",
        "neural_vla_claimed": strict_neural_restore,
        "author_config_claimed": False,
        "g1_policy_contract_id": CONTRACT_ID,
        "g1_policy_contract_sha256": CONTRACT_SHA256,
        "g1_contract_verified": g1_contract_verified,
        "g1_sim_eligible": g1_sim_eligible,
        "ready_for_sequential_preflight": ready_for_sequential_preflight,
        "g1_execution_enabled": False,
        "adaptive_retimer_enabled": False,
        "checkpoint": {"repo": HF_REPO, "revision": HF_REVISION},
        "server_metadata": metadata,
        "observation": observation_evidence,
        "summary": {
            "passed": structural_output_passed,
            "neural_output_passed": neural_output_passed,
            "structural_output_passed": structural_output_passed,
            "g1_sim_eligible": g1_sim_eligible,
            "ready_for_sequential_preflight": ready_for_sequential_preflight,
            "calls": args.calls,
            "valid_calls": valid_calls,
            "output_contract_valid_calls": output_contract_valid_calls,
            "raw_quaternion_exact_unit_calls": raw_quaternion_exact_unit_calls,
            "official_consumer_postprocess_calls": output_contract_valid_calls,
            "warmup_errors": warmup_errors,
            "unique_action_shapes": [list(shape) for shape in sorted(shapes)],
            "latency_ms": latency_summary,
        },
        "calls": records,
        "verdict": (
            "Neural output passed, but the documented official consumer boundary rejected the action. Do not execute it."
            if neural_output_passed and not structural_output_passed else
            "Neural and structural output passed, but semantics are not verified against the frozen G1 EDU contract. Keep output-only; simulation and hardware remain blocked."
            if structural_output_passed and not ready_for_sequential_preflight else
            "Neural output and the pinned G1 action contract passed; the artifact may proceed to sequential preflight, never directly to hardware."
            if ready_for_sequential_preflight else
            "Restore/inference gate failed. Do not execute the chunk in simulation or hardware."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
    print(json.dumps(report["summary"], indent=2))

    if neural_output_passed and chunks:
        selected = chunks[0]
        selected_audit = audit_neural_action_chunk(
            selected, expected_horizon=ACTION_HORIZON
        )
        timestamps = np.arange(len(selected), dtype=np.float64) / args.action_rate_hz
        payload = {
            "actions": selected,
            "timestamps": timestamps,
            "observation_state": np.asarray(observation["observation/state"]),
            "action_sha256": np.asarray(hashlib.sha256(selected.tobytes()).hexdigest()),
            "hf_revision": np.asarray(HF_REVISION),
            "g1_policy_contract_id": np.asarray(CONTRACT_ID),
            "g1_policy_contract_sha256": np.asarray(CONTRACT_SHA256),
            "g1_sim_eligible": np.asarray(g1_sim_eligible),
            "executable": np.asarray(False),
            "quarantined": np.asarray(not g1_sim_eligible),
            "source_report": np.asarray(str(args.output)),
        }
        if selected_audit.official_postprocessed_actions is not None:
            payload["official_postprocessed_actions"] = (
                selected_audit.official_postprocessed_actions
            )
        args.chunk_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.chunk_output, **payload)
        print(args.chunk_output)
    client.close()
    if not structural_output_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
