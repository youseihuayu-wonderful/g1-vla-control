#!/usr/bin/env python3
"""Quantify LGG100 stochastic action quality without executing any action."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from g1_policy_contract import ACTION_HORIZON
from lgg100_candidate_server import DEFAULT_PROMPT, HF_REVISION, build_policy
from lgg100_semantic_validation import _metrics, _normalize
from neural_action_audit import audit_neural_action_chunk

ROOT = Path(__file__).resolve().parent


def _quaternion_ensemble(values: np.ndarray) -> np.ndarray:
    """Sign-align and average [draw,sample,time,4] quaternion directions."""
    values = _normalize(values)
    reference = values[0:1]
    signs = np.where(
        np.sum(values * reference, axis=-1, keepdims=True) < 0.0, -1.0, 1.0
    )
    return _normalize(np.mean(values * signs, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "lgg100_stochastic_semantic_probe.json",
    )
    parser.add_argument(
        "--chunks", type=Path,
        default=ROOT / "results" / "lgg100_stochastic_chunks_quarantined.npz",
    )
    args = parser.parse_args()
    if args.draws < 2:
        raise SystemExit("draws must be at least 2")

    with np.load(args.samples, allow_pickle=False) as payload:
        cameras = {
            "observation/cam_left_high": np.asarray(payload["cam_left_high"]),
            "observation/cam_left_wrist": np.asarray(payload["cam_left_wrist"]),
            "observation/cam_right_wrist": np.asarray(payload["cam_right_wrist"]),
        }
        states = np.asarray(payload["state"], dtype=np.float64)
        reference = np.asarray(payload["reference_actions"], dtype=np.float64)
        episodes = np.asarray(payload["episode"])
        frames = np.asarray(payload["frame"])
        prompt = str(payload["prompt"].item())
    count = len(states)
    policy = build_policy(
        args.checkpoint_dir.resolve(), ACTION_HORIZON, DEFAULT_PROMPT
    )
    chunks = np.full(
        (args.draws, count, ACTION_HORIZON, 16), np.nan, dtype=np.float64
    )
    available = np.zeros((args.draws, count), dtype=bool)
    latency = np.zeros((args.draws, count), dtype=np.float64)
    raw_max_norm_error = np.full((args.draws, count), np.nan, dtype=np.float64)

    for draw in range(args.draws):
        for sample in range(count):
            observation = {key: value[sample] for key, value in cameras.items()}
            observation["observation/state"] = states[sample]
            observation["prompt"] = prompt
            start = time.monotonic()
            response = policy.infer(observation)
            latency[draw, sample] = (time.monotonic() - start) * 1000.0
            audit = audit_neural_action_chunk(response["actions"])
            raw_max_norm_error[draw, sample] = (
                audit.raw_max_quaternion_norm_error
                if audit.raw_max_quaternion_norm_error is not None else np.nan
            )
            if audit.canonicalized_actions_for_analysis is not None:
                chunks[draw, sample] = audit.canonicalized_actions_for_analysis
                available[draw, sample] = True
        print(json.dumps({
            "draw": draw,
            "bounded_analysis_available": int(np.sum(available[draw])),
            "samples": count,
        }), flush=True)

    hold = np.repeat(states[:, None, :], reference.shape[1], axis=1)
    hold_scores = np.asarray(_metrics(hold, reference, states)["per_sample_score"])
    draw_scores = np.full((args.draws, count), np.nan, dtype=np.float64)
    for draw in range(args.draws):
        for sample in np.flatnonzero(available[draw]):
            draw_scores[draw, sample] = _metrics(
                chunks[draw, sample:sample + 1],
                reference[sample:sample + 1],
                states[sample:sample + 1],
            )["score"]

    all_draws_available = np.all(available, axis=0)
    ensemble = np.full(
        (count, ACTION_HORIZON, 16), np.nan, dtype=np.float64
    )
    if np.any(all_draws_available):
        selected = chunks[:, all_draws_available]
        ensemble_values = np.mean(selected, axis=0)
        ensemble_values[..., 3:7] = _quaternion_ensemble(selected[..., 3:7])
        ensemble_values[..., 10:14] = _quaternion_ensemble(selected[..., 10:14])
        ensemble[all_draws_available] = ensemble_values
    ensemble_scores = np.full(count, np.nan, dtype=np.float64)
    for sample in np.flatnonzero(all_draws_available):
        ensemble_scores[sample] = _metrics(
            ensemble[sample:sample + 1],
            reference[sample:sample + 1],
            states[sample:sample + 1],
        )["score"]

    any_available = np.any(available, axis=0)
    best_draw_score = np.full(count, np.nan, dtype=np.float64)
    median_draw_score = np.full(count, np.nan, dtype=np.float64)
    best_draw_score[any_available] = np.nanmin(draw_scores[:, any_available], axis=0)
    median_draw_score[any_available] = np.nanmedian(draw_scores[:, any_available], axis=0)

    def comparison(scores: np.ndarray, mask: np.ndarray) -> dict:
        return {
            "samples": int(np.sum(mask)),
            "mean_score": float(np.mean(scores[mask])) if np.any(mask) else None,
            "hold_mean_score": float(np.mean(hold_scores[mask])) if np.any(mask) else None,
            "better_than_hold_rate": (
                float(np.mean(scores[mask] < hold_scores[mask])) if np.any(mask) else None
            ),
        }

    report = {
        "scope": "Stochastic offline LGG100 probe; output-only, quarantined, and never executable.",
        "checkpoint_revision": HF_REVISION,
        "draws_per_observation": args.draws,
        "action_horizon": ACTION_HORIZON,
        "observations": count,
        "total_inferences": args.draws * count,
        "bounded_analysis_availability_rate": float(np.mean(available)),
        "all_draws_available_observations": int(np.sum(all_draws_available)),
        "latency_ms": {
            "p50": float(np.quantile(latency, 0.50)),
            "p95": float(np.quantile(latency, 0.95)),
            "p99": float(np.quantile(latency, 0.99)),
            "max": float(np.max(latency)),
        },
        "raw_quaternion_norm_error": {
            "p50": float(np.nanquantile(raw_max_norm_error, 0.50)),
            "p95": float(np.nanquantile(raw_max_norm_error, 0.95)),
            "p99": float(np.nanquantile(raw_max_norm_error, 0.99)),
            "max": float(np.nanmax(raw_max_norm_error)),
        },
        "single_draw": comparison(draw_scores[0], available[0]),
        "median_draw": comparison(median_draw_score, any_available),
        "best_of_draws_diagnostic_only": comparison(best_draw_score, any_available),
        "ensemble_mean": comparison(ensemble_scores, all_draws_available),
        "g1_contract_verified": False,
        "g1_sim_eligible": False,
        "g1_execution_enabled": False,
        "execution_performed": False,
        "note": "Best-of-draws is diagnostic and may not be used online without a pre-registered action-selection rule.",
    }
    args.chunks.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.chunks,
        canonicalized_actions_for_analysis=chunks,
        bounded_available=available,
        draw_scores=draw_scores,
        ensemble_actions=ensemble,
        ensemble_scores=ensemble_scores,
        hold_scores=hold_scores,
        episode=episodes,
        frame=frames,
        executable=np.asarray(False),
        quarantined=np.asarray(True),
    )
    report["chunks_npz"] = str(args.chunks)
    report["chunks_npz_sha256"] = hashlib.sha256(args.chunks.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(args.chunks)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
