#!/usr/bin/env python3
"""Compare fixed LGG100 action hypotheses on public real-episode samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from g1_policy_contract import CONTRACT_ID, CONTRACT_SHA256, POLICY_RATE_HZ
from lgg100_candidate_server import HF_REVISION, OPENPI_AUDITED_COMMIT

ROOT = Path(__file__).resolve().parent


def _normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("zero quaternion in semantic hypothesis")
    return q / norm


def _multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _normalize(a)
    b = _normalize(b)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    result = np.stack((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ), axis=-1)
    return _normalize(result)


def _swap_arms(actions: np.ndarray) -> np.ndarray:
    result = actions.copy()
    result[..., 0:7] = actions[..., 7:14]
    result[..., 7:14] = actions[..., 0:7]
    result[..., 14] = actions[..., 15]
    result[..., 15] = actions[..., 14]
    return result


def _convert_quaternion_order(actions: np.ndarray) -> np.ndarray:
    """Interpret source quaternion fields as wxyz and publish xyzw."""
    result = actions.copy()
    for quaternion_slice in (slice(3, 7), slice(10, 14)):
        source = actions[..., quaternion_slice]
        result[..., quaternion_slice] = source[..., [1, 2, 3, 0]]
    return result


def _apply_delta(actions: np.ndarray, state: np.ndarray, order: str) -> np.ndarray:
    result = actions.copy()
    result[..., 0:3] += state[:, None, 0:3]
    result[..., 7:10] += state[:, None, 7:10]
    result[..., 14:16] += state[:, None, 14:16]
    for quaternion_slice in (slice(3, 7), slice(10, 14)):
        current = np.broadcast_to(
            state[:, None, quaternion_slice], result[..., quaternion_slice].shape
        )
        delta = result[..., quaternion_slice]
        result[..., quaternion_slice] = (
            _multiply_xyzw(current, delta)
            if order == "current_times_delta"
            else _multiply_xyzw(delta, current)
        )
    return result


def _metrics(candidate: np.ndarray, reference: np.ndarray, state: np.ndarray) -> dict:
    position_error = np.concatenate((
        candidate[..., 0:3] - reference[..., 0:3],
        candidate[..., 7:10] - reference[..., 7:10],
    ), axis=-1)
    position_norm = np.concatenate((
        np.linalg.norm(candidate[..., 0:3] - reference[..., 0:3], axis=-1),
        np.linalg.norm(candidate[..., 7:10] - reference[..., 7:10], axis=-1),
    ), axis=-1)
    quaternion_angles = []
    for quaternion_slice in (slice(3, 7), slice(10, 14)):
        lhs = _normalize(candidate[..., quaternion_slice])
        rhs = _normalize(reference[..., quaternion_slice])
        dot = np.clip(np.abs(np.sum(lhs * rhs, axis=-1)), 0.0, 1.0)
        quaternion_angles.append(np.rad2deg(2.0 * np.arccos(dot)))
    angle = np.concatenate(quaternion_angles, axis=-1)
    gripper_error = candidate[..., 14:16] - reference[..., 14:16]
    first_position_jump = np.maximum(
        np.linalg.norm(candidate[:, 0, 0:3] - state[:, 0:3], axis=-1),
        np.linalg.norm(candidate[:, 0, 7:10] - state[:, 7:10], axis=-1),
    )
    per_sample = (
        np.sqrt(np.mean(np.square(position_error), axis=(1, 2))) / 0.05
        + np.mean(angle, axis=1) / 15.0
        + np.sqrt(np.mean(np.square(gripper_error), axis=(1, 2))) / 0.50
    )
    return {
        "score": float(np.mean(per_sample)),
        "position_rmse_m": float(np.sqrt(np.mean(np.square(position_error)))),
        "position_error_norm_p95_m": float(np.quantile(position_norm, 0.95)),
        "quaternion_geodesic_mean_deg": float(np.mean(angle)),
        "quaternion_geodesic_p95_deg": float(np.quantile(angle, 0.95)),
        "gripper_rmse_rad": float(np.sqrt(np.mean(np.square(gripper_error)))),
        "first_target_jump_p95_m": float(np.quantile(first_position_jump, 0.95)),
        "per_sample_score": per_sample.tolist(),
    }


def _hypotheses(actions: np.ndarray, states: np.ndarray):
    for quaternion_order in ("xyzw", "wxyz"):
        ordered = (
            actions.copy()
            if quaternion_order == "xyzw"
            else _convert_quaternion_order(actions)
        )
        for swapped in (False, True):
            source = _swap_arms(ordered) if swapped else ordered
            suffix = "swapped" if swapped else "lr"
            yield f"absolute_{quaternion_order}_{suffix}", source
            for delta_order in ("current_times_delta", "delta_times_current"):
                yield (
                    f"delta_{quaternion_order}_{delta_order}_{suffix}",
                    _apply_delta(source, states, delta_order),
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "lgg100_semantic_validation.json",
    )
    parser.add_argument(
        "--attestation-candidate", type=Path,
        default=ROOT / "results" / "lgg100_semantic_attestation_candidate.json",
    )
    args = parser.parse_args()

    with np.load(args.samples, allow_pickle=False) as payload:
        states = np.asarray(payload["state"], dtype=np.float64)
        reference = np.asarray(payload["reference_actions"], dtype=np.float64)
        episodes = np.asarray(payload["episode"])
        frames = np.asarray(payload["frame"])
        dataset_revision = str(payload["dataset_revision"].item())
    with np.load(args.outputs, allow_pickle=False) as payload:
        actions = np.asarray(
            payload["canonicalized_actions_for_analysis"], dtype=np.float64
        )
        available = np.asarray(payload["canonicalized_available"], dtype=bool)
        checkpoint_revision = str(payload["checkpoint_revision"].item())
        openpi_commit = str(payload["openpi_commit"].item())
    if actions.shape != reference.shape or len(states) != len(actions):
        raise ValueError(
            f"Sample/output/reference mismatch {states.shape} {actions.shape} {reference.shape}"
        )
    total_sample_count = len(actions)
    usable = available & np.all(np.isfinite(actions), axis=(1, 2))
    unusable_indices = np.flatnonzero(~usable).tolist()
    actions = actions[usable]
    reference = reference[usable]
    states = states[usable]
    episodes = episodes[usable]
    frames = frames[usable]
    if len(actions) < 30:
        raise ValueError(
            f"Only {len(actions)}/{total_sample_count} samples satisfy the fixed bounded analysis gate; at least 30 are required"
        )

    metrics = {
        name: _metrics(candidate, reference, states)
        for name, candidate in _hypotheses(actions, states)
    }
    hold = np.repeat(states[:, None, :], reference.shape[1], axis=1)
    hold_metrics = _metrics(hold, reference, states)
    ranking = sorted(metrics, key=lambda name: metrics[name]["score"])
    best, second = ranking[:2]
    margin = (
        (metrics[second]["score"] - metrics[best]["score"])
        / max(metrics[second]["score"], 1e-12)
    )
    per_sample_scores = np.asarray([
        metrics[name]["per_sample_score"] for name in ranking
    ])
    winners = np.argmin(per_sample_scores, axis=0)
    best_win_rate = float(np.mean(winners == 0))
    analysis_availability_rate = len(actions) / total_sample_count
    semantic_criteria = {
        "at_least_30_usable_samples": len(actions) >= 30,
        "at_least_3_usable_episodes": len(np.unique(episodes)) >= 3,
        "bounded_analysis_availability_at_least_90_percent": (
            analysis_availability_rate >= 0.90
        ),
        "expected_hypothesis_wins": best == "absolute_xyzw_lr",
        "best_hypothesis_margin_at_least_10_percent": margin >= 0.10,
        "best_hypothesis_sample_win_rate_at_least_70_percent": best_win_rate >= 0.70,
        "position_rmse_below_0_15_m": metrics[best]["position_rmse_m"] < 0.15,
        "quaternion_mean_below_45_deg": (
            metrics[best]["quaternion_geodesic_mean_deg"] < 45.0
        ),
        "gripper_rmse_below_1_rad": metrics[best]["gripper_rmse_rad"] < 1.0,
    }
    policy_quality_criteria = {
        "single_draw_score_better_than_hold": (
            metrics[best]["score"] < hold_metrics["score"]
        ),
    }
    semantic_identification_supported = all(semantic_criteria.values())
    offline_single_draw_policy_quality_passed = all(
        policy_quality_criteria.values()
    )
    report = {
        "scope": "Behavioral semantic hypothesis comparison on real public episodes; no execution.",
        "total_sample_count": total_sample_count,
        "usable_sample_count": len(actions),
        "bounded_analysis_availability_rate": analysis_availability_rate,
        "unusable_sample_indices": unusable_indices,
        "usable_episodes": sorted(int(value) for value in np.unique(episodes)),
        "usable_frames": frames.tolist(),
        "dataset_revision": dataset_revision,
        "checkpoint_revision": checkpoint_revision,
        "openpi_commit": openpi_commit,
        "g1_policy_contract_id": CONTRACT_ID,
        "g1_policy_contract_sha256": CONTRACT_SHA256,
        "policy_rate_hz": POLICY_RATE_HZ,
        "quaternion_normalization_scope": "bounded_quarantined_analysis_only",
        "ranking": ranking,
        "best_hypothesis": best,
        "second_hypothesis": second,
        "best_margin_fraction": margin,
        "best_sample_win_rate": best_win_rate,
        "hypotheses": metrics,
        "hold_baseline": hold_metrics,
        "semantic_identification_criteria": semantic_criteria,
        "policy_quality_criteria": policy_quality_criteria,
        "semantic_identification_supported": semantic_identification_supported,
        "offline_single_draw_policy_quality_passed": (
            offline_single_draw_policy_quality_passed
        ),
        "operational_semantics_supported": semantic_identification_supported,
        "author_transform_recovered": False,
        "g1_contract_verified": False,
        "g1_sim_eligible": False,
        "execution_performed": False,
        "verdict": (
            "The frozen absolute pelvis-frame xyzw left/right hypothesis passed all semantic-identification criteria. Single-draw offline policy quality is reported separately; manual hash-bound review is still required before simulation eligibility."
            if semantic_identification_supported else
            "Semantic hypotheses did not pass every semantic-identification criterion. Keep all neural actions quarantined."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    attestation = {
        "status": (
            "candidate_pending_manual_review"
            if semantic_identification_supported else "rejected"
        ),
        "semantic_identification_supported": semantic_identification_supported,
        "offline_single_draw_policy_quality_passed": (
            offline_single_draw_policy_quality_passed
        ),
        "operational_semantics_supported": semantic_identification_supported,
        "g1_contract_verified": False,
        "g1_sim_eligible": False,
        "checkpoint_revision": HF_REVISION,
        "openpi_commit": OPENPI_AUDITED_COMMIT,
        "dataset_revision": dataset_revision,
        "g1_policy_contract_id": CONTRACT_ID,
        "g1_policy_contract_sha256": CONTRACT_SHA256,
        "sample_npz_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        "output_npz_sha256": hashlib.sha256(args.outputs.read_bytes()).hexdigest(),
        "sample_manifest_sha256": hashlib.sha256(
            args.sample_manifest.read_bytes()
        ).hexdigest(),
        "validation_report_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "selected_hypothesis": best,
        "semantic_identification_criteria": semantic_criteria,
        "policy_quality_criteria": policy_quality_criteria,
        "reviewer": None,
        "reviewed_at": None,
    }
    args.attestation_candidate.write_text(json.dumps(attestation, indent=2) + "\n")
    print(args.output)
    print(args.attestation_candidate)
    print(json.dumps({
        "best": best,
        "second": second,
        "margin": margin,
        "sample_win_rate": best_win_rate,
        "best_metrics": metrics[best],
        "hold_score": hold_metrics["score"],
        "semantic_identification_criteria": semantic_criteria,
        "policy_quality_criteria": policy_quality_criteria,
        "semantic_identification_supported": semantic_identification_supported,
        "offline_single_draw_policy_quality_passed": (
            offline_single_draw_policy_quality_passed
        ),
    }, indent=2))
    if not semantic_identification_supported:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
