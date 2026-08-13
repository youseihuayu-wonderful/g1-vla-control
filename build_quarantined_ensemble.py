#!/usr/bin/env python3
"""Build a fixed mean ensemble from quarantined bounded LGG100 draws."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _normalize_quaternions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(~np.isfinite(values)) or np.any(norms <= 1e-12):
        raise ValueError("quaternions must be finite and nonzero")
    return values / norms


def quaternion_ensemble(values: np.ndarray) -> np.ndarray:
    """Sign-align [draw,...,4] unit directions and return their unit mean."""
    values = _normalize_quaternions(values)
    reference = values[0:1]
    signs = np.where(
        np.sum(values * reference, axis=-1, keepdims=True) < 0.0,
        -1.0,
        1.0,
    )
    return _normalize_quaternions(np.mean(values * signs, axis=0))


def build_ensemble(draws: np.ndarray) -> np.ndarray:
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 3 or draws.shape[1:] != (50, 16):
        raise ValueError(f"expected [draw,50,16], got {draws.shape}")
    if len(draws) < 3 or np.any(~np.isfinite(draws)):
        raise ValueError("at least three finite bounded draws are required")
    ensemble = np.mean(draws, axis=0)
    ensemble[:, 3:7] = quaternion_ensemble(draws[:, :, 3:7])
    ensemble[:, 10:14] = quaternion_ensemble(draws[:, :, 10:14])
    return ensemble


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    probe = json.loads(args.probe_report.read_text())
    source_hash = _sha256(args.draws)
    if probe.get("output_npz_sha256") != source_hash:
        raise ValueError("probe report does not bind the source draws")
    if probe.get("g1_execution_enabled") is not False:
        raise ValueError("source probe must remain non-executable")
    with np.load(args.draws, allow_pickle=False) as payload:
        if not bool(payload["quarantined"].item()):
            raise ValueError("source draws must be quarantined")
        if bool(payload["executable"].item()):
            raise ValueError("source draws unexpectedly executable")
        draws = np.asarray(
            payload["canonicalized_actions_for_analysis"], dtype=np.float64
        )
        observation_state = np.asarray(payload["observation_state"])
        checkpoint_revision = str(payload["checkpoint_revision"].item())
    ensemble = build_ensemble(draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        actions=ensemble[None, ...],
        observation_state=observation_state,
        checkpoint_revision=np.asarray(checkpoint_revision),
        source_draw_count=np.asarray(len(draws)),
        source_draws_sha256=np.asarray(source_hash),
        executable=np.asarray(False),
        quarantined=np.asarray(True),
        derived_bounded_analysis=np.asarray(True),
    )
    report = {
        "scope": "Fixed component/quaternion mean of quarantined bounded LGG100 draws; diagnostic only.",
        "selection_rule": "mean_position_gripper_and_sign_aligned_unit_quaternion_mean",
        "source_draw_count": len(draws),
        "source_draws": str(args.draws),
        "source_draws_sha256": source_hash,
        "source_probe_report": str(args.probe_report),
        "source_probe_report_sha256": _sha256(args.probe_report),
        "source_observation": probe.get("observation"),
        "source_observation_sha256": probe.get("observation_sha256"),
        "checkpoint_revision": checkpoint_revision,
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "g1_contract_verified": False,
        "g1_sim_eligible": False,
        "g1_execution_enabled": False,
        "execution_performed": False,
        "quarantined": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
