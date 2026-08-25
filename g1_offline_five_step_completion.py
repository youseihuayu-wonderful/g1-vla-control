#!/usr/bin/env python3
"""Aggregate the five offline G1 work packages without promoting live gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def aggregate(results_dir: Path) -> dict[str, Any]:
    definitions = [
        (
            "H2_fk_source_parity",
            "g1_yuhao_pinocchio_mujoco_fk_source_parity_20260824.json",
            ("decision", "offline_h2_core_completed"),
        ),
        (
            "H3_dds_offline_forensic",
            "g1_dds_freshness_offline_analysis_20260824.json",
            ("decision", "offline_h3_forensic_completed"),
        ),
        (
            "H5_mock_fault_injection",
            "g1_fail_closed_mock_fault_injection_20260824.json",
            ("decision", "offline_h5_mock_fault_suite_passed"),
        ),
        (
            "H4_policy_shadow_replay",
            "g1_offline_policy_shadow_replay_20260824.json",
            ("decision", "offline_h4_replay_harness_completed"),
        ),
        (
            "H6_first_motion_checklist",
            "g1_first_motion_review_checklist_20260824.json",
            ("decision", "offline_h6_checklist_prepared"),
        ),
    ]
    steps = []
    for index, (name, filename, key_path) in enumerate(definitions, start=1):
        path = results_dir / filename
        report, sha256 = _load(path)
        value: Any = report
        for key in key_path:
            value = value[key]
        steps.append({
            "step": index,
            "name": name,
            "evidence_file": filename,
            "evidence_sha256": sha256,
            "offline_work_completed": value is True,
            "live_gate_promoted": False,
            "robot_motion_allowed": False,
        })
    all_completed = all(step["offline_work_completed"] for step in steps)
    return {
        "schema_version": "g1_offline_five_step_completion_v1",
        "scope": "Aggregate five offline work packages; no live gate or command authority.",
        "execution_order": [step["name"] for step in steps],
        "steps": steps,
        "summary": {
            "offline_steps_total": len(steps),
            "offline_steps_completed": sum(
                step["offline_work_completed"] for step in steps
            ),
            "all_offline_steps_completed": all_completed,
            "live_gates_completed": 0,
        },
        "remaining_live_blockers": [
            "real three-camera image server and frame capture",
            "physical EEF and camera calibration",
            "instrumented DDS recapture and final freshness threshold",
            "real streaming 15 Hz frozen-LGG100 Policy Shadow",
            "integrated watchdog and independently reviewed hardware adapter",
            "official hardware limits, trial-day checks, and dual authorization",
        ],
        "decision": {
            "offline_five_step_plan_completed": all_completed,
            "real_policy_shadow_passed": False,
            "motion_review_passed": False,
            "motion_authorized": False,
            "robot_motion_allowed": False,
        },
        "safety": {
            "offline_only": True,
            "network_accessed": False,
            "publisher_created": False,
            "hardware_transport_present": False,
            "robot_command_sent": False,
            "mode_change_requested": False,
        },
        "g1_execution_enabled": False,
        "hardware_execution_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["summary"]["all_offline_steps_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
