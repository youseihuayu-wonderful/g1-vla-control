#!/usr/bin/env python3
"""Low-risk G1 speed-optimization hardware gate check.

This is a read-only / shadow-only check. It may open TCP sockets to verify that
known robot-network endpoints are reachable, but it never authenticates, never
starts DDS, never creates a Unitree publisher, and never sends robot commands.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import socket
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class TcpEndpoint:
    name: str
    host: str
    port: int


DEFAULT_ENDPOINTS = (
    TcpEndpoint("robot_pc2_ssh", "192.168.1.11", 22),
    TcpEndpoint("development_pc_ssh", "192.168.1.13", 22),
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def tcp_probe(endpoint: TcpEndpoint, timeout_s: float) -> dict[str, Any]:
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    started = time.monotonic_ns()
    try:
        with socket.create_connection((endpoint.host, endpoint.port), timeout=timeout_s):
            reachable = True
            error = None
    except OSError as exc:
        reachable = False
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.monotonic_ns() - started) / 1_000_000.0
    return {
        **asdict(endpoint),
        "tcp_connect_reachable": reachable,
        "elapsed_ms": elapsed_ms,
        "error": error,
        "authentication_attempted": False,
        "remote_command_executed": False,
    }


def evaluate_artifacts(
    *,
    trajectory_analysis: dict[str, Any],
    speed_schedule: dict[str, Any],
    offline_replay: dict[str, Any],
    six_gate: dict[str, Any],
    d0_d1: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "trajectory_analysis_completed": (
            trajectory_analysis.get("decision", {}).get("trajectory_analyzer_completed") is True
        ),
        "speed_schedule_completed": (
            speed_schedule.get("decision", {}).get("speed_scheduler_completed") is True
            and speed_schedule.get("summary", {}).get("all_episodes_accepted") is True
            and speed_schedule.get("decision", {}).get("path_actions_byte_identical_all") is True
        ),
        "offline_replay_completed": (
            offline_replay.get("decision", {}).get("offline_replay_comparison_completed") is True
            and offline_replay.get("summary", {}).get("all_episodes_accepted") is True
            and offline_replay.get("decision", {}).get("risky_intervals_not_accelerated_all") is True
            and offline_replay.get("decision", {}).get("endpoints_preserved_all") is True
        ),
        "six_gate_motion_locked": (
            six_gate.get("safety_baseline", {}).get("publisher_created") is False
            and six_gate.get("safety_baseline", {}).get("robot_command_sent") is False
            and six_gate.get("safety_baseline", {}).get("hardware_execution_performed") is False
        ),
        "six_gate_not_fully_resolved": (
            int(six_gate.get("resolved_gate_count", 0)) < int(six_gate.get("total_gate_count", 6))
        ),
        "d0_d1_readonly_passed": (
            d0_d1.get("d0", {}).get("passed") is True
            and d0_d1.get("d1", {}).get("passed") is True
            and d0_d1.get("safety", {}).get("send_channel_created") is False
            and d0_d1.get("safety", {}).get("robot_command_publisher_created") is False
            and d0_d1.get("safety", {}).get("robot_command_sent") is False
            and d0_d1.get("safety", {}).get("hardware_motion_performed") is False
        ),
        "no_prior_hardware_motion_claim": (
            trajectory_analysis.get("decision", {}).get("hardware_execution_performed") is False
            and speed_schedule.get("decision", {}).get("hardware_execution_performed") is False
            and offline_replay.get("decision", {}).get("hardware_execution_performed") is False
            and d0_d1.get("hardware_execution_performed") is False
        ),
    }
    return checks


def run_low_risk_check(
    *,
    trajectory_analysis_path: Path = RESULTS / "trajectory_analysis.json",
    speed_schedule_path: Path = RESULTS / "speed_schedule.json",
    offline_replay_path: Path = RESULTS / "offline_replay_comparison.json",
    six_gate_path: Path = RESULTS / "g1_six_gate_execution_status_20260824.json",
    d0_d1_path: Path = RESULTS / "g1_dds_direct_d0_d1_hardware_20260829.json",
    output: Path = RESULTS / "g1_speed_low_risk_hardware_check.json",
    endpoints: tuple[TcpEndpoint, ...] = DEFAULT_ENDPOINTS,
    tcp_timeout_s: float = 1.0,
    skip_tcp: bool = False,
) -> dict[str, Any]:
    trajectory_analysis = load_json(trajectory_analysis_path)
    speed_schedule = load_json(speed_schedule_path)
    offline_replay = load_json(offline_replay_path)
    six_gate = load_json(six_gate_path)
    d0_d1 = load_json(d0_d1_path)

    checks = evaluate_artifacts(
        trajectory_analysis=trajectory_analysis,
        speed_schedule=speed_schedule,
        offline_replay=offline_replay,
        six_gate=six_gate,
        d0_d1=d0_d1,
    )
    tcp_results = [] if skip_tcp else [tcp_probe(endpoint, tcp_timeout_s) for endpoint in endpoints]
    any_tcp_reachable = any(result["tcp_connect_reachable"] for result in tcp_results)
    all_artifact_checks_passed = all(checks.values())

    unresolved_gate_count = int(six_gate.get("total_gate_count", 6)) - int(
        six_gate.get("resolved_gate_count", 0)
    )
    report = {
        "schema_version": "g1_speed_low_risk_hardware_check_v1",
        "scope": (
            "Read-only speed-optimization hardware gate check: artifact consistency, "
            "no-motion safety lock, and optional TCP reachability only."
        ),
        "inputs": {
            "trajectory_analysis": str(trajectory_analysis_path),
            "speed_schedule": str(speed_schedule_path),
            "offline_replay": str(offline_replay_path),
            "six_gate": str(six_gate_path),
            "d0_d1": str(d0_d1_path),
        },
        "artifact_checks": checks,
        "tcp_reachability": {
            "attempted": not skip_tcp,
            "timeout_s": tcp_timeout_s,
            "any_endpoint_reachable": any_tcp_reachable,
            "results": tcp_results,
        },
        "speed_shadow_summary": {
            "episodes": offline_replay.get("episode_count"),
            "duration_reduction_fraction": offline_replay.get("summary", {}).get(
                "total_duration_reduction_fraction"
            ),
            "fixed_rate_replay_frame_reduction_fraction": offline_replay.get("summary", {}).get(
                "fixed_rate_replay_frame_reduction_fraction"
            ),
            "risk_accelerated_episode_count": offline_replay.get("summary", {}).get(
                "risk_accelerated_episode_count"
            ),
        },
        "gate_state": {
            "current_gate": six_gate.get("current_gate"),
            "resolved_gate_count": six_gate.get("resolved_gate_count"),
            "total_gate_count": six_gate.get("total_gate_count"),
            "unresolved_gate_count": unresolved_gate_count,
            "robot_command_sent": six_gate.get("safety_baseline", {}).get("robot_command_sent"),
            "publisher_created": six_gate.get("safety_baseline", {}).get("publisher_created"),
            "hardware_execution_performed": six_gate.get("safety_baseline", {}).get(
                "hardware_execution_performed"
            ),
        },
        "decision": {
            "low_risk_hardware_check_completed": True,
            "artifact_checks_passed": all_artifact_checks_passed,
            "tcp_reachability_required_for_completion": False,
            "read_only_shadow_ready": all_artifact_checks_passed,
            "real_robot_motion_test_performed": False,
            "publisher_created": False,
            "robot_command_sent": False,
            "hardware_execution_performed": False,
            "low_risk_motion_allowed": False,
            "robot_motion_allowed": False,
            "reason": (
                "Speed scheduler is only qualified for read-only/shadow use. "
                "Motion remains locked because not all H1-H6 gates are resolved and H6 authorization is absent."
            ),
        },
    }
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def _endpoint(value: str) -> TcpEndpoint:
    try:
        name, host, port = value.split(":", 2)
        return TcpEndpoint(name=name, host=host, port=int(port))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "endpoint must be name:host:port"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-analysis", type=Path, default=RESULTS / "trajectory_analysis.json")
    parser.add_argument("--speed-schedule", type=Path, default=RESULTS / "speed_schedule.json")
    parser.add_argument("--offline-replay", type=Path, default=RESULTS / "offline_replay_comparison.json")
    parser.add_argument("--six-gate", type=Path, default=RESULTS / "g1_six_gate_execution_status_20260824.json")
    parser.add_argument("--d0-d1", type=Path, default=RESULTS / "g1_dds_direct_d0_d1_hardware_20260829.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "g1_speed_low_risk_hardware_check.json")
    parser.add_argument("--tcp-timeout-s", type=float, default=1.0)
    parser.add_argument("--endpoint", action="append", type=_endpoint, default=None)
    parser.add_argument("--skip-tcp", action="store_true")
    args = parser.parse_args()
    endpoints = tuple(args.endpoint) if args.endpoint else DEFAULT_ENDPOINTS
    report = run_low_risk_check(
        trajectory_analysis_path=args.trajectory_analysis,
        speed_schedule_path=args.speed_schedule,
        offline_replay_path=args.offline_replay,
        six_gate_path=args.six_gate,
        d0_d1_path=args.d0_d1,
        output=args.output,
        endpoints=endpoints,
        tcp_timeout_s=args.tcp_timeout_s,
        skip_tcp=args.skip_tcp,
    )
    print(args.output)
    print(json.dumps({
        "artifact_checks_passed": report["decision"]["artifact_checks_passed"],
        "read_only_shadow_ready": report["decision"]["read_only_shadow_ready"],
        "tcp_any_endpoint_reachable": report["tcp_reachability"]["any_endpoint_reachable"],
        "current_gate": report["gate_state"]["current_gate"],
        "unresolved_gate_count": report["gate_state"]["unresolved_gate_count"],
        "robot_motion_allowed": report["decision"]["robot_motion_allowed"],
    }, indent=2))
    return 0 if report["decision"]["artifact_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
