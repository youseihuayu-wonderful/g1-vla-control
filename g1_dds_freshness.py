#!/usr/bin/env python3
"""Offline DDS freshness analysis and fail-closed LowState watchdog.

The official LowState ``tick`` field has no time unit in the pinned IDL. This
module therefore uses local monotonic receive time for deadlines and only uses
tick changes as evidence of source progress. It contains no DDS publisher or
robot command path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


UINT32_MODULUS = 1 << 32
UINT32_HALF_RANGE = 1 << 31
DEFAULT_WARNING_MS = 20.0
DEFAULT_STALE_HOLD_MS = 50.0
DEFAULT_DISCONNECT_MS = 200.0
DEFAULT_RECOVERY_UNIQUE_TICKS = 2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tick_relation(previous: int, current: int) -> tuple[str, int]:
    delta = (int(current) - int(previous)) % UINT32_MODULUS
    if delta == 0:
        return "duplicate", 0
    if delta < UINT32_HALF_RANGE:
        return "forward", delta
    return "backward_or_reset", delta


@dataclass(frozen=True)
class FreshnessThresholds:
    warning_ms: float = DEFAULT_WARNING_MS
    stale_hold_ms: float = DEFAULT_STALE_HOLD_MS
    disconnect_ms: float = DEFAULT_DISCONNECT_MS
    recovery_unique_ticks: int = DEFAULT_RECOVERY_UNIQUE_TICKS

    def __post_init__(self) -> None:
        if not 0 < self.warning_ms < self.stale_hold_ms < self.disconnect_ms:
            raise ValueError("require 0 < warning < stale_hold < disconnect")
        if self.recovery_unique_ticks < 1:
            raise ValueError("recovery_unique_ticks must be positive")


class LowStateFreshnessWatchdog:
    """Track message age and unique-tick age without sending any command."""

    def __init__(self, thresholds: FreshnessThresholds = FreshnessThresholds()):
        self.thresholds = thresholds
        self.last_message_ns: int | None = None
        self.last_unique_tick_ns: int | None = None
        self.last_tick: int | None = None
        self.dds_connected = True
        self.fault_reason: str | None = None
        self.requires_recovery = True
        self.consecutive_unique_ticks = 0
        self.duplicate_count = 0
        self.forward_jump_count = 0
        self.backward_or_reset_count = 0

    def set_dds_connected(self, connected: bool) -> None:
        self.dds_connected = bool(connected)
        if not self.dds_connected:
            self.fault_reason = "dds_disconnected"
            self.requires_recovery = True
            self.consecutive_unique_ticks = 0

    def observe(
        self,
        *,
        received_monotonic_ns: int,
        tick: int,
        finite: bool = True,
    ) -> str:
        received_monotonic_ns = int(received_monotonic_ns)
        tick = int(tick)
        if received_monotonic_ns < 0:
            raise ValueError("received_monotonic_ns must be non-negative")
        if self.last_message_ns is not None and received_monotonic_ns < self.last_message_ns:
            self.fault_reason = "receive_clock_decreased"
            self.requires_recovery = True
            self.consecutive_unique_ticks = 0
            return "invalid"
        self.last_message_ns = received_monotonic_ns
        if not finite:
            self.fault_reason = "lowstate_nonfinite"
            self.requires_recovery = True
            self.consecutive_unique_ticks = 0
            return "invalid"

        if self.last_tick is None:
            relation, delta = "first", 0
        else:
            relation, delta = _tick_relation(self.last_tick, tick)
        if relation == "duplicate":
            self.duplicate_count += 1
            return relation
        if relation == "backward_or_reset":
            self.backward_or_reset_count += 1
            self.last_tick = tick
            self.fault_reason = "tick_backward_or_reset"
            self.requires_recovery = True
            self.consecutive_unique_ticks = 0
            return relation

        self.last_tick = tick
        self.last_unique_tick_ns = received_monotonic_ns
        if relation == "forward" and delta > 1:
            self.forward_jump_count += 1
        self.consecutive_unique_ticks += 1
        if self.consecutive_unique_ticks >= self.thresholds.recovery_unique_ticks:
            self.requires_recovery = False
            self.fault_reason = None
        return relation

    def evaluate(self, *, now_monotonic_ns: int) -> dict[str, Any]:
        now_monotonic_ns = int(now_monotonic_ns)
        message_age_ms = (
            None
            if self.last_message_ns is None
            else (now_monotonic_ns - self.last_message_ns) / 1_000_000.0
        )
        unique_tick_age_ms = (
            None
            if self.last_unique_tick_ns is None
            else (now_monotonic_ns - self.last_unique_tick_ns) / 1_000_000.0
        )
        if (
            message_age_ms is not None and message_age_ms < 0
        ) or (
            unique_tick_age_ms is not None and unique_tick_age_ms < 0
        ):
            self.fault_reason = "evaluation_clock_precedes_observation"
            self.requires_recovery = True
            self.consecutive_unique_ticks = 0

        reason = self.fault_reason
        state = "fresh"
        if not self.dds_connected:
            state, reason = "disconnected", "dds_disconnected"
        elif message_age_ms is None or unique_tick_age_ms is None:
            state, reason = "hold", "lowstate_unavailable"
        elif (
            message_age_ms >= self.thresholds.disconnect_ms
            or unique_tick_age_ms >= self.thresholds.disconnect_ms
        ):
            state, reason = "disconnected", "lowstate_disconnect_timeout"
        elif (
            message_age_ms >= self.thresholds.stale_hold_ms
            or unique_tick_age_ms >= self.thresholds.stale_hold_ms
        ):
            state, reason = "hold", "lowstate_stale"
        elif self.requires_recovery:
            state = "hold"
            reason = reason or "insufficient_unique_tick_recovery"
        elif (
            message_age_ms >= self.thresholds.warning_ms
            or unique_tick_age_ms >= self.thresholds.warning_ms
        ):
            state, reason = "warning", "lowstate_age_warning"

        hold = state in {"hold", "disconnected"}
        if reason in {
            "dds_disconnected",
            "lowstate_disconnect_timeout",
            "lowstate_stale",
        }:
            self.requires_recovery = True
            self.consecutive_unique_ticks = 0
        return {
            "state": state,
            "reason": reason,
            "message_age_ms": message_age_ms,
            "unique_tick_age_ms": unique_tick_age_ms,
            "hold": hold,
            "publish_allowed": not hold,
            "robot_command_sent": False,
            "last_tick": self.last_tick,
            "consecutive_unique_ticks": self.consecutive_unique_ticks,
            "diagnostics": {
                "duplicate_count": self.duplicate_count,
                "forward_jump_count": self.forward_jump_count,
                "backward_or_reset_count": self.backward_or_reset_count,
            },
        }


def analyze_existing_captures(
    capture_paths: list[Path], sdk_channel_path: Path
) -> dict[str, Any]:
    captures = [json.loads(path.read_text()) for path in capture_paths]
    extracted = []
    total_samples = 0
    total_duplicates = 0
    total_reader_errors_lower_bound = 0
    maximum_gap_ms = 0.0
    for path, report in zip(capture_paths, captures):
        capture = report["capture"]
        samples = int(capture.get("captured_samples", 0))
        tick = capture["tick"]
        duplicates = int(tick["duplicate_sample_count"])
        reader_errors = capture.get("reader_take_sample_errors_observed")
        if reader_errors is None:
            reader_errors = 1 if capture.get("reader_take_sample_error_observed") else 0
        timing = capture["timing_gap_ms"]
        max_gap = float(timing.get("gap_max_ms", timing.get("max", 0.0)))
        total_samples += samples
        total_duplicates += duplicates
        total_reader_errors_lower_bound += int(reader_errors)
        maximum_gap_ms = max(maximum_gap_ms, max_gap)
        extracted.append({
            "file": path.name,
            "sha256": _sha256(path),
            "captured_samples": samples,
            "reader_errors_observed": int(reader_errors),
            "duplicate_ticks": duplicates,
            "tick_decreases": int(tick["decrease_count"]),
            "median_receive_gap_ms": float(
                timing.get("gap_median_ms", timing.get("median", 0.0))
            ),
            "maximum_receive_gap_ms": max_gap,
        })

    sdk_source = sdk_channel_path.read_text()
    generic_reader_error_is_bare_except = (
        'print("[Reader] take sample error")' in sdk_source
        and "except:" in sdk_source
    )
    thresholds = FreshnessThresholds()
    return {
        "schema_version": "g1_dds_freshness_offline_analysis_v1",
        "scope": "Offline forensic analysis of bounded subscriber-only LowState summaries.",
        "inputs": extracted,
        "aggregate": {
            "captured_samples": total_samples,
            "duplicate_ticks": total_duplicates,
            "duplicate_fraction": total_duplicates / total_samples,
            "reader_errors_observed_lower_bound": total_reader_errors_lower_bound,
            "reader_errors_per_captured_sample_lower_bound": (
                total_reader_errors_lower_bound / total_samples
            ),
            "tick_decreases": sum(item["tick_decreases"] for item in extracted),
            "maximum_receive_gap_ms": maximum_gap_ms,
        },
        "source_audit": {
            "sdk_channel_file": sdk_channel_path.name,
            "sdk_channel_sha256": _sha256(sdk_channel_path),
            "reader_error_uses_generic_bare_except": generic_reader_error_is_bare_except,
            "reader_error_exact_exception_preserved": False,
            "reader_error_exact_cause_identified": False,
            "official_lowstate_tick_type": "uint32",
            "official_lowstate_tick_period_or_unit_documented_in_pinned_idl": False,
        },
        "interpretation": {
            "duplicate_tick_proves_dds_duplicate_delivery": False,
            "duplicate_tick_proves_robot_tick_semantics": False,
            "reason": (
                "The available summaries do not preserve DDS sample identity or the "
                "exception type, and the official IDL specifies only uint32 tick. "
                "A duplicate tick is therefore treated conservatively as no new "
                "source progress, not classified as a DDS or robot duplicate."
            ),
            "tick_jump_can_be_converted_to_time": False,
            "watchdog_clock": "local_monotonic_receive_time",
            "tick_usage": "progress_only_with_uint32_rollover_handling",
        },
        "provisional_thresholds": {
            "warning_ms": thresholds.warning_ms,
            "stale_hold_ms": thresholds.stale_hold_ms,
            "disconnect_ms": thresholds.disconnect_ms,
            "recovery_unique_ticks": thresholds.recovery_unique_ticks,
            "sample_interval_at_15_hz_ms": 1000.0 / 15.0,
            "stale_hold_is_less_than_one_policy_interval": (
                thresholds.stale_hold_ms < 1000.0 / 15.0
            ),
            "basis": (
                "50 ms is more than 12x the observed 3.98534 ms maximum receive "
                "gap but remains below one 15 Hz policy interval; 200 ms is three "
                "policy intervals."
            ),
            "final_hardware_threshold_frozen": False,
        },
        "decision": {
            "offline_h3_forensic_completed": True,
            "candidate_watchdog_preregistered": True,
            "reader_error_root_cause_resolved": False,
            "duplicate_tick_origin_resolved": False,
            "tick_period_resolved": False,
            "dds_freshness_fully_qualified": False,
            "live_recapture_required": True,
            "robot_motion_allowed": False,
        },
        "safety": {
            "offline_only": True,
            "network_accessed": False,
            "publisher_created": False,
            "robot_command_sent": False,
        },
        "g1_execution_enabled": False,
        "hardware_execution_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--sdk-channel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_existing_captures(args.capture, args.sdk_channel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
