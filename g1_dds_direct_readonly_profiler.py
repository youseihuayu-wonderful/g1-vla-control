#!/usr/bin/env python3
"""Direct subscriber-only CycloneDDS profiler for the real-G1 LowState path.

This diagnostic bypasses the Unitree SDK callback/BQueue layer and directly
uses a CycloneDDS DataReader plus WaitSet. It records source timestamps,
publication handles, wait/take timing, batch backlog, tick behavior, and q29
payload hashes. It has no command topic, controller, mode switch, or robot
motion path. Raw q29 records are private hardware evidence and must not be
committed to public Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import statistics
import time
from typing import Any

from g1_unitree_lowstate import (
    FULL_BODY_INDICES,
    LOWSTATE_TOPIC,
    OFFICIAL_SDK_COMMIT,
    extract_lowstate_snapshot,
    validate_network_interface,
)


CYCLONEDDS_VERSION = "0.10.2"
CYCLONEDDS_SDIST_SHA256 = (
    "f834962eabbdcdf4e9cd75cf87222f3c5ef22d9cb9e7ed651d9a8710fe984a30"
)
PROVISIONAL_STALE_HOLD_MS = 50.0
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _timing(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p95": None,
            "maximum": None,
            "over_provisional_50ms_count": 0,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
        "over_provisional_50ms_count": sum(
            value > PROVISIONAL_STALE_HOLD_MS for value in values
        ),
    }


def _transitions(values: list[int]) -> list[int]:
    return [right - left for left, right in zip(values, values[1:])]


def _process_metrics() -> dict[str, float | int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_cpu_s": usage.ru_utime,
        "system_cpu_s": usage.ru_stime,
        "maximum_rss": usage.ru_maxrss,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def _network_metrics(network_interface: str) -> dict[str, int | None]:
    base = Path("/sys/class/net") / network_interface / "statistics"
    result: dict[str, int | None] = {}
    for name in ("rx_packets", "rx_dropped", "rx_errors", "rx_missed_errors"):
        try:
            result[name] = int((base / name).read_text().strip())
        except (FileNotFoundError, PermissionError, ValueError):
            result[name] = None
    return result


def _metric_delta(
    before: dict[str, int | float | None],
    after: dict[str, int | float | None],
) -> dict[str, int | float | None]:
    return {
        key: (
            after[key] - value
            if value is not None and after.get(key) is not None
            else None
        )
        for key, value in before.items()
    }


def summarize_profile(
    records: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    status_events: dict[str, int],
) -> dict[str, Any]:
    ticks = [int(record["tick"]) for record in records]
    tick_steps = _transitions(ticks)
    application_ns = [int(record["application_monotonic_ns"]) for record in records]
    application_gaps_ms = [step / 1_000_000.0 for step in _transitions(application_ns)]
    all_source_ns = [int(record["source_timestamp_ns"]) for record in records]
    source_ns = [value for value in all_source_ns if value > 0]
    source_steps = [
        right - left
        for left, right in zip(all_source_ns, all_source_ns[1:])
        if left > 0 and right > 0
    ]
    source_positive_gaps_ms = [step / 1_000_000.0 for step in source_steps if step > 0]
    identity = [
        (int(record["publication_handle"]), int(record["source_timestamp_ns"]))
        for record in records
        if int(record["source_timestamp_ns"]) > 0
    ]
    duplicate_tick_indices = [index for index, step in enumerate(tick_steps) if step == 0]
    duplicate_tick_same_q = sum(
        records[index]["q29_sha256"] == records[index + 1]["q29_sha256"]
        for index in duplicate_tick_indices
    )
    ages_ms = [
        (int(record["application_unix_ns"]) - int(record["source_timestamp_ns"]))
        / 1_000_000.0
        for record in records
        if int(record["source_timestamp_ns"]) > 0
    ]
    plausible_ages_ms = [age for age in ages_ms if -86_400_000.0 <= age <= 86_400_000.0]
    batch_sizes = [int(batch["batch_size"]) for batch in batches]
    wait_ms = [float(batch["wait_duration_ms"]) for batch in batches]
    take_ms = [float(batch["take_duration_ms"]) for batch in batches]
    positive_tick_steps = [step for step in tick_steps if step > 0]
    return {
        "sample_count": len(records),
        "application_gap_ms": _timing(application_gaps_ms),
        "source_timestamp": {
            "available_count": len(source_ns),
            "unique_count": len(set(source_ns)),
            "duplicate_transition_count": sum(step == 0 for step in source_steps),
            "decreasing_transition_count": sum(step < 0 for step in source_steps),
            "positive_gap_ms": _timing(source_positive_gaps_ms),
            "clock_age_ms": _timing(plausible_ages_ms),
            "clock_age_plausible_count": len(plausible_ages_ms),
        },
        "sample_identity": {
            "publication_handle_count": len({item[0] for item in identity}),
            "publication_handle_source_timestamp_unique_count": len(set(identity)),
            "identity_available_count": len(identity),
            "publication_handle_source_timestamp_duplicate_count": (
                len(identity) - len(set(identity))
            ),
            "identity_pair_is_unique": (
                len(identity) == len(records) and len(identity) == len(set(identity))
            ),
        },
        "tick": {
            "unique_count": len(set(ticks)),
            "duplicate_transition_count": sum(step == 0 for step in tick_steps),
            "decreasing_transition_count": sum(step < 0 for step in tick_steps),
            "positive_increment_minimum": min(positive_tick_steps, default=None),
            "positive_increment_maximum": max(positive_tick_steps, default=None),
            "duplicate_tick_same_q29_count": duplicate_tick_same_q,
            "duplicate_tick_changed_q29_count": (
                len(duplicate_tick_indices) - duplicate_tick_same_q
            ),
            "tick_is_unique_sample_identity": len(ticks) == len(set(ticks)),
        },
        "reader_batches": {
            "count": len(batches),
            "maximum_batch_size": max(batch_sizes, default=0),
            "multi_sample_batch_count": sum(size > 1 for size in batch_sizes),
            "samples_in_multi_sample_batches": sum(size for size in batch_sizes if size > 1),
            "wait_duration_ms": _timing(wait_ms),
            "take_duration_ms": _timing(take_ms),
        },
        "dds_status": dict(status_events),
    }


def _record_from_sample(
    sample: Any,
    *,
    sample_index: int,
    batch_index: int,
    batch_offset: int,
    batch_size: int,
    application_unix_ns: int,
    application_monotonic_ns: int,
) -> dict[str, Any]:
    snapshot = extract_lowstate_snapshot(
        sample,
        received_unix_ns=application_unix_ns,
        received_monotonic_ns=application_monotonic_ns,
    )
    q29 = [motor.q for motor in snapshot.full_body_motor_state]
    dq29 = [motor.dq for motor in snapshot.full_body_motor_state]
    if len(q29) != len(FULL_BODY_INDICES):
        raise ValueError("direct DDS sample did not contain 29 motor positions")
    q_payload = json.dumps(q29, separators=(",", ":"), allow_nan=False).encode()
    state_payload = json.dumps(
        {
            "tick": snapshot.tick,
            "mode_pr": snapshot.mode_pr,
            "mode_machine": snapshot.mode_machine,
            "q29": q29,
            "dq29": dq29,
            "imu_quaternion": snapshot.imu_quaternion,
        },
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()
    info = sample.sample_info
    return {
        "sample_index": sample_index,
        "batch_index": batch_index,
        "batch_offset": batch_offset,
        "batch_size": batch_size,
        "application_unix_ns": application_unix_ns,
        "application_monotonic_ns": application_monotonic_ns,
        "tick": snapshot.tick,
        "mode_pr": snapshot.mode_pr,
        "mode_machine": snapshot.mode_machine,
        "q29": q29,
        "dq29": dq29,
        "q29_sha256": hashlib.sha256(q_payload).hexdigest(),
        "payload_sha256": hashlib.sha256(state_payload).hexdigest(),
        "source_timestamp_ns": int(info.source_timestamp),
        "publication_handle": int(info.publication_handle),
        "instance_handle": int(info.instance_handle),
        "sample_state": int(info.sample_state),
        "view_state": int(info.view_state),
        "instance_state": int(info.instance_state),
        "valid_data": bool(info.valid_data),
        "sample_rank": int(info.sample_rank),
        "generation_rank": int(info.generation_rank),
        "absolute_generation_rank": int(info.absolute_generation_rank),
    }


def profile_direct_reader(
    network_interface: str,
    *,
    sample_count: int,
    timeout_s: float,
    maximum_batch_size: int = 64,
    condition_label: str = "unspecified",
) -> dict[str, Any]:
    network_interface = validate_network_interface(network_interface)
    if not 20 <= sample_count <= 3000:
        raise ValueError("sample_count must be in [20,3000]")
    if not 1.0 <= timeout_s <= 120.0:
        raise ValueError("timeout_s must be in [1,120]")
    if not 1 <= maximum_batch_size <= 256:
        raise ValueError("maximum_batch_size must be in [1,256]")
    if not _LABEL_PATTERN.fullmatch(condition_label):
        raise ValueError("condition_label contains unsupported characters")

    # Lazy imports keep offline tests independent of the robot DDS runtime.
    from cyclonedds.core import (
        InstanceState,
        Listener,
        ReadCondition,
        SampleState,
        ViewState,
        WaitSet,
    )
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from cyclonedds.util import duration
    from unitree_sdk2py.core.channel_config import ChannelConfigHasInterface
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    status_events = {
        "sample_lost_total_count": 0,
        "sample_rejected_total_count": 0,
        "subscription_matched_total_count": 0,
        "subscription_current_count": 0,
    }

    def on_sample_lost(_reader: Any, status: Any) -> None:
        status_events["sample_lost_total_count"] = int(status.total_count)

    def on_sample_rejected(_reader: Any, status: Any) -> None:
        status_events["sample_rejected_total_count"] = int(status.total_count)

    def on_subscription_matched(_reader: Any, status: Any) -> None:
        status_events["subscription_matched_total_count"] = int(status.total_count)
        status_events["subscription_current_count"] = int(status.current_count)

    listener = Listener(
        on_sample_lost=on_sample_lost,
        on_sample_rejected=on_sample_rejected,
        on_subscription_matched=on_subscription_matched,
    )
    domain_config = ChannelConfigHasInterface.replace(
        "$__IF_NAME__$", network_interface
    )
    network_before = _network_metrics(network_interface)
    process_before = _process_metrics()
    records: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    errors: list[str] = []
    waitset = condition = reader = topic = participant = domain = None
    started_unix_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    try:
        domain = Domain(0, domain_config)
        participant = DomainParticipant(0)
        topic = Topic(participant, LOWSTATE_TOPIC, LowState_)
        reader = DataReader(participant, topic, listener=listener)
        condition = ReadCondition(
            reader, ViewState.Any | InstanceState.Alive | SampleState.NotRead
        )
        waitset = WaitSet(participant)
        waitset.attach(condition)
        deadline = time.monotonic() + timeout_s
        while len(records) < sample_count:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break
            wait_started_ns = time.monotonic_ns()
            triggered = waitset.wait(
                duration(milliseconds=min(100.0, remaining_s * 1000.0))
            )
            wait_ended_ns = time.monotonic_ns()
            if triggered == 0:
                continue
            take_started_ns = time.monotonic_ns()
            samples = reader.take(
                min(maximum_batch_size, sample_count - len(records)),
                condition=condition,
            )
            take_ended_ns = time.monotonic_ns()
            if not samples:
                continue
            batch_index = len(batches)
            batch_size = len(samples)
            batches.append({
                "batch_index": batch_index,
                "wait_duration_ms": (wait_ended_ns - wait_started_ns) / 1_000_000.0,
                "take_duration_ms": (take_ended_ns - take_started_ns) / 1_000_000.0,
                "batch_size": batch_size,
                "take_ended_monotonic_ns": take_ended_ns,
            })
            for batch_offset, sample in enumerate(samples):
                records.append(_record_from_sample(
                    sample,
                    sample_index=len(records),
                    batch_index=batch_index,
                    batch_offset=batch_offset,
                    batch_size=batch_size,
                    application_unix_ns=time.time_ns(),
                    application_monotonic_ns=time.monotonic_ns(),
                ))
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if waitset is not None and condition is not None:
            try:
                waitset.detach(condition)
            except Exception as exc:
                errors.append(f"cleanup_{type(exc).__name__}: {exc}")
        reader = None
        condition = None
        waitset = None
        topic = None
        participant = None
        domain = None

    ended_monotonic_ns = time.monotonic_ns()
    network_after = _network_metrics(network_interface)
    process_after = _process_metrics()
    summary = summarize_profile(records, batches, status_events)
    success = len(records) == sample_count and not errors
    return {
        "schema_version": "g1_dds_direct_readonly_profiler_v1",
        "success": success,
        "source": {
            "unitree_sdk_commit": OFFICIAL_SDK_COMMIT,
            "cyclonedds_version": CYCLONEDDS_VERSION,
            "cyclonedds_sdist_sha256": CYCLONEDDS_SDIST_SHA256,
            "topic": LOWSTATE_TOPIC,
            "network_interface": network_interface,
            "reader_path": "direct_datareader_waitset_no_sdk_callback_queue",
        },
        "requested": {
            "sample_count": sample_count,
            "timeout_s": timeout_s,
            "maximum_batch_size": maximum_batch_size,
            "condition_label": condition_label,
        },
        "observed": {
            "captured_samples": len(records),
            "started_unix_ns": started_unix_ns,
            "elapsed_s": (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000.0,
            "errors": errors,
            "summary": summary,
            "process_before": process_before,
            "process_after": process_after,
            "process_delta": _metric_delta(process_before, process_after),
            "network_before": network_before,
            "network_after": network_after,
            "network_delta": _metric_delta(network_before, network_after),
        },
        "records": records,
        "batches": batches,
        "safety": {
            "subscriber_only": True,
            "direct_reader_only": True,
            "sdk_callback_queue_bypassed": True,
            "send_channel_created": False,
            "robot_command_publisher_created": False,
            "mode_change_requested": False,
            "robot_command_sent": False,
            "hardware_motion_performed": False,
        },
        "decision": {
            "profile_completed": success,
            "dds_freshness_fully_qualified": False,
            "real_policy_shadow_allowed": False,
            "robot_motion_allowed": False,
        },
        "g1_execution_enabled": False,
        "hardware_execution_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--maximum-batch-size", type=int, default=64)
    parser.add_argument("--condition-label", default="unspecified")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = profile_direct_reader(
        args.network_interface,
        sample_count=args.samples,
        timeout_s=args.timeout_s,
        maximum_batch_size=args.maximum_batch_size,
        condition_label=args.condition_label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
