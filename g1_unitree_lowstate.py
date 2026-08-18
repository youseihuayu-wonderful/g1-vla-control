#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subscriber-only Unitree G1 LowState capture.

This module deliberately contains no robot command type, command channel, mode
switcher, or client API. Importing it does not initialize DDS. The official SDK
is imported lazily only when ``capture_lowstate`` is called.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
import re
import statistics
import time
from typing import Any


OFFICIAL_SDK_REPOSITORY = "https://github.com/unitreerobotics/unitree_sdk2_python.git"
OFFICIAL_SDK_COMMIT = "65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5"
LOWSTATE_TOPIC = "rt/lowstate"
G1_MOTOR_COUNT = 29

G1_29DOF_JOINT_INDEX = {
    "left_hip_pitch_joint": 0,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 2,
    "left_knee_joint": 3,
    "left_ankle_pitch_joint": 4,
    "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6,
    "right_hip_roll_joint": 7,
    "right_hip_yaw_joint": 8,
    "right_knee_joint": 9,
    "right_ankle_pitch_joint": 10,
    "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12,
    "waist_roll_joint": 13,
    "waist_pitch_joint": 14,
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19,
    "left_wrist_pitch_joint": 20,
    "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26,
    "right_wrist_pitch_joint": 27,
    "right_wrist_yaw_joint": 28,
}

CONTRACT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
CONTRACT_ARM_INDICES = tuple(G1_29DOF_JOINT_INDEX[name] for name in CONTRACT_ARM_JOINTS)
_INTERFACE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


@dataclass(frozen=True)
class ArmMotorState:
    name: str
    index: int
    mode: int
    q: float
    dq: float
    ddq: float
    tau_est: float
    temperature: tuple[int, int]
    vol: float
    motor_state: int


@dataclass(frozen=True)
class LowStateSnapshot:
    received_unix_ns: int
    received_monotonic_ns: int
    version: tuple[int, int]
    mode_pr: int
    mode_machine: int
    tick: int
    imu_quaternion: tuple[float, float, float, float]
    imu_gyroscope: tuple[float, float, float]
    imu_accelerometer: tuple[float, float, float]
    imu_rpy: tuple[float, float, float]
    imu_temperature: int
    arm_motor_state: tuple[ArmMotorState, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _float_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    result = tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))
    if len(result) != length:
        raise ValueError(f"{label} must contain {length} values, got {len(result)}")
    return result


def _int_vector(value: Any, length: int, label: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value)
    if len(result) != length:
        raise ValueError(f"{label} must contain {length} values, got {len(result)}")
    return result


def validate_network_interface(name: str) -> str:
    if not _INTERFACE_PATTERN.fullmatch(name):
        raise ValueError(f"invalid network interface: {name!r}")
    return name


def extract_lowstate_snapshot(
    message: Any,
    *,
    received_unix_ns: int | None = None,
    received_monotonic_ns: int | None = None,
) -> LowStateSnapshot:
    """Validate and copy the read-only fields needed by the frozen G1 contract."""

    motor_state = tuple(message.motor_state)
    if len(motor_state) < G1_MOTOR_COUNT:
        raise ValueError(
            f"G1 LowState requires at least {G1_MOTOR_COUNT} motors, got {len(motor_state)}"
        )

    arm = []
    for name, index in zip(CONTRACT_ARM_JOINTS, CONTRACT_ARM_INDICES):
        state = motor_state[index]
        arm.append(
            ArmMotorState(
                name=name,
                index=index,
                mode=int(state.mode),
                q=_finite(state.q, f"motor_state[{index}].q"),
                dq=_finite(state.dq, f"motor_state[{index}].dq"),
                ddq=_finite(state.ddq, f"motor_state[{index}].ddq"),
                tau_est=_finite(state.tau_est, f"motor_state[{index}].tau_est"),
                temperature=_int_vector(
                    state.temperature, 2, f"motor_state[{index}].temperature"
                ),
                vol=_finite(state.vol, f"motor_state[{index}].vol"),
                motor_state=int(state.motorstate),
            )
        )

    imu = message.imu_state
    return LowStateSnapshot(
        received_unix_ns=time.time_ns() if received_unix_ns is None else int(received_unix_ns),
        received_monotonic_ns=(
            time.monotonic_ns()
            if received_monotonic_ns is None
            else int(received_monotonic_ns)
        ),
        version=_int_vector(message.version, 2, "version"),
        mode_pr=int(message.mode_pr),
        mode_machine=int(message.mode_machine),
        tick=int(message.tick),
        imu_quaternion=_float_vector(imu.quaternion, 4, "imu.quaternion"),
        imu_gyroscope=_float_vector(imu.gyroscope, 3, "imu.gyroscope"),
        imu_accelerometer=_float_vector(imu.accelerometer, 3, "imu.accelerometer"),
        imu_rpy=_float_vector(imu.rpy, 3, "imu.rpy"),
        imu_temperature=int(imu.temperature),
        arm_motor_state=tuple(arm),
    )


def _timing_summary(snapshots: list[LowStateSnapshot]) -> dict[str, float | int | None]:
    gaps_ms = [
        (right.received_monotonic_ns - left.received_monotonic_ns) / 1_000_000.0
        for left, right in zip(snapshots, snapshots[1:])
    ]
    if not gaps_ms:
        return {"gap_count": 0, "gap_min_ms": None, "gap_median_ms": None, "gap_max_ms": None}
    return {
        "gap_count": len(gaps_ms),
        "gap_min_ms": min(gaps_ms),
        "gap_median_ms": statistics.median(gaps_ms),
        "gap_max_ms": max(gaps_ms),
    }


def capture_lowstate(
    network_interface: str,
    *,
    sample_count: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Capture official G1 LowState samples without constructing a send channel."""

    network_interface = validate_network_interface(network_interface)
    if not 1 <= sample_count <= 3000:
        raise ValueError("sample_count must be between 1 and 3000")
    if not 1.0 <= timeout_s <= 120.0:
        raise ValueError("timeout_s must be between 1 and 120 seconds")

    # Lazy import keeps local validation independent of CycloneDDS. Do not expand
    # this import list without a separate hardware-safety review.
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    queue: Queue[tuple[Any, int, int]] = Queue(maxsize=min(sample_count * 2, 6000))
    callback_overflow = 0

    def handler(message: Any) -> None:
        nonlocal callback_overflow
        try:
            queue.put_nowait((message, time.time_ns(), time.monotonic_ns()))
        except Full:
            callback_overflow += 1

    ChannelFactoryInitialize(0, network_interface)
    subscriber = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
    subscriber.Init(handler, 10)
    snapshots: list[LowStateSnapshot] = []
    deadline = time.monotonic() + timeout_s
    try:
        while len(snapshots) < sample_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                message, unix_ns, monotonic_ns = queue.get(timeout=min(remaining, 0.5))
            except Empty:
                continue
            snapshots.append(
                extract_lowstate_snapshot(
                    message,
                    received_unix_ns=unix_ns,
                    received_monotonic_ns=monotonic_ns,
                )
            )
    finally:
        subscriber.Close()

    success = len(snapshots) == sample_count
    return {
        "schema_version": "g1_unitree_lowstate_readonly_v1",
        "success": success,
        "source": {
            "repository": OFFICIAL_SDK_REPOSITORY,
            "commit": OFFICIAL_SDK_COMMIT,
            "idl": "unitree_hg.msg.dds_.LowState_",
            "topic": LOWSTATE_TOPIC,
            "network_interface": network_interface,
            "field_names_preserved_from_official_idl": True,
            "units_and_imu_quaternion_order_verified_for_this_robot": False,
        },
        "safety": {
            "subscriber_only": True,
            "send_channel_created": False,
            "robot_command_sent": False,
            "mode_change_requested": False,
        },
        "requested_samples": sample_count,
        "captured_samples": len(snapshots),
        "callback_overflow": callback_overflow,
        "timing": _timing_summary(snapshots),
        "first_tick": snapshots[0].tick if snapshots else None,
        "last_tick": snapshots[-1].tick if snapshots else None,
        "samples": [snapshot.to_dict() for snapshot in snapshots],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = capture_lowstate(
        args.network_interface,
        sample_count=args.samples,
        timeout_s=args.timeout_s,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if str(args.output) == "-":
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(args.output)
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
