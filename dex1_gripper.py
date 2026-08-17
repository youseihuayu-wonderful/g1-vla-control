"""Official Unitree Dex1-1 model integration and VLA command mapping."""

from __future__ import annotations

from pathlib import Path
import math
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
DEX1_URDF = ROOT / "third_party" / "dex1_1_service" / "urdf" / "dex1_1.urdf"
MOTOR_MIN_RAD = 0.0
MOTOR_MAX_RAD = 5.5
JAW_MIN_M = -0.020
JAW_MAX_M = 0.0245


def motor_radians_to_jaw_position(command: float) -> float:
    """Map the real Dex1 motor convention to the URDF prismatic joint.

    Unitree's example drives the calibrated motor through approximately
    0..5.5 rad. Synchronized dataset frame 0 shows ~5.38 rad while visibly
    open. The URDF prismatic sign is opposite, so this mapping is reversed.
    """
    fraction = np.clip(
        (float(command) - MOTOR_MIN_RAD) / (MOTOR_MAX_RAD - MOTOR_MIN_RAD),
        0.0,
        1.0,
    )
    return float(JAW_MAX_M - fraction * (JAW_MAX_M - JAW_MIN_M))


def jaw_position_to_motor_radians(position: float) -> float:
    """Inverse of :func:`motor_radians_to_jaw_position`."""
    fraction = np.clip(
        (JAW_MAX_M - float(position)) / (JAW_MAX_M - JAW_MIN_M), 0.0, 1.0
    )
    return float(MOTOR_MIN_RAD + fraction * (MOTOR_MAX_RAD - MOTOR_MIN_RAD))


def _remove_articulated_hands(spec: mujoco.MjSpec) -> None:
    # Delete actuators first so no transmission refers to deleted hand joints.
    for actuator in list(spec.actuators):
        if "_hand_" in actuator.name:
            spec.delete(actuator)

    hand_roots = (
        "left_hand_thumb_0_link", "left_hand_middle_0_link", "left_hand_index_0_link",
        "right_hand_thumb_0_link", "right_hand_middle_0_link", "right_hand_index_0_link",
    )
    for name in hand_roots:
        body = spec.body(name)
        if body is not None:
            spec.delete(body)

    # Palm meshes are direct geoms on wrist_yaw, not child bodies.
    for side in ("left", "right"):
        wrist = spec.body(f"{side}_wrist_yaw_link")
        for geom in list(wrist.geoms):
            if "hand_palm" in geom.meshname or "rubber_hand" in geom.meshname:
                spec.delete(geom)


def _attach_one(spec: mujoco.MjSpec, side: str) -> None:
    wrist = spec.body(f"{side}_wrist_yaw_link")
    if wrist is None:
        raise RuntimeError(f"Missing {side} wrist body")

    # Dex1's finger direction is +Y in its URDF. Rotate it onto the G1 wrist's
    # forward +X axis and mount just beyond the wrist-yaw housing.
    # The same non-handed parallel gripper is used on both wrists. In both
    # cases child +Y must point along wrist +X; mirroring this yaw would make
    # the right gripper point backward.
    yaw = -math.pi / 2
    mount = wrist.add_frame(
        name=f"{side}_dex1_mount",
        pos=[0.065, 0.0, 0.0],
        euler=[0.0, 0.0, yaw],
    )
    child = mujoco.MjSpec.from_file(str(DEX1_URDF))
    # MjSpec versions differ on whether mesh.file retains compiler.meshdir.
    # Resolve every mesh before clearing meshdir so attached specs are portable.
    mesh_directory = Path(child.compiler.meshdir)
    for mesh in child.meshes:
        source = Path(mesh.file)
        if not source.is_absolute():
            with_meshdir = DEX1_URDF.parent / mesh_directory / source
            without_meshdir = DEX1_URDF.parent / source
            source = (
                with_meshdir if with_meshdir.exists() else without_meshdir
            )
        if not source.is_file():
            raise FileNotFoundError(f"Dex1 mesh not found: {source}")
        mesh.file = str(source.resolve())
    child.compiler.meshdir = ""
    spec.attach(child, prefix=f"{side}_dex1_", frame=mount)

    for finger in ("Joint1_1", "Joint2_1"):
        joint_name = f"{side}_dex1_{finger}"
        actuator = spec.add_actuator(
            name=f"{side}_dex1_{finger}_actuator",
            target=joint_name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrllimited=1,
            ctrlrange=[JAW_MIN_M, JAW_MAX_M],
            forcelimited=1,
            forcerange=[-20.0, 20.0],
        )
        actuator.set_to_position(kp=450.0, kv=12.0)


def replace_hands_with_dex1(spec: mujoco.MjSpec) -> None:
    if not DEX1_URDF.exists():
        raise FileNotFoundError(f"Official Dex1-1 URDF not found: {DEX1_URDF}")
    _remove_articulated_hands(spec)
    _attach_one(spec, "left")
    _attach_one(spec, "right")


class Dex1Controller:
    """Maps the two VLA gripper channels to four symmetric finger actuators."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.actuators: dict[str, np.ndarray] = {}
        self.qpos: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            actuator_ids = []
            qpos_ids = []
            for finger in ("Joint1_1", "Joint2_1"):
                joint_name = f"{side}_dex1_{finger}"
                actuator_name = f"{joint_name}_actuator"
                actuator_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
                )
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                if actuator_id < 0 or joint_id < 0:
                    raise RuntimeError(f"Missing Dex1 actuator/joint: {joint_name}")
                actuator_ids.append(actuator_id)
                qpos_ids.append(model.jnt_qposadr[joint_id])
            self.actuators[side] = np.asarray(actuator_ids, dtype=int)
            self.qpos[side] = np.asarray(qpos_ids, dtype=int)

    def set_motor_commands(
        self, data: mujoco.MjData, left_command: float, right_command: float
    ) -> None:
        data.ctrl[self.actuators["left"]] = motor_radians_to_jaw_position(left_command)
        data.ctrl[self.actuators["right"]] = motor_radians_to_jaw_position(right_command)

    def motor_states(self, data: mujoco.MjData) -> np.ndarray:
        """Return measured left/right motor-equivalent state from jaw qpos."""
        return np.asarray([
            jaw_position_to_motor_radians(float(np.mean(data.qpos[self.qpos[side]])))
            for side in ("left", "right")
        ])

    def set_open(self, data: mujoco.MjData) -> None:
        self.set_motor_commands(data, MOTOR_MAX_RAD, MOTOR_MAX_RAD)

    def set_closed(self, data: mujoco.MjData) -> None:
        self.set_motor_commands(data, MOTOR_MIN_RAD, MOTOR_MIN_RAD)
