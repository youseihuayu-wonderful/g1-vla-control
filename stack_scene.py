"""Programmatic G1 block-stacking scene with VLA observation cameras."""

from __future__ import annotations

from pathlib import Path
import os
import mujoco
import numpy as np

from dex1_gripper import (
    Dex1Controller, JAW_MIN_M, motor_radians_to_jaw_position,
    replace_hands_with_dex1,
)

ROOT = Path(__file__).resolve().parent
_BUNDLED_MENAGERIE = ROOT / "third_party" / "mujoco_menagerie"
MENAGERIE_ROOT = Path(os.environ.get(
    "MUJOCO_MENAGERIE_PATH",
    _BUNDLED_MENAGERIE if _BUNDLED_MENAGERIE.exists()
    else Path.home() / "mujoco_menagerie",
))
MODEL_PATH = MENAGERIE_ROOT / "unitree_g1" / "scene.xml"
CAMERA_NAMES = ("cam_left_high", "cam_left_wrist", "cam_right_wrist")
TASK_PROMPT = (
    "Stack the blocks by color: put the red block in the center, then stack "
    "the blue block on the red block, then stack the yellow block on the blue block."
)
REFERENCE_EP0_STATE = np.array([
    -0.11507253, -0.02362091, -0.05944176, 0.22499184,
    0.01432115, -0.24764203, -0.05209543,
    -0.02505901, 0.06801048, 0.18285531, 0.07339139,
    -0.06506236, -0.25428128, -0.09432784,
    5.36817312, 5.38309956,
])


def _joint_width(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def _capture_and_remove_stand_key(spec: mujoco.MjSpec) -> dict:
    """Capture stand values by name so tree edits cannot shift keyframe indices."""
    model = spec.compile()
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    joints = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        width = _joint_width(model.jnt_type[joint_id])
        address = model.jnt_qposadr[joint_id]
        joints[name] = data.qpos[address : address + width].copy()
    controls = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id):
        float(data.ctrl[actuator_id])
        for actuator_id in range(model.nu)
    }
    for key in list(spec.keys):
        if key.name == "stand":
            spec.delete(key)
    return {"joints": joints, "controls": controls}


def _rebuild_stand_key(spec: mujoco.MjSpec, captured: dict) -> None:
    model = spec.compile()
    qpos = model.qpos0.copy()
    ctrl = np.zeros(model.nu)
    for name, value in captured["joints"].items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        address = model.jnt_qposadr[joint_id]
        qpos[address : address + len(value)] = value
    for name, value in captured["controls"].items():
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        if actuator_id >= 0:
            ctrl[actuator_id] = value
    for side in ("left", "right"):
        for finger in ("Joint1_1", "Joint2_1"):
            joint_name = f"{side}_dex1_{finger}"
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            qpos[model.jnt_qposadr[joint_id]] = JAW_MIN_M
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"{joint_name}_actuator",
            )
            ctrl[actuator_id] = JAW_MIN_M
    spec.add_key(name="stand", qpos=qpos, ctrl=ctrl)


def _add_cube(spec: mujoco.MjSpec, name: str, position, color) -> None:
    body = spec.worldbody.add_body(name=name, pos=position)
    body.add_freejoint(name=f"{name}_free")
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.04, 0.04, 0.04],
        mass=0.16,
        friction=[0.9, 0.02, 0.002],
        solref=[0.01, 1.0],
        rgba=color,
    )


def _world_camera_xyaxes(camera_world, target_world) -> list[float]:
    look = np.asarray(target_world) - np.asarray(camera_world)
    look /= np.linalg.norm(look)
    camera_x = np.cross(look, np.array([0.0, 0.0, 1.0]))
    camera_x /= np.linalg.norm(camera_x)
    camera_z = -look
    camera_y = np.cross(camera_z, camera_x)
    return [*camera_x, *camera_y]


def _camera_xyaxes(
    spec: mujoco.MjSpec, body_name: str, local_position, target_world
) -> list[float]:
    """Compute fixed camera axes in an assembled body's local frame."""
    assembled_model = spec.compile()
    assembled_data = mujoco.MjData(assembled_model)
    reset_to_stand(assembled_model, assembled_data)
    body_id = mujoco.mj_name2id(
        assembled_model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    parent_rotation = assembled_data.xmat[body_id].reshape(3, 3)
    camera_world = (
        assembled_data.xpos[body_id] + parent_rotation @ np.asarray(local_position)
    )
    look = np.asarray(target_world) - camera_world
    look /= np.linalg.norm(look)
    camera_x = np.cross(look, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(camera_x) < 1e-6:
        camera_x = np.array([0.0, -1.0, 0.0])
    camera_x /= np.linalg.norm(camera_x)
    camera_z = -look
    camera_y = np.cross(camera_z, camera_x)
    world_rotation = np.column_stack((camera_x, camera_y, camera_z))
    local_rotation = parent_rotation.T @ world_rotation
    return [*local_rotation[:, 0], *local_rotation[:, 1]]


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    captured_stand = _capture_and_remove_stand_key(spec)
    replace_hands_with_dex1(spec)
    # Public checkpoint statistics are best explained by a grasp point 50 mm
    # along wrist +X while retaining the wrist-yaw orientation.
    for side in ("left", "right"):
        spec.body(f"{side}_wrist_yaw_link").add_site(
            name=f"{side}_eef",
            pos=[0.05, 0.0, 0.0],
            size=[0.006],
            rgba=[0.1, 0.9, 0.9, 0.35],
        )

    # White tabletop similar to the reference Stack-the-cubes episode.
    spec.worldbody.add_geom(
        name="work_table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.68, 0.0, 0.735],
        size=[0.45, 0.58, 0.055],
        friction=[1.0, 0.02, 0.002],
        rgba=[0.92, 0.93, 0.95, 1.0],
    )
    # A target body gives tracking cameras a stable look-at point.
    spec.worldbody.add_body(name="workspace_target", pos=[0.46, 0.0, 0.84])

    table_surface = 0.79
    _add_cube(spec, "red_cube", [0.46, 0.00, table_surface + 0.041], [0.92, 0.12, 0.12, 1])
    _add_cube(spec, "blue_cube", [0.43, 0.18, table_surface + 0.041], [0.08, 0.35, 0.95, 1])
    _add_cube(spec, "yellow_cube", [0.43, -0.18, table_surface + 0.041], [1.0, 0.62, 0.05, 1])

    # Rebuild by joint/actuator name after inserting Dex1 joints into the tree.
    _rebuild_stand_key(spec, captured_stand)

    camera_target = [0.46, 0.0, 0.84]
    high_pos = [0.08, 0.0, 1.38]
    # Reference frames show each optical center between the two fingers. Dex1
    # child +Y is forward, so x=+X/y=+Z gives camera optical -Z along +Y.
    wrist_pos = [0.0, 0.070, 0.0]
    wrist_pitch = np.deg2rad(30.0)
    wrist_xyaxes = [
        1.0, 0.0, 0.0,
        0.0, float(np.sin(wrist_pitch)), float(np.cos(wrist_pitch)),
    ]
    left_gripper = spec.body("left_dex1_base_link")
    right_gripper = spec.body("right_dex1_base_link")
    if left_gripper is None or right_gripper is None:
        raise RuntimeError("Expected Dex1 camera mounting bodies are missing")

    spec.worldbody.add_camera(
        name="cam_left_high",
        pos=high_pos,
        xyaxes=_world_camera_xyaxes(high_pos, camera_target),
        fovy=58,
        resolution=[640, 480],
    )
    left_gripper.add_camera(
        name="cam_left_wrist",
        pos=wrist_pos,
        xyaxes=wrist_xyaxes,
        fovy=90,
        resolution=[640, 480],
    )
    right_gripper.add_camera(
        name="cam_right_wrist",
        pos=wrist_pos,
        xyaxes=wrist_xyaxes,
        fovy=90,
        resolution=[640, 480],
    )
    return spec.compile()


def reset_to_stand(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    # Preserve qpos0 for free cube joints. Keyframes inherited from the robot
    # predate those joints, so MuJoCo extends their values with zeros.
    mujoco.mj_resetData(model, data)
    first_cube_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_free"
    )
    robot_nq = int(model.jnt_qposadr[first_cube_joint])
    data.qpos[:robot_nq] = model.key_qpos[key, :robot_nq]
    data.ctrl[:] = model.key_ctrl[key]
    Dex1Controller(model).set_open(data)
    mujoco.mj_forward(model, data)


def translate_cubes(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    translation_m: np.ndarray,
) -> None:
    """Translate all three free cube poses for explicit scene randomization."""
    translation = np.asarray(translation_m, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("cube translation must contain three finite values")
    for color in ("red", "blue", "yellow"):
        joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{color}_cube_free"
        )
        if joint < 0:
            raise ValueError(f"Missing {color} cube free joint")
        qpos = int(model.jnt_qposadr[joint])
        data.qpos[qpos:qpos + 3] += translation
    mujoco.mj_forward(model, data)


def reset_to_reference_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset to the synchronized first state of public dataset episode 0."""
    reset_to_stand(model, data)
    arm_names = (
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw",
        )
    )
    for name, value in zip(arm_names, REFERENCE_EP0_STATE[:14], strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
        data.ctrl[actuator_id] = value
    Dex1Controller(model).set_motor_commands(
        data, REFERENCE_EP0_STATE[14], REFERENCE_EP0_STATE[15]
    )
    # Put jaw qpos at the mapped target too, avoiding a transient in frame 0.
    for side, command in zip(("left", "right"), REFERENCE_EP0_STATE[14:], strict=True):
        target = motor_radians_to_jaw_position(float(command))
        for finger in ("Joint1_1", "Joint2_1"):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_dex1_{finger}"
            )
            data.qpos[model.jnt_qposadr[joint_id]] = target
    mujoco.mj_forward(model, data)
