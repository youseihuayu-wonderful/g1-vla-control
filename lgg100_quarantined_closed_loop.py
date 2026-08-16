#!/usr/bin/env python3
"""Fail-closed, quarantined real-LGG100 receding-horizon MuJoCo study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import socket
import time
from typing import Any

import mujoco
import numpy as np

from action_schema import EEFActionChunk, pelvis_vla_action_to_world_mujoco
from adaptive_speed_context import ContextAwareRetimer, decide_speed
from dex1_gripper import Dex1Controller
from experimental_sim_observation import (
    apply_experimental_camera_calibration,
    render_experimental_policy_observation,
)
from g1_policy_contract import (
    ACTION_HORIZON,
    CONTRACT_ID,
    CONTRACT_SHA256,
    POLICY_RATE_HZ,
)
from g1_sim_speed_context import build_simulation_speed_context
from lgg100_candidate_server import HF_REPO, HF_REVISION
from local_websocket_policy_client import LocalWebsocketPolicyClient
from neural_action_audit import audit_neural_action_chunk
from quarantined_chunk_executor import QuarantinedChunkExecutor
from safety_governor import G1TargetPreflight
from stack_scene import build_model, reset_to_reference_pose, translate_cubes
from swept_path_preflight import G1SweptPathPreflight

ROOT = Path(__file__).resolve().parent
_COLLISION_PHASE = {
    "free_space": "free_space", "approach": "free_space",
    "grasp": "grasp", "lift": "grasp", "place": "place",
    "retreat": "free_space",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _infer_with_timeout(client, observation: dict[str, Any], timeout_ms: float):
    def handler(signum, frame):
        del signum, frame
        raise TimeoutError(f"LGG100 inference exceeded {timeout_ms:.0f} ms")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_ms / 1000.0)
    try:
        return client.infer(observation)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _cube_state(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    values = {}
    for color in ("red", "blue", "yellow"):
        body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{color}_cube"
        )
        joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{color}_cube_free"
        )
        dof = int(model.jnt_dofadr[joint])
        values[color] = {
            "position_m": data.xpos[body].tolist(),
            "linear_speed_m_s": float(np.linalg.norm(data.qvel[dof:dof + 3])),
        }
    red = np.asarray(values["red"]["position_m"])
    blue = np.asarray(values["blue"]["position_m"])
    yellow = np.asarray(values["yellow"]["position_m"])

    def stacked(upper: np.ndarray, lower: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(upper[:2] - lower[:2]) <= 0.045
            and 0.060 <= upper[2] - lower[2] <= 0.100
        )

    blue_on_red = stacked(blue, red)
    yellow_on_blue = stacked(yellow, blue)
    stationary = all(
        values[color]["linear_speed_m_s"] <= 0.05 for color in values
    )
    values["task_metrics"] = {
        "blue_on_red": blue_on_red,
        "yellow_on_blue": yellow_on_blue,
        "all_cubes_stationary": stationary,
        "full_stack_geometric_success": bool(
            blue_on_red and yellow_on_blue and stationary
        ),
    }
    return values


def _target_preflight(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actions: np.ndarray,
    phases: tuple[str, ...],
) -> dict:
    gate = G1TargetPreflight(model)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    failures = []
    for index, (action, phase) in enumerate(zip(actions, phases, strict=True)):
        world = pelvis_vla_action_to_world_mujoco(
            action, data.xpos[pelvis], data.xquat[pelvis]
        )
        result = gate.check(data, world, phase=_COLLISION_PHASE[phase])
        if not result.accepted:
            failures.append({
                "index": index,
                "phase": phase,
                "reason": result.reason,
                "collision_reasons": list(result.collision_reasons),
                "position_error_m": result.position_error_m,
                "orientation_error_rad": result.orientation_error_rad,
            })
    return {
        "checked": len(actions),
        "accepted": len(actions) - len(failures),
        "all_accepted": not failures,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--camera-calibration", type=Path, required=True)
    parser.add_argument("--semantic-report", type=Path, required=True)
    parser.add_argument("--phase-speed-report", type=Path, required=True)
    parser.add_argument("--cube-x-offset-m", type=float, default=0.08)
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument("--prefix-duration-s", type=float, default=0.10)
    parser.add_argument("--maximum-observation-age-ms", type=float, default=100.0)
    parser.add_argument("--call-timeout-ms", type=float, default=500.0)
    parser.add_argument("--warmup-timeout-ms", type=float, default=60_000.0)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument(
        "--paused-step-synchronous-diagnostic", action="store_true",
        help="allow wall-clock age to exceed the realtime budget only because MuJoCo is paused",
    )
    parser.add_argument("--online-ik-iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks-output", type=Path, required=True)
    parser.add_argument("--allow-quarantined-closed-loop", action="store_true")
    args = parser.parse_args()
    if not args.allow_quarantined_closed_loop:
        raise SystemExit(
            "Closed loop is disabled by default; explicit quarantined MuJoCo acknowledgement required."
        )
    if args.cycles < 1 or not 0.01 <= args.prefix_duration_s <= 0.25:
        raise SystemExit("cycles must be positive and prefix duration must be 0.01..0.25 s")
    if not 10 <= args.online_ik_iterations <= 250:
        raise SystemExit("online IK iterations must be 10..250")

    semantic = json.loads(args.semantic_report.read_text())
    phase_speed = json.loads(args.phase_speed_report.read_text())
    if (
        semantic.get("semantic_identification_supported") is not True
        or semantic.get("g1_policy_contract_id") != CONTRACT_ID
        or semantic.get("g1_policy_contract_sha256") != CONTRACT_SHA256
        or semantic.get("g1_contract_verified") is not False
    ):
        raise ValueError("semantic identification evidence is not suitable")
    phase_binding_matches = bool(
        phase_speed.get("g1_policy_contract_id") == CONTRACT_ID
        and phase_speed.get("g1_policy_contract_sha256") == CONTRACT_SHA256
        and phase_speed.get("action_horizon") == ACTION_HORIZON
        and phase_speed.get("g1_execution_enabled") is False
    )
    required_phase_gate = (
        phase_speed.get("controlled_phase_speed_behavior_passed") is True
        if args.adaptive
        else phase_speed.get("controlled_near_far_speed_behavior_passed") is True
    )
    if not phase_binding_matches or not required_phase_gate:
        requirement = (
            "full near/far plus mixed-transition coverage"
            if args.adaptive else "near/far behavior for the Adaptive-OFF baseline"
        )
        raise ValueError(f"phase-speed evidence does not satisfy {requirement}")

    with socket.create_connection((args.host, args.port), timeout=5.0):
        pass
    client = LocalWebsocketPolicyClient(host=args.host, port=args.port)
    metadata = _json_safe(client.get_server_metadata())
    strict_restore = bool(
        metadata.get("neural_checkpoint_loaded") is True
        and metadata.get("strict_parameter_tree_restore") is True
        and metadata.get("hf_repo") == HF_REPO
        and metadata.get("hf_revision") == HF_REVISION
        and metadata.get("g1_policy_contract_id") == CONTRACT_ID
        and metadata.get("g1_policy_contract_sha256") == CONTRACT_SHA256
        and metadata.get("action_horizon_author_confirmed") is True
        and metadata.get("safe_for_g1_hardware") is False
    )
    if not strict_restore:
        raise ValueError("policy server metadata does not prove strict LGG100 restore")

    calibration = json.loads(args.camera_calibration.read_text())
    model = build_model()
    camera_parameters = apply_experimental_camera_calibration(model, calibration)
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    # Match the exact public episode-0 t=0 state used by observation generation,
    # target preflight, swept preflight, and the persistent dynamics executor.
    cube_translation = np.array([args.cube_x_offset_m, 0.0, 0.0])
    translate_cubes(model, data, cube_translation)
    executor = QuarantinedChunkExecutor(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    swept_gate = G1SweptPathPreflight(
        model, ik_iterations=args.online_ik_iterations
    )
    retimer = ContextAwareRetimer()
    initial_cubes = _cube_state(model, data)
    raw_chunks: list[np.ndarray] = []
    analysis_chunks: list[np.ndarray] = []
    records: list[dict] = []
    abort_reason = None
    stable_success_cycles = 0

    try:
        warmup_observation, _ = render_experimental_policy_observation(
            model, data, renderer, camera_parameters
        )
        _infer_with_timeout(client, warmup_observation, args.warmup_timeout_ms)
        for cycle in range(args.cycles):
            observation_start = time.monotonic()
            observation, source_images = render_experimental_policy_observation(
                model, data, renderer, camera_parameters
            )
            render_done = time.monotonic()
            try:
                response = _infer_with_timeout(
                    client, observation, args.call_timeout_ms
                )
                inference_done = time.monotonic()
                latency_ms = (inference_done - render_done) * 1000.0
                observation_age_after_inference_ms = (
                    inference_done - observation_start
                ) * 1000.0
            except Exception as exc:
                executor.hold()
                abort_reason = f"policy_inference_failed:{type(exc).__name__}"
                records.append({
                    "cycle": cycle,
                    "execution_performed": False,
                    "accepted": False,
                    "reasons": [abort_reason],
                    "error": str(exc),
                })
                break
            actions = np.asarray(response.get("actions"), dtype=np.float64)
            audit = audit_neural_action_chunk(actions)
            if audit.canonicalized_actions_for_analysis is None:
                executor.hold()
                abort_reason = "neural_action_audit_failed"
                records.append({
                    "cycle": cycle,
                    "execution_performed": False,
                    "accepted": False,
                    "reasons": [abort_reason, *audit.reasons],
                    "inference_latency_ms": latency_ms,
                    "observation_age_after_inference_ms": (
                        observation_age_after_inference_ms
                    ),
                })
                break
            analysis = audit.canonicalized_actions_for_analysis
            raw_chunks.append(actions.copy())
            analysis_chunks.append(analysis.copy())
            if (
                observation_age_after_inference_ms
                > args.maximum_observation_age_ms
                and not args.paused_step_synchronous_diagnostic
            ):
                executor.hold()
                abort_reason = "observation_stale_after_inference"
                records.append({
                    "cycle": cycle,
                    "execution_performed": False,
                    "accepted": False,
                    "reasons": [abort_reason],
                    "inference_latency_ms": latency_ms,
                    "observation_age_after_inference_ms": (
                        observation_age_after_inference_ms
                    ),
                    "maximum_observation_age_ms": args.maximum_observation_age_ms,
                })
                break

            measured_grippers = np.asarray(
                observation["observation/state"][14:16], dtype=np.float64
            )
            eef_tracking_error = executor.current_eef_tracking_error_m()
            gripper_tracking_error = float(np.max(np.abs(
                executor.last_filtered_command[14:16] - measured_grippers
            )))
            context_observation_age_ms = (
                0.0 if args.paused_step_synchronous_diagnostic
                else observation_age_after_inference_ms
            )
            contexts = []
            context_evidence = []
            for action in analysis:
                context, evidence = build_simulation_speed_context(
                    model,
                    data,
                    commanded_grippers_rad=action[14:16],
                    measured_grippers_rad=measured_grippers,
                    eef_tracking_error_m=eef_tracking_error,
                    observation_age_ms=context_observation_age_ms,
                    policy_response_age_ms=0.0,
                    preflight_passed=True,
                    collision_free=True,
                    command_limits_passed=True,
                    gripper_tracking_error_rad=gripper_tracking_error,
                )
                contexts.append(context)
                context_evidence.append(evidence)
            phases = tuple(context.task_phase for context in contexts)
            decisions = tuple(decide_speed(context) for context in contexts)
            hold_reasons = [
                f"sample_{index}:{reason}"
                for index, decision in enumerate(decisions)
                if decision.hold
                for reason in decision.reasons
            ]
            timestamps = np.arange(ACTION_HORIZON, dtype=np.float64) / POLICY_RATE_HZ
            nominal_chunk = EEFActionChunk(timestamps, analysis)
            executed_chunk = nominal_chunk
            retiming_record = None
            if args.adaptive:
                retiming = retimer.plan(nominal_chunk, contexts)
                if not retiming.accepted or retiming.chunk is None:
                    executor.hold()
                    abort_reason = "adaptive_retimer_hold"
                    records.append({
                        "cycle": cycle,
                        "execution_performed": False,
                        "accepted": False,
                        "reasons": [abort_reason, *retiming.reasons],
                        "inference_latency_ms": latency_ms,
                        "observation_age_after_inference_ms": (
                            observation_age_after_inference_ms
                        ),
                        "context_observation_age_ms": (
                            context_observation_age_ms
                        ),
                    })
                    break
                executed_chunk = retiming.chunk
                retiming_record = {
                    "scale_range": [
                        float(retiming.scale_profile.min()),
                        float(retiming.scale_profile.max()),
                    ],
                    "duration_s": retiming.metrics["duration_s"],
                    "path_actions_byte_identical": (
                        retiming.path_actions_byte_identical
                    ),
                }
            committed_count = max(2, int(np.searchsorted(
                executed_chunk.timestamps,
                args.prefix_duration_s,
                side="right",
            )))
            committed_count = min(committed_count, ACTION_HORIZON)
            committed_chunk = EEFActionChunk(
                executed_chunk.timestamps[:committed_count],
                executed_chunk.actions[:committed_count],
            )
            committed_phases = phases[:committed_count]
            # Swept preflight includes target reachability/error checks, joint
            # interpolation, and phase-aware collision checks. Running the
            # standalone target gate here would duplicate the expensive IK.
            target_preflight = {
                "covered_by_swept_path_preflight": True,
                "checked": committed_count,
                "all_accepted": None,
            }
            preflight_start = time.monotonic()
            swept = swept_gate.check(
                data, committed_chunk, phase=committed_phases
            )
            preflight_done = time.monotonic()
            preflight_latency_ms = (preflight_done - preflight_start) * 1000.0
            observation_age_at_commit_ms = (
                preflight_done - observation_start
            ) * 1000.0
            stale_at_commit = bool(
                observation_age_at_commit_ms > args.maximum_observation_age_ms
                and not args.paused_step_synchronous_diagnostic
            )
            if hold_reasons or not swept.accepted or stale_at_commit:
                executor.hold()
                abort_reason = (
                    "context_hold" if hold_reasons else
                    "swept_path_preflight_failed" if not swept.accepted
                    else "observation_stale_at_action_commit"
                )
                records.append({
                    "cycle": cycle,
                    "execution_performed": False,
                    "accepted": False,
                    "reasons": [abort_reason, *hold_reasons],
                    "inference_latency_ms": latency_ms,
                    "observation_age_after_inference_ms": (
                        observation_age_after_inference_ms
                    ),
                    "preflight_latency_ms": preflight_latency_ms,
                    "observation_age_at_commit_ms": observation_age_at_commit_ms,
                    "maximum_observation_age_ms": args.maximum_observation_age_ms,
                    "paused_step_synchronous_diagnostic": (
                        args.paused_step_synchronous_diagnostic
                    ),
                    "raw_sha256": audit.raw_sha256,
                    "preflight_scope": "committed_prefix_only",
                    "committed_action_count": committed_count,
                    "target_preflight": target_preflight,
                    "swept_path_preflight": {
                        "accepted": swept.accepted,
                        "reason": swept.reason,
                        "rejection_target_index": swept.rejection_target_index,
                        "rejection_substep": swept.rejection_substep,
                        "collision_reasons": list(swept.collision_reasons),
                    },
                })
                break

            execution = executor.execute_prefix(
                executed_chunk,
                phases,
                prefix_duration_s=args.prefix_duration_s,
            )
            cubes = _cube_state(model, data)
            task_success = cubes["task_metrics"]["full_stack_geometric_success"]
            stable_success_cycles = stable_success_cycles + 1 if task_success else 0
            phase_counts = {
                phase: phases.count(phase) for phase in sorted(set(phases))
            }
            record = {
                "cycle": cycle,
                "execution_performed": True,
                "accepted": execution["accepted"],
                "reasons": execution["reasons"],
                "inference_latency_ms": latency_ms,
                "observation_age_after_inference_ms": (
                    observation_age_after_inference_ms
                ),
                "context_observation_age_ms": context_observation_age_ms,
                "preflight_latency_ms": preflight_latency_ms,
                "observation_age_at_commit_ms": observation_age_at_commit_ms,
                "maximum_observation_age_ms": args.maximum_observation_age_ms,
                "paused_step_synchronous_diagnostic": (
                    args.paused_step_synchronous_diagnostic
                ),
                "server_timing": _json_safe(response.get("server_timing", {})),
                "raw_sha256": audit.raw_sha256,
                "raw_contract_passed": audit.raw_contract_passed,
                "bounded_analysis_available": True,
                "raw_max_quaternion_norm_error": (
                    audit.raw_max_quaternion_norm_error
                ),
                "image_sha256": {
                    camera: hashlib.sha256(image.tobytes()).hexdigest()
                    for camera, image in source_images.items()
                },
                "live_tracking_error_m": eef_tracking_error,
                "live_gripper_tracking_error_rad": gripper_tracking_error,
                "minimum_clearance_m": (
                    context_evidence[0].minimum_dex_cube_clearance_m
                ),
                "phase_counts": phase_counts,
                "speed_target_range": [
                    float(min(decision.target_scale for decision in decisions)),
                    float(max(decision.target_scale for decision in decisions)),
                ],
                "preflight_scope": "committed_prefix_only",
                "committed_action_count": committed_count,
                "target_preflight": target_preflight,
                "swept_path_preflight": {
                    "accepted": swept.accepted,
                    "checked_targets": swept.checked_targets,
                    "checked_interpolated_configurations": (
                        swept.checked_interpolated_configurations
                    ),
                },
                "adaptive_retiming": retiming_record,
                "execution": execution,
                "cubes": cubes,
            }
            records.append(record)
            print(json.dumps({
                "cycle": cycle,
                "inference_latency_ms": latency_ms,
                "observation_age_at_commit_ms": observation_age_at_commit_ms,
                "clearance_m": record["minimum_clearance_m"],
                "phase_counts": phase_counts,
                "execution_accepted": execution["accepted"],
                "task_metrics": cubes["task_metrics"],
            }), flush=True)
            if not execution["accepted"]:
                abort_reason = "dynamic_safety_abort"
                break
            if stable_success_cycles >= 3:
                break
    finally:
        renderer.close()
        client.close()

    final_cubes = _cube_state(model, data)
    task_success = bool(stable_success_cycles >= 3)
    completed_without_abort = abort_reason is None
    raw_array = (
        np.stack(raw_chunks) if raw_chunks
        else np.empty((0, ACTION_HORIZON, 16), dtype=np.float64)
    )
    analysis_array = (
        np.stack(analysis_chunks) if analysis_chunks
        else np.empty((0, ACTION_HORIZON, 16), dtype=np.float64)
    )
    args.chunks_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.chunks_output,
        raw_actions=raw_array,
        canonicalized_actions_for_analysis=analysis_array,
        cube_translation_m=cube_translation,
        adaptive=np.asarray(args.adaptive),
        executable=np.asarray(False),
        quarantined=np.asarray(True),
    )
    report = {
        "scope": "Experimental quarantined real-LGG100 receding-horizon MuJoCo closed loop; never hardware.",
        "checkpoint_revision": HF_REVISION,
        "strict_neural_restore": strict_restore,
        "semantic_report_sha256": _sha256(args.semantic_report),
        "phase_speed_report_sha256": _sha256(args.phase_speed_report),
        "camera_calibration_sha256": _sha256(args.camera_calibration),
        "cube_translation_m": cube_translation.tolist(),
        "adaptive_retiming_enabled": args.adaptive,
        "phase_speed_gate_requirement": (
            "full_near_far_and_mixed_transition"
            if args.adaptive else "near_far_only_for_adaptive_off"
        ),
        "phase_speed_gate_passed": required_phase_gate,
        "paused_step_synchronous_diagnostic": (
            args.paused_step_synchronous_diagnostic
        ),
        "online_ik_iterations": args.online_ik_iterations,
        "prefix_duration_s": args.prefix_duration_s,
        "requested_cycles": args.cycles,
        "completed_cycles": sum(
            record.get("execution_performed") is True for record in records
        ),
        "abort_reason": abort_reason,
        "closed_loop_completed_without_abort": completed_without_abort,
        "initial_cubes": initial_cubes,
        "final_cubes": final_cubes,
        "stable_success_cycles": stable_success_cycles,
        "task_success": task_success,
        "records": records,
        "chunks_output": str(args.chunks_output),
        "chunks_output_sha256": _sha256(args.chunks_output),
        "physical_scene_calibration_verified": False,
        "policy_task_quality_passed": False,
        "g1_contract_verified": False,
        "g1_sim_eligible": False,
        "g1_execution_enabled": False,
        "hardware_execution_performed": False,
        "real_time_watchdog_validated": bool(
            not args.paused_step_synchronous_diagnostic
            and records
            and all(
                record.get("execution_performed") is not True
                or record.get("observation_age_at_commit_ms", np.inf)
                <= args.maximum_observation_age_ms
                for record in records
            )
        ),
        "verdict": (
            "Experimental geometric stack persisted for three replans, but uncalibrated scene and contract gates remain blocked."
            if task_success else
            "Closed-loop task success was not demonstrated; keep all neural actions quarantined."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(json.dumps({
        "completed_cycles": report["completed_cycles"],
        "abort_reason": abort_reason,
        "closed_loop_completed_without_abort": completed_without_abort,
        "task_success": task_success,
        "final_task_metrics": final_cubes["task_metrics"],
    }, indent=2))
    if not task_success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
