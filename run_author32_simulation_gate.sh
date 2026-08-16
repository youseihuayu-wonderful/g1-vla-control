#!/usr/bin/env bash
# Author-confirmed LGG100 horizon-32 simulation gate. Simulation only; never hardware.
set -Eeuo pipefail

ROOT="${SHIHUA_ROOT:-/home/user1/workspace/shihua}"
REPO="$ROOT/projects/g1-vla-control"
OPENPI="$ROOT/projects/openpi"
PY="$ROOT/.venvs/g1-sim/bin/python"
UV="/home/user1/.local/bin/uv"
CHECKPOINT="$ROOT/models/stack-cube-eef-24k"
RESULTS="$ROOT/results"
LOGS="$ROOT/logs"
CALIBRATION="$RESULTS/camera_calibration_search_v2.json"
SEMANTIC="$RESULTS/lgg100_author32_semantic_validation.json"
PORT=8765
GPU_INDEX="${CUDA_DEVICE_INDEX:-5}"
PIDFILE="$LOGS/lgg100_author32_gate_8765.pid"
SERVER_LOG="$LOGS/lgg100_author32_gate_8765.log"
STATUS="$RESULTS/lgg100_author32_simulation_gate_status.json"
CURRENT_STAGE="preflight"
SERVER_PID=""

mkdir -p "$RESULTS" "$LOGS"

write_status() {
  local code="$1"
  CURRENT_STAGE_VALUE="$CURRENT_STAGE" EXIT_CODE_VALUE="$code" STATUS_PATH="$STATUS" \
    "$PY" - <<'PY'
import datetime, json, os, pathlib
path = pathlib.Path(os.environ["STATUS_PATH"])
payload = {
    "scope": "Author-confirmed horizon-32 simulation orchestration; never hardware.",
    "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "last_stage": os.environ["CURRENT_STAGE_VALUE"],
    "exit_code": int(os.environ["EXIT_CODE_VALUE"]),
    "completed": int(os.environ["EXIT_CODE_VALUE"]) == 0,
    "g1_contract_verified": False,
    "g1_sim_eligible": False,
    "g1_execution_enabled": False,
    "hardware_execution_performed": False,
}
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

cleanup_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    local pgid cmd
    pgid="$(ps -o pgid= -p "$SERVER_PID" | tr -d ' ')"
    cmd="$(ps -o cmd= -p "$SERVER_PID")"
    if [[ "$pgid" == "$SERVER_PID" && "$cmd" == *"lgg100_candidate_server.py"* && "$cmd" == *"--port $PORT"* ]]; then
      kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
      for _ in $(seq 1 30); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
      fi
    else
      echo "Refusing to stop an unverified process: pid=$SERVER_PID pgid=$pgid cmd=$cmd" >&2
    fi
  fi
  rm -f "$PIDFILE"
}

on_exit() {
  local code=$?
  trap - EXIT INT TERM
  cleanup_server
  write_status "$code" || true
  exit "$code"
}
trap on_exit EXIT INT TERM

for path in "$PY" "$UV" "$CHECKPOINT/params/_METADATA" "$CALIBRATION" "$SEMANTIC"; do
  [[ -e "$path" ]] || { echo "missing prerequisite: $path" >&2; exit 1; }
done
[[ "$(hostname)" == "nnmc65" ]] || { echo "unexpected host" >&2; exit 1; }
[[ "$(id -un)" == "user1" ]] || { echo "unexpected user" >&2; exit 1; }
if ss -ltn | grep -q "127.0.0.1:$PORT"; then
  echo "loopback port $PORT is already occupied" >&2
  exit 1
fi

CURRENT_STAGE="contract_and_gpu_preflight"
cd "$REPO"
[[ "$(git status --porcelain)" == "" ]] || { echo "repository is dirty" >&2; exit 1; }
"$PY" - <<'PY'
import json
from g1_policy_contract import ACTION_HORIZON, CONTRACT
assert ACTION_HORIZON == 32
assert CONTRACT["observation"]["discrete_state_input"] is False
assert CONTRACT["observation"]["preprocessing"]["operation"] == "resize_with_pad"
assert CONTRACT["production_policy"]["model_config_name"] == "pi05_g1_eef"
print(json.dumps({"action_horizon": ACTION_HORIZON, "core_config": "pi05_g1_eef"}))
PY
nvidia-smi -i "$GPU_INDEX" --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader

CURRENT_STAGE="exact_t0_observations"
for spec in near:008:0.08 mixed:016:0.16 far:024:0.24; do
  IFS=: read -r role tag offset <<< "$spec"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" MUJOCO_GL=egl "$PY" build_calibrated_observation.py \
    --calibration-report "$CALIBRATION" \
    --cube-x-offset-m "$offset" \
    --output "$RESULTS/lgg100_author32_t0_offset_${tag}_observation.npz" \
    --evidence "$RESULTS/lgg100_author32_t0_offset_${tag}_observation.json" \
    > "$LOGS/lgg100_author32_t0_offset_${tag}_observation.log" 2>&1
done

CURRENT_STAGE="strict_loopback_policy_server"
cd "$OPENPI"
: > "$SERVER_LOG"
setsid env CUDA_VISIBLE_DEVICES="$GPU_INDEX" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$UV" run python "$REPO/lgg100_candidate_server.py" \
  --checkpoint-dir "$CHECKPOINT" --host 127.0.0.1 --port "$PORT" \
  --action-horizon 32 --allow-candidate-restore \
  > "$SERVER_LOG" 2>&1 < /dev/null &
SERVER_PID=$!
echo "$SERVER_PID" > "$PIDFILE"
for _ in $(seq 1 180); do
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$SERVER_LOG"; exit 1; }
  ss -ltn | grep -q "127.0.0.1:$PORT" && break
  sleep 1
done
ss -ltn | grep -q "127.0.0.1:$PORT" || { tail -100 "$SERVER_LOG"; exit 1; }

CURRENT_STAGE="five_draw_observation_probes"
cd "$REPO"
for tag in 008 016 024; do
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" MUJOCO_GL=egl "$PY" lgg100_observation_probe.py \
    --observation "$RESULTS/lgg100_author32_t0_offset_${tag}_observation.npz" \
    --host 127.0.0.1 --port "$PORT" --draws 5 \
    --output "$RESULTS/lgg100_author32_t0_offset_${tag}_draws_quarantined.npz" \
    --report "$RESULTS/lgg100_author32_t0_offset_${tag}_probe.json" \
    > "$LOGS/lgg100_author32_t0_offset_${tag}_probe.log" 2>&1
done

CURRENT_STAGE="quarantined_ensembles"
for tag in 008 016 024; do
  "$PY" build_quarantined_ensemble.py \
    --draws "$RESULTS/lgg100_author32_t0_offset_${tag}_draws_quarantined.npz" \
    --probe-report "$RESULTS/lgg100_author32_t0_offset_${tag}_probe.json" \
    --output "$RESULTS/lgg100_author32_t0_offset_${tag}_ensemble_quarantined.npz" \
    --report "$RESULTS/lgg100_author32_t0_offset_${tag}_ensemble.json" \
    > "$LOGS/lgg100_author32_t0_offset_${tag}_ensemble.log" 2>&1
done

CURRENT_STAGE="swept_preflight"
for spec in 008:0.08 016:0.16 024:0.24; do
  IFS=: read -r tag offset <<< "$spec"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" MUJOCO_GL=egl "$PY" lgg100_quarantined_preflight.py \
    --chunks "$RESULTS/lgg100_author32_t0_offset_${tag}_ensemble_quarantined.npz" \
    --semantic-report "$SEMANTIC" \
    --cube-x-offset-m "$offset" \
    --observation "$RESULTS/lgg100_author32_t0_offset_${tag}_observation.npz" \
    --output "$RESULTS/lgg100_author32_t0_offset_${tag}_ensemble_preflight.json" \
    > "$LOGS/lgg100_author32_t0_offset_${tag}_preflight.log" 2>&1
done

CURRENT_STAGE="single_chunk_dynamics"
for spec in 008:0.08 016:0.16 024:0.24; do
  IFS=: read -r tag offset <<< "$spec"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" MUJOCO_GL=egl "$PY" lgg100_quarantined_sim_diagnostic.py \
    --chunks "$RESULTS/lgg100_author32_t0_offset_${tag}_ensemble_quarantined.npz" \
    --semantic-report "$SEMANTIC" \
    --preflight-report "$RESULTS/lgg100_author32_t0_offset_${tag}_ensemble_preflight.json" \
    --cube-x-offset-m "$offset" \
    --output "$RESULTS/lgg100_author32_t0_offset_${tag}_ensemble_sim.json" \
    --allow-quarantined-sim-diagnostic \
    > "$LOGS/lgg100_author32_t0_offset_${tag}_sim.log" 2>&1
done

CURRENT_STAGE="phase_speed_sweep_gate"
set +e
"$PY" phase_speed_sweep_validation.py \
  --scenario near 0.08 \
    "$RESULTS/lgg100_author32_t0_offset_008_ensemble.json" \
    "$RESULTS/lgg100_author32_t0_offset_008_ensemble_preflight.json" \
    "$RESULTS/lgg100_author32_t0_offset_008_ensemble_sim.json" \
  --scenario mixed 0.16 \
    "$RESULTS/lgg100_author32_t0_offset_016_ensemble.json" \
    "$RESULTS/lgg100_author32_t0_offset_016_ensemble_preflight.json" \
    "$RESULTS/lgg100_author32_t0_offset_016_ensemble_sim.json" \
  --scenario far 0.24 \
    "$RESULTS/lgg100_author32_t0_offset_024_ensemble.json" \
    "$RESULTS/lgg100_author32_t0_offset_024_ensemble_preflight.json" \
    "$RESULTS/lgg100_author32_t0_offset_024_ensemble_sim.json" \
  --output "$RESULTS/lgg100_author32_t0_phase_speed_sweep_validation.json" \
  > "$LOGS/lgg100_author32_phase_speed_sweep.log" 2>&1
PHASE_EXIT=$?
set -e
[[ -f "$RESULTS/lgg100_author32_t0_phase_speed_sweep_validation.json" ]]
"$PY" - <<'PY'
import json
path = "/home/user1/workspace/shihua/results/lgg100_author32_t0_phase_speed_sweep_validation.json"
report = json.load(open(path))
assert report["controlled_near_far_speed_behavior_passed"] is True
# Full coverage may remain false when an author-32 t=0 chunk contains no
# free-space-to-precision transition. Adaptive ON remains blocked in that case.
PY

CURRENT_STAGE="adaptive_off_realtime_watchdog"
set +e
CUDA_VISIBLE_DEVICES="$GPU_INDEX" MUJOCO_GL=egl "$PY" lgg100_quarantined_closed_loop.py \
  --host 127.0.0.1 --port "$PORT" \
  --camera-calibration "$CALIBRATION" \
  --semantic-report "$SEMANTIC" \
  --phase-speed-report "$RESULTS/lgg100_author32_t0_phase_speed_sweep_validation.json" \
  --cube-x-offset-m 0.08 --cycles 1 --prefix-duration-s 0.03333333333333333 \
  --maximum-observation-age-ms 100 --online-ik-iterations 30 \
  --output "$RESULTS/lgg100_author32_closed_loop_t0_offset008_baseline_realtime.json" \
  --chunks-output "$RESULTS/lgg100_author32_closed_loop_t0_offset008_baseline_realtime_quarantined.npz" \
  --allow-quarantined-closed-loop \
  > "$LOGS/lgg100_author32_closed_loop_baseline_realtime.log" 2>&1
REALTIME_EXIT=$?
set -e
[[ -f "$RESULTS/lgg100_author32_closed_loop_t0_offset008_baseline_realtime.json" ]]

CURRENT_STAGE="adaptive_off_paused_baseline"
set +e
CUDA_VISIBLE_DEVICES="$GPU_INDEX" MUJOCO_GL=egl "$PY" lgg100_quarantined_closed_loop.py \
  --host 127.0.0.1 --port "$PORT" \
  --camera-calibration "$CALIBRATION" \
  --semantic-report "$SEMANTIC" \
  --phase-speed-report "$RESULTS/lgg100_author32_t0_phase_speed_sweep_validation.json" \
  --cube-x-offset-m 0.08 --cycles 30 --prefix-duration-s 0.03333333333333333 \
  --maximum-observation-age-ms 100 --online-ik-iterations 30 \
  --paused-step-synchronous-diagnostic \
  --output "$RESULTS/lgg100_author32_closed_loop_t0_offset008_baseline_paused.json" \
  --chunks-output "$RESULTS/lgg100_author32_closed_loop_t0_offset008_baseline_paused_quarantined.npz" \
  --allow-quarantined-closed-loop \
  > "$LOGS/lgg100_author32_closed_loop_baseline_paused.log" 2>&1
PAUSED_EXIT=$?
set -e
[[ -f "$RESULTS/lgg100_author32_closed_loop_t0_offset008_baseline_paused.json" ]]

CURRENT_STAGE="summary"
PHASE_EXIT_VALUE="$PHASE_EXIT" REALTIME_EXIT_VALUE="$REALTIME_EXIT" PAUSED_EXIT_VALUE="$PAUSED_EXIT" "$PY" - <<'PY'
import hashlib, json, os, pathlib
root = pathlib.Path("/home/user1/workspace/shihua")
r = root / "results"
def load(name): return json.loads((r / name).read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
phase = load("lgg100_author32_t0_phase_speed_sweep_validation.json")
realtime = load("lgg100_author32_closed_loop_t0_offset008_baseline_realtime.json")
paused = load("lgg100_author32_closed_loop_t0_offset008_baseline_paused.json")
summary = {
    "scope": "Author-confirmed horizon-32 MuJoCo gates and Adaptive-OFF baseline; never hardware.",
    "phase_speed_gate": {
        "process_exit": int(os.environ.get("PHASE_EXIT_VALUE", "0")),
        "near_far_passed": phase["controlled_near_far_speed_behavior_passed"],
        "mixed_transition_coverage_passed": phase[
            "mixed_transition_coverage_passed"
        ],
        "full_phase_speed_passed": phase[
            "controlled_phase_speed_behavior_passed"
        ],
        "path": str(r / "lgg100_author32_t0_phase_speed_sweep_validation.json"),
        "sha256": sha(r / "lgg100_author32_t0_phase_speed_sweep_validation.json"),
    },
    "realtime_baseline": {
        "process_exit": int(os.environ["REALTIME_EXIT_VALUE"]),
        "abort_reason": realtime.get("abort_reason"),
        "completed_cycles": realtime.get("completed_cycles"),
        "task_success": realtime.get("task_success"),
        "real_time_watchdog_validated": realtime.get("real_time_watchdog_validated"),
        "path": str(r / "lgg100_author32_closed_loop_t0_offset008_baseline_realtime.json"),
        "sha256": sha(r / "lgg100_author32_closed_loop_t0_offset008_baseline_realtime.json"),
    },
    "paused_baseline": {
        "process_exit": int(os.environ["PAUSED_EXIT_VALUE"]),
        "abort_reason": paused.get("abort_reason"),
        "completed_cycles": paused.get("completed_cycles"),
        "task_success": paused.get("task_success"),
        "path": str(r / "lgg100_author32_closed_loop_t0_offset008_baseline_paused.json"),
        "sha256": sha(r / "lgg100_author32_closed_loop_t0_offset008_baseline_paused.json"),
    },
    "adaptive_on_executed": False,
    "policy_task_quality_passed": False,
    "physical_scene_calibration_verified": False,
    "g1_contract_verified": False,
    "g1_sim_eligible": False,
    "g1_execution_enabled": False,
    "hardware_execution_performed": False,
    "verdict": "Adaptive ON remains blocked until the author-32 Adaptive-OFF baseline succeeds under the registered gates.",
}
out = r / "lgg100_author32_simulation_gate_summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
print(out)
print(json.dumps(summary, indent=2))
PY

CURRENT_STAGE="complete"
