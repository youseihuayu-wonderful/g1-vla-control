#!/usr/bin/env bash
#SBATCH --job-name=lgg100-q0
#SBATCH --partition=all
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=../../logs/%x-%j.log
#SBATCH --signal=B:TERM@120

# Q0: one-GPU, output-only qualification for the frozen LGG100 checkpoint.
# This job never trains, runs MuJoCo dynamics, or sends a hardware command.
set -Eeuo pipefail
umask 077

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "Q0 must run inside a Slurm allocation" >&2
  exit 2
}

REPO="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
[[ -f "$REPO/lgg100_candidate_server.py" ]] || {
  echo "Submit Q0 from the g1-vla-control repository" >&2
  exit 2
}
ROOT="${SHIHUA_ROOT:-$(cd "$REPO/../.." && pwd)}"
OPENPI="$ROOT/projects/openpi"
PY="${G1_PYTHON:-$OPENPI/.venv/bin/python}"
UV="${UV_BIN:-$HOME/.local/bin/uv}"
CHECKPOINT="$ROOT/models/stack-cube-eef-24k"
OBSERVATION="$ROOT/results/lgg100_author32_t0_offset_008_observation.npz"
RESULT_ROOT="$ROOT/results/lgg100_slurm_q0"
LOG_ROOT="$ROOT/logs/lgg100_slurm_q0"
RUN_ID="job_${SLURM_JOB_ID}"
RUN_RESULTS="$RESULT_ROOT/$RUN_ID"
RUN_LOGS="$LOG_ROOT/$RUN_ID"
STATUS="$RUN_RESULTS/status.json"
LATEST_STATUS="$RESULT_ROOT/latest_status.json"
PORT=$((18000 + SLURM_JOB_ID % 10000))
SERVER_LOG="$RUN_LOGS/policy_server.log"
GPU_LOG="$RUN_LOGS/gpu_memory.csv"
SERVER_PID=""
MONITOR_PID=""
FINAL_STATE="FAILED"
CURRENT_STAGE="ALLOCATED"
FAILURE_REASON="job_exited_before_completion"

mkdir -p "$RUN_RESULTS" "$RUN_LOGS"

write_status() {
  local state="$1" reason="$2" code="$3"
  STATE_VALUE="$state" REASON_VALUE="$reason" EXIT_CODE_VALUE="$code" \
  STAGE_VALUE="$CURRENT_STAGE" STATUS_PATH="$STATUS" LATEST_PATH="$LATEST_STATUS" \
  JOB_ID_VALUE="$SLURM_JOB_ID" "$PY" - <<'PY'
import datetime
import json
import os
from pathlib import Path

payload = {
    "schema_version": "lgg100_slurm_q0_status_v1",
    "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "job_id": os.environ["JOB_ID_VALUE"],
    "state": os.environ["STATE_VALUE"],
    "stage": os.environ["STAGE_VALUE"],
    "reason": os.environ["REASON_VALUE"],
    "exit_code": int(os.environ["EXIT_CODE_VALUE"]),
    "scope": "One allocated L40S; frozen LGG100 output-only qualification.",
    "neural_training": False,
    "mujoco_dynamics_executed": False,
    "g1_execution_enabled": False,
    "hardware_execution_performed": False,
}
for target in (Path(os.environ["STATUS_PATH"]), Path(os.environ["LATEST_PATH"])):
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(target)
PY
}

stop_verified_group() {
  local pid="$1" expected="$2"
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  local pgid cmd
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  cmd="$(ps -o cmd= -p "$pid")"
  if [[ "$pgid" != "$pid" || "$cmd" != *"$expected"* ]]; then
    echo "Refusing to stop unverified process pid=$pid" >&2
    return 1
  fi
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  stop_verified_group "$SERVER_PID" "lgg100_candidate_server.py" || true
  if [[ "$FINAL_STATE" == "COMPLETE" ]]; then
    write_status "COMPLETE" "qualification_passed" 0 || true
    exit 0
  fi
  write_status "FAILED" "$FAILURE_REASON" "$code" || true
  exit "$code"
}
trap cleanup EXIT INT TERM

fail() {
  FAILURE_REASON="$1"
  echo "Q0 failed: $FAILURE_REASON" >&2
  exit 1
}

assert_no_foreign_compute_processes() {
  local gpu_token="$1" expected_pgid="${2:-}" pid pgid
  while IFS= read -r pid; do
    pid="$(echo "$pid" | tr -d '[:space:]')"
    [[ -n "$pid" ]] || continue
    [[ -n "$expected_pgid" ]] || fail "allocated_gpu_has_preexisting_compute_process"
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [[ "$pgid" == "$expected_pgid" ]] || fail "foreign_compute_process_appeared"
  done < <(nvidia-smi -i "$gpu_token" --query-compute-apps=pid --format=csv,noheader,nounits)
}

write_status "ALLOCATED" "slurm_allocation_started" 0

CURRENT_STAGE="PREREQUISITES"
for path in \
  "$PY" \
  "$UV" \
  "$OPENPI/pyproject.toml" \
  "$CHECKPOINT/_CHECKPOINT_METADATA" \
  "$CHECKPOINT/params/_METADATA" \
  "$OBSERVATION"; do
  [[ -e "$path" ]] || fail "missing_prerequisite"
done
[[ "$(git -C "$REPO" status --porcelain)" == "" ]] || fail "repository_dirty"

CURRENT_STAGE="GPU_PREFLIGHT"
GPU_TOKEN="${CUDA_VISIBLE_DEVICES:-}"
[[ -n "$GPU_TOKEN" && "$GPU_TOKEN" != *,* ]] || fail "exactly_one_slurm_gpu_required"
command -v nvidia-smi >/dev/null || fail "nvidia_smi_missing"
assert_no_foreign_compute_processes "$GPU_TOKEN"
GPU_FREE_MIB="$(nvidia-smi -i "$GPU_TOKEN" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
[[ "$GPU_FREE_MIB" =~ ^[0-9]+$ ]] || fail "gpu_memory_query_failed"
(( GPU_FREE_MIB >= 40000 )) || fail "allocated_gpu_not_fully_available"
nvidia-smi -i "$GPU_TOKEN" \
  --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits > "$RUN_RESULTS/gpu_preflight.csv"

CURRENT_STAGE="CONTRACT_PREFLIGHT"
cd "$REPO"
"$PY" - <<'PY' > "$RUN_RESULTS/contract_preflight.json"
import json
from g1_policy_contract import ACTION_HORIZON, CONTRACT
assert ACTION_HORIZON == 32
assert CONTRACT["observation"]["discrete_state_input"] is False
assert CONTRACT["observation"]["preprocessing"]["operation"] == "resize_with_pad"
assert CONTRACT["production_policy"]["model_config_name"] == "pi05_g1_eef"
print(json.dumps({
    "action_horizon": ACTION_HORIZON,
    "model_config": "pi05_g1_eef",
    "hardware_execution_performed": False,
}))
PY

CURRENT_STAGE="STRICT_RESTORE"
write_status "RUNNING" "strict_restore_started" 0
: > "$SERVER_LOG"
: > "$GPU_LOG"
cd "$OPENPI"
setsid env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$UV" run --frozen python "$REPO/lgg100_candidate_server.py" \
  --checkpoint-dir "$CHECKPOINT" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --action-horizon 32 \
  --allow-candidate-restore \
  > "$SERVER_LOG" 2>&1 < /dev/null &
SERVER_PID=$!

(
  echo "timestamp_utc,memory_used_mib,memory_free_mib,utilization_gpu_percent"
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    printf '%s,' "$(date -u +%FT%TZ)"
    nvidia-smi -i "$GPU_TOKEN" \
      --query-gpu=memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ' '
    sleep 1
  done
) >> "$GPU_LOG" 2>&1 &
MONITOR_PID=$!

for _ in $(seq 1 240); do
  kill -0 "$SERVER_PID" 2>/dev/null || {
    tail -100 "$SERVER_LOG" >&2
    fail "strict_restore_server_exited"
  }
  ss -ltnH "sport = :$PORT" | grep -q . && break
  sleep 1
done
ss -ltnH "sport = :$PORT" | grep -q . || fail "policy_server_ready_timeout"
assert_no_foreign_compute_processes "$GPU_TOKEN" "$SERVER_PID"
write_status "READY" "strict_restore_server_ready" 0

CURRENT_STAGE="THREE_CALL_SMOKE"
cd "$REPO"
"$PY" lgg100_observation_probe.py \
  --observation "$OBSERVATION" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --draws 3 \
  --output "$RUN_RESULTS/smoke_actions_quarantined.npz" \
  --report "$RUN_RESULTS/smoke_report.json" \
  > "$RUN_LOGS/smoke_probe.log" 2>&1
assert_no_foreign_compute_processes "$GPU_TOKEN" "$SERVER_PID"

CURRENT_STAGE="THIRTY_CALL_OUTPUT_ONLY"
"$PY" lgg100_observation_probe.py \
  --observation "$OBSERVATION" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --draws 30 \
  --output "$RUN_RESULTS/formal_actions_quarantined.npz" \
  --report "$RUN_RESULTS/formal_report.json" \
  > "$RUN_LOGS/formal_probe.log" 2>&1
assert_no_foreign_compute_processes "$GPU_TOKEN" "$SERVER_PID"

CURRENT_STAGE="QUALIFICATION_SUMMARY"
REPORT_PATH="$RUN_RESULTS/formal_report.json" GPU_LOG_PATH="$GPU_LOG" \
SUMMARY_PATH="$RUN_RESULTS/qualification_summary.json" "$PY" - <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path

report_path = Path(os.environ["REPORT_PATH"])
report = json.loads(report_path.read_text())
summary = report["summary"]
assert report["draws"] == 30
assert summary["finite_shape_passes"] == 30
assert report["g1_execution_enabled"] is False
assert report["execution_performed"] is False

used = []
with Path(os.environ["GPU_LOG_PATH"]).open() as handle:
    for row in csv.DictReader(handle):
        try:
            used.append(int(row["memory_used_mib"]))
        except (KeyError, TypeError, ValueError):
            pass
assert used
payload = {
    "schema_version": "lgg100_slurm_q0_qualification_v1",
    "scope": "Frozen LGG100 output-only qualification on one Slurm-allocated L40S.",
    "draws": 30,
    "finite_shape_passes": summary["finite_shape_passes"],
    "latency_ms": summary["latency_ms"],
    "peak_memory_used_mib": max(used),
    "formal_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "qualification_passed": True,
    "neural_training": False,
    "mujoco_dynamics_executed": False,
    "g1_execution_enabled": False,
    "hardware_execution_performed": False,
}
Path(os.environ["SUMMARY_PATH"]).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

FINAL_STATE="COMPLETE"
FAILURE_REASON="qualification_passed"
