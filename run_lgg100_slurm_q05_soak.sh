#!/usr/bin/env bash
#SBATCH --job-name=lgg100-q05-soak
#SBATCH --partition=all
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:20:00
#SBATCH --output=../../logs/%x-%j.log
#SBATCH --signal=B:TERM@120

# Q0.5: four-hour multi-scenario output-only endurance inference.
# Every GPU call audits a frozen LGG100 chunk; no dummy occupancy is used.
set -Eeuo pipefail
umask 077

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "Q0.5 must run inside a Slurm allocation" >&2
  exit 2
}
REPO="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
[[ -f "$REPO/lgg100_output_only_soak.py" ]] || {
  echo "Submit Q0.5 from the g1-vla-control repository" >&2
  exit 2
}
ROOT="${SHIHUA_ROOT:-$(cd "$REPO/../.." && pwd)}"
OPENPI="$ROOT/projects/openpi"
PY="${G1_PYTHON:-$OPENPI/.venv/bin/python}"
UV="${UV_BIN:-$HOME/.local/bin/uv}"
CHECKPOINT="$ROOT/models/stack-cube-eef-24k"
RESULT_ROOT="$ROOT/results/lgg100_slurm_q05_soak"
LOG_ROOT="$ROOT/logs/lgg100_slurm_q05_soak"
RUN_ID="job_${SLURM_JOB_ID}"
RUN_RESULTS="$RESULT_ROOT/$RUN_ID"
RUN_LOGS="$LOG_ROOT/$RUN_ID"
STATUS="$RUN_RESULTS/job_status.json"
PORT=$((28000 + SLURM_JOB_ID % 10000))
SERVER_LOG="$RUN_LOGS/policy_server.log"
GPU_LOG="$RUN_LOGS/gpu_memory.csv"
SERVER_PID=""
MONITOR_PID=""
FINAL_STATE="FAILED"
CURRENT_STAGE="ALLOCATED"
FAILURE_REASON="job_exited_before_completion"
TARGET_DURATION_S=14400
TARGET_RATE_HZ=5

mkdir -p "$RUN_RESULTS" "$RUN_LOGS"

write_status() {
  local state="$1" reason="$2" code="$3"
  STATE_VALUE="$state" REASON_VALUE="$reason" EXIT_CODE_VALUE="$code" \
  STAGE_VALUE="$CURRENT_STAGE" STATUS_PATH="$STATUS" JOB_ID_VALUE="$SLURM_JOB_ID" \
  "$PY" - <<'PY'
import datetime,json,os,pathlib
p=pathlib.Path(os.environ["STATUS_PATH"])
payload={
 "schema_version":"lgg100_slurm_q05_job_status_v1",
 "updated_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "job_id":os.environ["JOB_ID_VALUE"],
 "state":os.environ["STATE_VALUE"],
 "stage":os.environ["STAGE_VALUE"],
 "reason":os.environ["REASON_VALUE"],
 "exit_code":int(os.environ["EXIT_CODE_VALUE"]),
 "scope":"One allocated L40S; four-hour frozen LGG100 output-only endurance inference.",
 "neural_training":False,
 "mujoco_dynamics_executed":False,
 "g1_execution_enabled":False,
 "hardware_execution_performed":False,
}
t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(payload,indent=2)+"\n");t.replace(p)
PY
}

fail() {
  FAILURE_REASON="$1"
  echo "Q0.5 failed: $FAILURE_REASON" >&2
  exit 1
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
    write_status "COMPLETE" "four_hour_soak_passed" 0 || true
    exit 0
  fi
  write_status "FAILED" "$FAILURE_REASON" "$code" || true
  exit "$code"
}
trap cleanup EXIT INT TERM

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
  "$PY" "$UV" "$OPENPI/pyproject.toml" \
  "$CHECKPOINT/_CHECKPOINT_METADATA" "$CHECKPOINT/params/_METADATA" \
  "$ROOT/results/lgg100_author32_t0_offset_008_observation.npz" \
  "$ROOT/results/lgg100_author32_t0_offset_016_observation.npz" \
  "$ROOT/results/lgg100_author32_t0_offset_024_observation.npz"; do
  [[ -e "$path" ]] || fail "missing_prerequisite"
done
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || fail "repository_dirty"

CURRENT_STAGE="GPU_PREFLIGHT"
GPU_TOKEN="${CUDA_VISIBLE_DEVICES:-}"
[[ -n "$GPU_TOKEN" && "$GPU_TOKEN" != *,* ]] || fail "exactly_one_slurm_gpu_required"
assert_no_foreign_compute_processes "$GPU_TOKEN"
GPU_FREE_MIB="$(nvidia-smi -i "$GPU_TOKEN" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
[[ "$GPU_FREE_MIB" =~ ^[0-9]+$ ]] || fail "gpu_memory_query_failed"
(( GPU_FREE_MIB >= 40000 )) || fail "allocated_gpu_not_fully_available"
nvidia-smi -i "$GPU_TOKEN" \
  --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits > "$RUN_RESULTS/gpu_preflight.csv"

CURRENT_STAGE="STRICT_RESTORE"
write_status "RUNNING" "strict_restore_started" 0
: > "$SERVER_LOG"
: > "$GPU_LOG"
cd "$OPENPI"
setsid env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$UV" run --frozen python "$REPO/lgg100_candidate_server.py" \
  --checkpoint-dir "$CHECKPOINT" --host 127.0.0.1 --port "$PORT" \
  --action-horizon 32 --allow-candidate-restore \
  > "$SERVER_LOG" 2>&1 < /dev/null &
SERVER_PID=$!
(
  echo "timestamp_utc,memory_used_mib,memory_free_mib,utilization_gpu_percent"
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    printf '%s,' "$(date -u +%FT%TZ)"
    nvidia-smi -i "$GPU_TOKEN" --query-gpu=memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ' '
    sleep 5
  done
) >> "$GPU_LOG" 2>&1 &
MONITOR_PID=$!
for _ in $(seq 1 240); do
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$SERVER_LOG" >&2; fail "strict_restore_server_exited"; }
  ss -ltnH "sport = :$PORT" | grep -q . && break
  sleep 1
done
ss -ltnH "sport = :$PORT" | grep -q . || fail "policy_server_ready_timeout"
assert_no_foreign_compute_processes "$GPU_TOKEN" "$SERVER_PID"
write_status "READY" "strict_restore_server_ready" 0

CURRENT_STAGE="FOUR_HOUR_MULTI_SCENARIO_SOAK"
write_status "RUNNING" "four_hour_output_only_soak_started" 0
cd "$REPO"
"$PY" lgg100_output_only_soak.py \
  --host 127.0.0.1 --port "$PORT" \
  --scenario near "$ROOT/results/lgg100_author32_t0_offset_008_observation.npz" \
  --scenario mixed "$ROOT/results/lgg100_author32_t0_offset_016_observation.npz" \
  --scenario far "$ROOT/results/lgg100_author32_t0_offset_024_observation.npz" \
  --duration-s "$TARGET_DURATION_S" --rate-hz "$TARGET_RATE_HZ" \
  --heartbeat-s 60 --sample-every 300 \
  --output-dir "$RUN_RESULTS/soak" \
  > "$RUN_LOGS/soak.log" 2>&1
assert_no_foreign_compute_processes "$GPU_TOKEN" "$SERVER_PID"

CURRENT_STAGE="SOAK_SUMMARY"
SOAK_PATH="$RUN_RESULTS/soak/soak_summary.json" GPU_LOG_PATH="$GPU_LOG" \
SUMMARY_PATH="$RUN_RESULTS/qualification_summary.json" "$PY" - <<'PY'
import csv,json,os,pathlib
soak=json.loads(pathlib.Path(os.environ["SOAK_PATH"]).read_text())
assert soak["completed_target_duration"] is True
assert soak["target_duration_s"] >= 14400
assert soak["completed_calls"] == soak["finite_shape_passes"]
used=[]
with pathlib.Path(os.environ["GPU_LOG_PATH"]).open() as f:
    for row in csv.DictReader(f):
        try: used.append(int(row["memory_used_mib"]))
        except (KeyError,TypeError,ValueError): pass
assert used
payload={
 "schema_version":"lgg100_slurm_q05_qualification_v1",
 "scope":"Four-hour frozen LGG100 multi-scenario output-only endurance inference.",
 "completed_target_duration":True,
 "actual_duration_s":soak["actual_duration_s"],
 "completed_calls":soak["completed_calls"],
 "finite_shape_passes":soak["finite_shape_passes"],
 "raw_contract_passes":soak["raw_contract_passes"],
 "bounded_analysis_available":soak["bounded_analysis_available"],
 "latency_ms":soak["latency_ms"],
 "raw_quaternion_norm_error":soak["raw_quaternion_norm_error"],
 "peak_memory_used_mib":max(used),
 "neural_training":False,
 "mujoco_dynamics_executed":False,
 "g1_contract_verified":False,
 "g1_sim_eligible":False,
 "g1_execution_enabled":False,
 "hardware_execution_performed":False,
}
pathlib.Path(os.environ["SUMMARY_PATH"]).write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps(payload,indent=2))
PY

FINAL_STATE="COMPLETE"
FAILURE_REASON="four_hour_soak_passed"
