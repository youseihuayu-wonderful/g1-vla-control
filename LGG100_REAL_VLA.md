# LGG100 production VLA integration for G1 EDU

`LGG100/stack-cube-eef-24k` is the selected production VLA. This runbook loads
the actual public weights and queries them with the frozen G1 EDU observation
contract. Yuhao directly confirmed the core config name `pi05_g1_eef`,
`action_horizon=32`, and `discrete_state_input=False`. The current first stage
remains output-only: MuJoCo dynamics and G1 hardware remain blocked until the
remaining calibration, task-quality, realtime, and safety gates pass. The full real-VLA → semantic
attestation → Adaptive Module → task-level A/B plan is in
[`REAL_LGG100_ADAPTIVE_WORKFLOW.md`](REAL_LGG100_ADAPTIVE_WORKFLOW.md).

## Evidence boundary

The checkpoint repository contains 17 files / 7,168,237,327 bytes
(6.676 GiB), including Orbax params and 16-D state/action norm stats, at:

```text
revision: cced7a7ff7b454fdcac555457a1a2a3dc262ac77
```

It contains no README, OpenPI commit, TrainConfig, DataConfig, or custom
joint↔EEF transform. Repository history contains only the initial commit and a
single checkpoint upload. The public `openpi-fintune` repository, direct author confirmation, and the
pinned `leihao100/g1-client` deployment repository now establish the EEF
contract, official quaternion consumer boundary, and receding-horizon reference.
They still do not provide the complete historical TrainConfig. The project
owner has separately confirmed the deployment cadence as 15 Hz; the 30 Hz help
text is retained only as stale-documentation provenance.

Confirmed/supported model fields:

```text
Pi0Config
pi05=True
paligemma_variant=gemma_2b_lora
PaliGemma LoRA rank=16
action_expert_variant=gemma_300m
internal action_dim=32
max_token_len=200
published state/action dim=16
quantile q01/q99 normalization
action_horizon=32
discrete_state_input=False
model_config_name=pi05_g1_eef
```

The server restores with `remove_extra_params=False`; missing or extra
parameter leaves fail startup instead of being silently discarded. After the pinned author transform and deployment audit, strict server metadata
may report:

```text
g1_contract_verified: true
g1_action_compatible: true
official_consumer_quaternion_postprocessing_required: true
safe_for_g1_hardware: false
```

The first two fields establish the EEF action boundary only. They do not prove
sequential IK, swept collision, task success, physical calibration, or hardware
safety.

## Gate L0 — actual Ubuntu NVIDIA host

The existing local SSH alias `home` points to a macOS MacBook Pro, not an
Ubuntu NVIDIA server. Use the real GPU host below:

```bash
ssh GPU_USER@GPU_HOST
cat /etc/os-release
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
```

Requirements:

- Ubuntu 22.04 (the audited OpenPI platform);
- NVIDIA GPU with >8 GB VRAM for inference;
- at least 30 GB free disk for source, environment, cache, and checkpoint;
- outbound GitHub, Hugging Face, and model-tokenizer asset access.

Do not expose TCP 8000 publicly. Only SSH needs to be reachable.

## Gate L1 — install the pinned OpenPI source

On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl build-essential tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

mkdir -p ~/robot-vla
cd ~/robot-vla
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install huggingface_hub

git clone --recurse-submodules https://github.com/danielchen26/g1-vla-control.git \
  ~/g1_vla_control
```

Verify CUDA, not CPU:

```bash
cd ~/robot-vla/openpi
uv run python - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
assert jax.default_backend() == "gpu"
PY
```

## Gate L2 — download the exact Hugging Face revision

```bash
cd ~/robot-vla/openpi
uv run python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="LGG100/stack-cube-eef-24k",
    revision="cced7a7ff7b454fdcac555457a1a2a3dc262ac77",
    local_dir=Path.home() / "robot-vla/checkpoints/stack-cube-eef-24k",
)
print(path)
PY
```
Confirm the required files:

```bash
CHECKPOINT="$HOME/robot-vla/checkpoints/stack-cube-eef-24k"
test -f "$CHECKPOINT/_CHECKPOINT_METADATA"
test -f "$CHECKPOINT/params/_METADATA"
test -f "$CHECKPOINT/assets/stack-cube-eef/norm_stats.json"
du -sh "$CHECKPOINT"
```

## Gate L3 — strict real-weight restore

Start in tmux:

```bash
tmux new -s lgg100
cd ~/robot-vla/openpi
CHECKPOINT="$HOME/robot-vla/checkpoints/stack-cube-eef-24k"
uv run python ~/g1_vla_control/lgg100_candidate_server.py \
  --checkpoint-dir "$CHECKPOINT" \
  --action-horizon 32 \
  --port 8000 \
  --allow-candidate-restore \
  2>&1 | tee "$HOME/robot-vla/lgg100_server.log"
```

A valid startup must print:

```text
Strict LGG100 candidate restore succeeded
```

If strict parameter equality fails, stop. Do not switch to
`remove_extra_params=True`, drop LoRA leaves, or force a nearby model config.

Detach tmux with `Ctrl-B`, then `D`. Check from a second SSH session:

```bash
ss -ltnp | grep ':8000'
nvidia-smi
tail -100 ~/robot-vla/lgg100_server.log
```

## Gate L4 — SSH tunnel

On the Mac in a dedicated terminal:

```bash
ssh \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -N \
  -L 127.0.0.1:8000:127.0.0.1:8000 \
  GPU_USER@GPU_HOST
```

Keep this terminal open. If local port 8000 is occupied, bind local 8001 and
pass `--port 8001` to the client.

## Gate L5 — local client environment

Use a separate environment so pinned `openpi-client` can keep NumPy `<2`:

```bash
conda create -n g1-vla-client python=3.12 pip -y
conda activate g1-vla-client
cd ~/g1_vla_control
python -m pip install "numpy>=1.26,<2" -r requirements.txt
python -m pip install \
  "git+https://github.com/Physical-Intelligence/openpi.git@15a9616a00943ada6c20a0f158e3adb39df2ccac#subdirectory=packages/openpi-client"
```

## Gate L6 — real neural output-only simulation observation

```bash
cd ~/g1_vla_control
conda activate g1-vla-client
python lgg100_sim_smoke.py \
  --host 127.0.0.1 \
  --port 8000 \
  --calls 5 \
  --warmup-calls 1 \
  --call-timeout-ms 60000
```

Inputs use the shared G1 preprocessing: each MuJoCo camera renders 640×480 RGB,
then `g1_policy_contract.py` performs the OpenPI-compatible aspect-preserving
bilinear resize with zero padding to 224×224. The resulting policy observation is:

```text
cam_left_high       uint8 [224,224,3]
cam_left_wrist      uint8 [224,224,3]
cam_right_wrist     uint8 [224,224,3]
pelvis-frame state  float [16]
prompt              exact public task text
```

Outputs:

```text
results/lgg100_vla_smoke_real.json
results/lgg100_action_chunk_real.npz
```

The neural-output gate passes only if server metadata proves strict real-weight
restore and all calls return one stable finite `[T,16]` shape. It still records:

```text
author_core_config_directly_confirmed: true
complete_author_train_config_available: false
g1_contract_verified: true
g1_sim_eligible: false
g1_execution_enabled: false
adaptive_retimer_enabled: false
```

## Gate L7 — author contract recovered; sequential preflight still required

The pinned author sources establish quaternion-only post-processing before IK.
`lgg100_adaptive_ab.py` still requires all of the following in both JSON and NPZ:

```text
g1_policy_contract_id = g1_edu_dual_dex1_eef_v1
g1_policy_contract_sha256 = current frozen SHA
g1_contract_verified = true
g1_sim_eligible = true
```

The server may now set `g1_contract_verified=true` because the pinned author
transform, deployment implementation, and behavioral semantic validation agree.
It must keep `g1_sim_eligible=false` until complete multi-chunk sequential IK and
swept-path qualification passes. Retraining is not required. Contract/action
fingerprints continue to detect accidental mismatch and raw neural hashes remain
unchanged.

After genuine semantic verification, every target must pass finite checks,
quaternion norm, IK reachability, joint limits, and phase-aware collision
preflight. A rejection writes evidence and performs no execution. For a passing
preflight, both branches use:

- the exact same saved neural chunk and SHA-256;
- the exact same initial MuJoCo state;
- identical path geometry;
- Gate-A EEF filtering;
- Gate-A.2 joint-command filtering.

Only timestamps differ:

```text
baseline: nominal 1.0x timestamps
adaptive: AdaptiveRetimer timestamps from distance + stability policy
```

Only after L7 is legitimately unlocked, the evidence path is:

```text
results/lgg100_adaptive_ab.json
```

The conservative single-chunk candidate gate requires lower duration without
worse endpoint error, contact rate, actual joint jerk, or command-limit
compliance. Even a pass does **not** prove better block-stacking success.

## Gate L8 — task-level conclusion (not implemented or claimed yet)

To answer whether adaptive speed is genuinely useful, repeat paired trials over
multiple initial cube positions and multiple independently generated policy
chunks, then run a closed-loop policy replan. Pre-register and compare:

- full stack success rate;
- completion time distribution;
- collision/drop rate;
- preflight rejection rate;
- Gate-A/A.2 intervention rate;
- EEF tracking error;
- actual joint speed/acceleration/jerk;
- P50/P95/P99 inference and observation age.

Adaptive is accepted only if task success is non-inferior, collision and safety
metrics do not degrade, and completion time improves. Until L8 passes, the
correct conclusion is **“single-chunk retiming evidence only.”**
