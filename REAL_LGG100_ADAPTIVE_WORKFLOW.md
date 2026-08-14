# 真正 LGG100 + G1 Adaptive Module 执行计划

本计划只针对：

```text
LGG100/stack-cube-eef-24k
revision cced7a7ff7b454fdcac555457a1a2a3dc262ac77
→ G1 Education + 双 Dex1 MuJoCo
→ Adaptive Retimer
→ 以后迁移到同一台 G1 EDU
```

当前真实 LGG100 神经调用数为 **0**。现有 17 cm 测试只证明确定性 G1
contract 路径能在远处加速、近处减速，不证明 LGG100 任务更快。

---

## 0. 最终目标和禁止事项

最终目标：比较真正 LGG100 在同一组 G1 叠积木场景中的：

```text
A: LGG100 + Baseline 1.0×
B: LGG100 + Adaptive Module
```

Adaptive 只有在成功率和安全不下降、完成时间改善时才能启用。

在 Gate R0–R8 完成前禁止：

- 把真实 LGG100 action 发给 G1 硬件；
- 仅因为输出是 16-D 就宣布兼容；
- 手改 JSON/NPZ，把 `g1_contract_verified` 改成 true；
- 为通过恢复而启用 `remove_extra_params=True`；
- 用两个不同的随机 chunk 做 single-chunk Baseline/Adaptive 比较；
- 把 deterministic fixture 结果写成 LGG100 结果；
- 在 8000 端口公开暴露 policy server。

---

## 1. 当前已有与仍缺的代码

### 已有

| 文件 | 作用 | 当前状态 |
|---|---|---|
| `g1_policy_contract.yaml` | G1 三相机、16-D state/action、30 Hz、50 horizon | 已冻结 v1.1 |
| `lgg100_candidate_server.py` | CUDA 上严格加载真实 Orbax 权重 | 代码完成，未在 GPU 运行 |
| `lgg100_sim_smoke.py` | G1 MuJoCo observation → 真实 VLA output-only | 代码完成，真实调用 0 |
| `lgg100_adaptive_ab.py` | 同一真实 chunk 的 Baseline/Adaptive 配对 | 代码完成，被 semantic gate 阻塞 |
| `g1_adaptive_phase_validation.py` | 17 cm fixture 的远处加速/近处减速覆盖 | 已通过；不是 LGG100 |
| `safety_governor.py` | IK、碰撞、EEF/Joint command limits | 已有 sim gate |

### 必须新增

| 计划文件 | 作用 | 完成条件 |
|---|---|---|
| `lgg100_semantic_validation.py` | 用多帧公开 episode 验证 action 逐维/frame/unit/horizon | 输出 hash-bound semantic attestation |
| `lgg100_semantic_attestation.json` | 绑定 HF revision、contract SHA、norm SHA、transform 和指标 | 人工审阅后才能解锁 MuJoCo |
| `adaptive_speed_context.py` | 给 retimer 增加 phase/clearance/tracking/contact/stale 输入 | fail-closed tests 通过 |
| `lgg100_closed_loop_eval.py` | 真正 LGG100 multi-chunk MuJoCo 闭环 | Baseline 在 adaptive OFF 时先通过 |
| `lgg100_adaptive_batch_ab.py` | 成批配对任务 A/B 和随机化统计 | 生成任务级 verdict |

---

# Phase A — 取得真正的 LGG100 输出

## R0 · 提供 Ubuntu NVIDIA 主机

### 我们需要

```text
SSH alias，或 GPU_USER@GPU_HOST
SSH port（若不是 22）
```

不要发送密码。使用 SSH key。

### 只读检查

```bash
cat /etc/os-release
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
df -h ~
```

### 通过条件

- Ubuntu NVIDIA；
- JAX 能识别 GPU；
- 至少 30 GB 可用磁盘；
- 实际 strict restore 和一次 inference 不 OOM；
- 建议至少 16 GB VRAM，24 GB 更稳妥。

### 停止条件

- JAX backend 是 CPU；
- checkpoint/inference OOM；
- 只能公开暴露 8000；
- host 不是已授权机器。

### 产物

```text
results/lgg100_gpu_preflight.json       # 待实现/生成
```

---

## R1 · 固定代码和权重

### 固定版本

```text
OpenPI: 15a9616a00943ada6c20a0f158e3adb39df2ccac
LGG100: cced7a7ff7b454fdcac555457a1a2a3dc262ac77
G1 contract: g1_edu_dual_dex1_eef_v1 + 当前 SHA
```

### 执行

严格按照 `LGG100_REAL_VLA.md` 的 L1/L2 安装和下载。

### 通过条件

- 17 个 HF 文件完整；
- `_CHECKPOINT_METADATA`、`params/_METADATA`、`norm_stats.json` 存在；
- 记录 checkpoint metadata 和 norm stats SHA-256；
- Git working tree 和 revision 可复现。

### 停止条件

revision 漂移、文件缺失或 hash 与本次审计不一致。

### 产物

```text
checkpoint snapshot
results/checkpoint_metadata_audit.json
```

---

## R2 · 严格恢复真实神经权重

### GPU 命令

```bash
cd ~/robot-vla/openpi
CHECKPOINT="$HOME/robot-vla/checkpoints/stack-cube-eef-24k"
uv run python ~/g1_vla_control/lgg100_candidate_server.py \
  --checkpoint-dir "$CHECKPOINT" \
  --action-horizon 32 \
  --port 8000 \
  --allow-candidate-restore
```

### 必须看到

```text
Strict LGG100 candidate restore succeeded
```

### 通过条件

- `jax.default_backend() == "gpu"`；
- `remove_extra_params=False` 严格参数树恢复成功；
- metadata 中 revision、norm hash、checkpoint hash 正确；
- GPU 无 OOM。

### 停止条件

任何 missing/extra leaf、shape mismatch、CPU fallback、OOM。

### 证据边界

R2 通过只证明 **真实权重被加载**，不证明 G1 action 语义正确。因此此时仍必须是：

```text
g1_contract_verified=false
g1_sim_eligible=false
g1_execution_enabled=false
```

---

## R3 · 建立 SSH tunnel

Mac 上：

```bash
ssh -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -N \
  -L 127.0.0.1:8000:127.0.0.1:8000 \
  GPU_USER@GPU_HOST
```

只绑定 `127.0.0.1`。不开放公网端口。

---

## R4 · 真实 output-only smoke

### 输入

```text
3× MuJoCo 640×480 RGB
→ aspect-preserving bilinear resize
→ zero padding to 224×224
pelvis-frame state[16]
canonical stack-cube prompt
```

### 命令

先 3-call probe：

```bash
python lgg100_sim_smoke.py \
  --host 127.0.0.1 --port 8000 \
  --calls 3 --warmup-calls 1
```

通过后正式 30-call：

```bash
python lgg100_sim_smoke.py \
  --host 127.0.0.1 --port 8000 \
  --calls 30 --warmup-calls 2
```

### 通过条件

- `neural_vla_claimed=true`；
- 30/30 finite；
- 所有输出稳定为 `[32,16]`；
- 无 NaN/Inf；
- 记录 P50/P95/P99 latency；
- 每个 chunk 有 SHA-256；
- 仍然 output-only。

### 停止条件

shape 漂移、非 16-D、canonical horizon 非 32、四元数错误、超时、OOM、server metadata 不匹配。

### 产物

```text
results/lgg100_vla_smoke_real.json
results/lgg100_action_chunk_real.npz
```

注意：当前脚本重复同一 MuJoCo observation；它只验证真实神经、传输和输出结构，不验证任务能力。

---

# Phase B — 证明 LGG100 输出真的能解释为 G1 16-D

## R5 · 多帧语义验证（关键 Gate）

这是目前最大的缺口。必须实现 `lgg100_semantic_validation.py`。

### 输入

从多个公开 episodes 抽取至少 30 个时间点，每个时间点包含：

```text
三路同步 RGB
raw 14-joint + 2-Dex1 state/action
时间戳
candidate FK 后的 16-D pelvis EEF
后续 32 帧 reference action
```

### 必须比较的假设

- pelvis frame vs world/base frame；
- EEF site：`wrist_yaw_link + [0.050,0,0] m`；
- quaternion `xyzw` vs `wxyz`；
- absolute target vs delta；
- delta 左乘/右乘和所在 frame；
- 30 Hz 与 author-confirmed canonical horizon 32；
- q01/q99 normalization；
- 左右手顺序和 Dex1 方向；
- action 与 future state 的时序 lag。

### 验证方法

1. 每个 observation 只做 output-only inference。
2. 对每个语义假设计算：
   - EEF position error；
   - quaternion geodesic error；
   - Dex1 error；
   - IK reachable rate；
   - joint/collision reject rate；
   - 与公开动作统计和 norm stats 的一致性。
3. 可视化每个 chunk，不允许靠静默 normalize/clamp 掩盖错误。
4. 要求同一个假设在全部样本上稳定胜出，而不是每帧换解释。

### 通过条件

生成并人工审阅：

```text
results/lgg100_semantic_validation.json
results/lgg100_semantic_attestation.json
results/lgg100_semantic_trajectory_view.*
```

Attestation 必须绑定：

```text
HF revision
OpenPI commit
G1 contract ID + SHA
checkpoint metadata SHA
norm stats SHA
transform version
horizon/rate
样本 manifest 和指标
reviewer/time
```

只有 attestation 通过后，server/client 才允许以代码方式产生：

```text
g1_contract_verified=true
g1_sim_eligible=true
```

### 停止条件

- 多个语义假设无法区分；
- 只有 shape 对，物理轨迹不合理；
- 大量 quaternion/IK/collision 失败；
- 需要补零、交换左右手或逐帧手工修正；
- 没有 hash-bound attestation。

禁止手工编辑 smoke JSON 解锁。

---

# Phase C — 把我们的 Adaptive Module 接到真实 chunk

## R6 · Module 输入补全

当前 module 只使用：

```text
EEF 到 chunk 最终目标的距离
robot stability
```

它还不知道物体和任务阶段。真正用于叠积木前必须增加：

```text
task_phase: free_space / approach / grasp / lift / place / retreat
minimum_object_and_robot_clearance_m
EEF_tracking_error_m
Dex1 command + measured state
contact state
chunk age / observation age / network stale
IK margin / joint-limit margin
pelvis stability
```

### Fail-closed 规则

- stale、未知 phase、低 clearance、tracking error 大、IK margin 小：禁止加速；
- approach/grasp/place/contact：强制 ≤1.0×，保守阶段可 ≤0.5×；
- 只有 free-space、clearance 足够、稳定且 tracking 正常时允许 >1.0×；
- 任何 safety gate 失败：hold/reject，不是简单降速继续撞。

### 必须新增的测试

- 远离目标但靠近桌面/方块时不能加速；
- free-space 且 clearance 足够时可以加速；
- grasp/place/contact 时减速；
- stale chunk 立即 hold；
- scale rate 连续，无 chunk 边界速度跳变；
- Baseline/Adaptive 的 action samples byte-identical。

### 产物

```text
adaptive_speed_context.py
results/adaptive_context_validation.json
```

---

## R7 · 真实 LGG100 chunk 的纯离线 Preflight

在任何 MuJoCo dynamics 前运行：

```text
fingerprint
→ [32,16]
→ finite
→ quaternion norm
→ Dex1 range
→ pelvis-frame workspace
→ 32/32 IK
→ joint limits
→ phase-aware collision
→ EEF/Joint command envelope
```

### 通过条件

所有将执行的 target 都通过。拒绝项必须记录明确 reason。

### 停止条件

任一 target 不可达、碰撞、越界或 fingerprint/contract 不匹配。

### 产物

```text
results/lgg100_real_chunk_preflight.json
```

---

## R8 · 同一真实 chunk 的 Single-chunk A/B

语义和 R7 都通过后才允许：

```bash
python lgg100_adaptive_ab.py \
  --smoke-report results/lgg100_vla_smoke_real.json \
  --chunk results/lgg100_action_chunk_real.npz \
  --phase grasp \
  --allow-sim-execution
```

### 公平性

```text
LGG100 只推理一次
同一 chunk SHA
同一 action samples
同一 MuJoCo 初始 state
同一 IK/controller/safety filters
只有 timestamps 不同
```

### 必须报告

- chunk 是否实际覆盖 far/near 区域；
- far/near 的 scale 分布；
- duration；
- endpoint/tracking error；
- command 和 actual v/a/jerk；
- collision/contact；
- path geometry 是否完全相同。

### 通过条件

- far 覆盖时确实出现 `scale > 1`；
- near/approach/grasp/place 覆盖时确实出现 `scale < 1`；
- command limits 100% 通过；
- actual jerk、endpoint 和 forbidden contact 不恶化；
- 若该 chunk 没覆盖 far 或 near，结论必须是“coverage 不足”，不能判 module 成功。

### 产物

```text
results/lgg100_adaptive_ab.json
```

Single-chunk 只证明 retiming mechanics，不证明叠积木成功率。

---

# Phase D — 真正 VLA 闭环和任务级判定

## R9 · 先跑 Adaptive OFF 的 LGG100 闭环

必须实现 `lgg100_closed_loop_eval.py`。

循环：

```text
render current 3 RGB + read state
→ LGG100 inference
→ semantic/age/preflight gate
→ execute conservative stride
→ new observation
→ replan
```

第一阶段：

```text
adaptive_enabled=false
scale=1.0 或保守固定低速
```

### 通过条件

- multi-chunk 连续，无 frame/quat/normalization 跳变；
- stale/timeout 会 hold；
- 至少完成无随机化开发 trials；
- 每次 inference、chunk、执行 stride 和状态都有 timestamp/hash；
- 无 forbidden collision，失败可 replay。

如果 Baseline LGG100 闭环本身不会叠积木，不允许用 Adaptive 掩盖 policy 问题。

### 产物

```text
results/lgg100_closed_loop_baseline.json
results/lgg100_closed_loop_replays/
```

---

## R10 · 真正 LGG100 + Module 的任务级配对 A/B

### 两层测试

```text
开发：至少 30 个配对 trials
正式：建议至少 100 个配对随机化 trials
```

每个 pair 使用相同：

- cube 初始位置/质量/摩擦；
- lighting/camera perturbation；
- network latency/jitter profile；
- MuJoCo 初始 state；
- prompt；
- model sampling seed（如果 OpenPI 暴露）；
- 软件 revision 和 contract SHA。

注意：multi-chunk 闭环中两个分支状态会分叉，不能强行复用后续 chunk。公平方法是相同初始条件和随机 seed 的配对 trial，并记录每次独立 inference。单 chunk A/B 才复用完全相同的 chunk。

### 预注册指标

- full-stack success；
- red/blue/yellow 各阶段完成率；
- completion time P50/P95/P99；
- collision/drop rate；
- preflight rejection/hold rate；
- far/near/phase scale coverage；
- EEF tracking error；
- command 与 actual joint v/a/jerk；
- inference latency、observation age、stale rate。

### 初始工程验收阈值

这些是项目验收初值，不是 Unitree 官方限制：

- Adaptive success rate ≥ Baseline success rate − 5 个百分点；
- median completion time 至少改善 10%；
- P95 completion time 不恶化超过 5%；
- forbidden collision/drop 不增加；
- P95 actual joint jerk 不恶化超过 5%；
- command hard limits 100% 通过；
- 所有 timeout/stale 都 fail-closed。

任何一项不通过：

```text
production_adaptive_enabled=false
```

### 产物

```text
results/lgg100_adaptive_task_ab.json
results/lgg100_adaptive_task_replays/
```

---

## R11 · 随机化、延迟和故障注入

在 R10 通过后测试：

- 方块位置、质量、摩擦；
- 相机轻微位姿、曝光、光照和背景；
- inference latency、jitter、丢包、断线；
- stale chunk；
- IK near-limit；
- tracking error；
- 接触和方块滑落。

必须验证 Module 在不确定条件下首先减速/hold，而不是继续加速。

---

## R12 · 发布决策

只有以下全部为 true 才允许在 MuJoCo production profile 开启：

```text
real_lgg100_checkpoint_loaded=true
g1_contract_verified=true
g1_sim_eligible=true
baseline_closed_loop_passed=true
adaptive_local_phase_behavior_passed=true
adaptive_task_noninferior=true
adaptive_time_improvement_passed=true
safety_metrics_noninferior=true
production_adaptive_enabled=true
```

这仍然只允许 MuJoCo。G1 EDU 硬件继续要求：

```text
camera calibration
Unitree official limits
low-level feedback adapter
watchdog/hold/communication-loss tests
hardware E-stop
Shadow/HIL
operator sign-off
```

---

# 2. 下一次实际执行时我们按什么顺序做

拿到 Ubuntu NVIDIA SSH host 后，严格按以下顺序：

```text
1. R0 GPU 只读检查
2. R1 固定 OpenPI/HF/G1 contract
3. R2 strict real-weight restore
4. R3 SSH tunnel
5. R4 3-call probe
6. R4 30-call output-only
7. 审计真实 JSON/NPZ
8. 实现并运行 R5 semantic validation
9. 实现 R6 phase/clearance-aware module context
10. R7 真实 chunk 离线 preflight
11. R8 同一真实 chunk A/B
12. R9 Adaptive OFF 的 LGG100 multi-chunk closed loop
13. R10 30-trial development A/B
14. R10/R11 100-trial randomized formal A/B
15. R12 决定是否启用 module
```

当前立即需要用户提供的唯一远程资源是：

```text
Ubuntu NVIDIA 的 SSH alias 或 GPU_USER@GPU_HOST（以及非默认端口）
```

不要发送密码。
