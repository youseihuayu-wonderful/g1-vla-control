# LGG100 冻结 Checkpoint × Adaptive Speed A/B GPU 执行计划

**版本：** 1.1

**作者：** Shihua Yu

**状态：** Q0 已完成；独立 Q0.5 四小时多场景 output-only endurance job 正在运行

**范围：** 仅 Simulation / output-only inference；不训练 VLA、不发送机器人动作

## 1. 研究目标

在完全冻结 Hugging Face LGG100 `pi05_g1_eef` checkpoint 的前提下，验证 Adaptive Speed Module 是否能在不改变单个输入 chunk 内 action samples 的条件下：

1. 实现预期的 Near / Far / Mixed 阶段速度行为；
2. 缩短完整 MuJoCo 任务时间；
3. 不降低任务成功率；
4. 不增加 IK、碰撞、tracking、stale-action、watchdog 或 hold failure；
5. 保持完整 observation-to-commit 延迟不高于 100 ms。

这不是神经网络训练任务。GPU 仅用于恢复现成 checkpoint 和执行 VLA inference。

## 2. A/B 条件的冻结定义

### A：Adaptive-OFF

- checkpoint、revision、模型参数和 normalization 全部冻结；
- VLA 输出 `[32,16]` action samples；
- 使用已确认且在 manifest 中显式冻结的 15 Hz 均匀 timestamps（相邻 sample `1/15 s`）；
- Adaptive Retimer 不修改 timestamps；
- contract、IK、swept-path、limits、watchdog 和 hold Gate 始终启用。

### B：Adaptive-ON

- 使用与 A 完全相同的 checkpoint、revision、模型配置和 observation schema；
- VLA 输出同一格式的 `[32,16]` action samples；
- Adaptive Retimer 只能根据受控 context 修改 timestamps；
- Adaptive Retimer 不得修改 action samples；
- Adaptive Retimer 不得绕过任何安全 Gate。

### 重要因果边界

正式 cadence 已由项目负责人确认为 15 Hz。每个 OFF/ON trial manifest 都必须保存 `control_hz=15.0`、`exec_steps=0`、`prefetch_lead_steps=5`、`blend_steps=5` 和 `time_alignment=true`，不得依赖 CLI help text 或隐式 default。

在 single-chunk 测试中，A/B 必须消费同一个保存 chunk，并证明 action samples byte-identical。

在完整 closed-loop 中，A/B 在第一步之后可能产生不同的 MuJoCo state，因此后续 observations 和 VLA chunks 可以自然分化。完整闭环不得宣称所有后续 chunks byte-identical；应验证的是：对每个实际收到的 chunk，Retimer 输入和输出 action samples 完全一致，只有 timestamps 改变。

## 3. 系统分工

### 远端分配的 L40S

仅执行：

1. checkpoint strict restore；
2. JAX compile；
3. output-only VLA inference；
4. server metadata、checkpoint hash、peak VRAM 和 inference timing 记录。

### 本机

执行：

1. MuJoCo observation 生成；
2. 三路图像和 state schema validation；
3. 通过 SSH tunnel 请求远端 VLA；
4. action chunk contract validation；
5. Adaptive-OFF 或 Adaptive-ON timestamp 处理；
6. sequential IK、swept-path、limits、collision 和 watchdog；
7. MuJoCo dynamics 与可视化；
8. trial logging、统计和报告生成。

### 禁止项

- 不训练或更新 VLA 权重；
- 不用 dummy process、`sleep` 或无意义显存分配占卡；
- 不在未被 Slurm 分配的 GPU 上启动服务；
- 不停止、暂停或挤占 resident workload；
- 不开放公网 policy port；
- 不发送任何真实机器人动作。

## 4. GPU 资源计划

“一个 GPU core”在本计划中指一张完整 L40S device，不是 CUDA core，也不使用未验证的共享显存。

| 阶段 | GPU 请求 | 最大 wall time | 用途 | 提前释放条件 |
|---|---:|---:|---|---|
| Q0 资格检查 | 1 × L40S | 1 小时 | strict restore、30-call output-only、peak VRAM | 完成或任一 Gate 失败 |
| Q1 Pilot A/B | 1 × L40S | 4 小时 | OFF baseline、single-chunk paired test、小规模 closed-loop | 完成、失败或无进展 watchdog |
| Q2 Formal A/B | 最多 2 × L40S；每个 array task 1 张 | 每 task 最多 6 小时 | 独立 seed shard；平衡 A/B 顺序 | shard 完成或任一安全 Gate 失败 |

### 资源决策

1. 立即阶段只申请 **1 张 L40S**；
2. 同一 policy server 顺序执行 A/B，避免设备差异成为混杂因素；
3. 只有 Q1 Pilot 通过后才允许 Q2；
4. Q2 最多并行 2 个独立 shard，每个 shard 仍只使用 1 张 GPU；
5. 不申请 4 张或 8 张 GPU，因为当前 inference A/B 不需要模型并行训练；
6. wall time 是上限，不是必须占满的时长。

## 5. Slurm 获取与通知工作流

### 作业约束

```bash
#SBATCH --partition=all
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:00:00   # Q0；Q1 改为 04:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --signal=B:TERM@120
```

作业必须：

1. 验证 `CUDA_VISIBLE_DEVICES` 只暴露一张由 Slurm 分配的 GPU；
2. 记录 allocation metadata，但公开报告不得包含内部节点、用户或网络 identity；
3. 不接受硬编码 hostname 或物理 GPU index；
4. 将 policy server 绑定到受控接口，并仅通过 SSH tunnel 使用；
5. 写入原子状态 JSON：`QUEUED → ALLOCATED → RESTORING → READY → RUNNING → COMPLETE/FAILED`；
6. 在 `ALLOCATED` 和 `READY` 时触发本地 Herdr/macOS notification；
7. 退出时只停止由当前 job 创建且经过 PID、PGID、command 和 job-id 验证的进程；
8. 任何失败立即释放 allocation。

### 不使用 cron 抢卡

Slurm 自身负责等待可用 GPU。不得使用 cron 轮询后抢占设备，也不得在 scheduler 外创建占卡进程。

## 6. 分阶段执行 Gate

### G0：脚本可移植化

必须完成：

- 删除硬编码 hostname；
- 删除硬编码 GPU index；
- 从 `CUDA_VISIBLE_DEVICES` 和 Slurm metadata 获取设备；
- 使用独立 port、PID、log、result 和 status；
- 实现 signal/timeout cleanup；
- static test 证明不存在 hardware publisher 或机器人命令。

通过后权限：可以提交 Q0，但不能运行 MuJoCo action。

### G1：GPU 与 checkpoint 资格

检查：

- GPU 由当前 Slurm job 独占分配；
- CUDA/JAX backend 正确，不允许 CPU fallback；
- checkpoint revision、文件树、metadata 和 normalization hash 一致；
- strict restore 不忽略 missing/extra parameter；
- 记录 compile time、steady-state VRAM 和 peak VRAM。

停止条件：OOM、hash drift、CPU fallback、额外参数被静默忽略或 GPU allocation 不明确。

### G2：Output-only inference

执行：

- 3-call smoke；
- 30-call正式 probe；
- 验证 30/30 finite `[32,16]`；
- 验证 quaternion、frame、left/right order 和 range；
- 记录 inference P50/P95/P99；
- 保存 fingerprinted chunk artifact。

通过后权限：可以进入本机 IK/preflight；仍不能提交 MuJoCo dynamics。

### G3：完整 sequential preflight

针对 Near、Far、Mixed 和多个连续 chunks：

- 32-step sequential IK；
- 5 mm / 3° threshold；
- swept-path interpolation；
- joint limits；
- self/environment collision；
- continuous target transition；
- action freshness。

当前已知缺口：现有候选尚未获得完整 32-step、多 chunk 资格。

### G4：Adaptive-OFF baseline

先执行 5-seed pilot，再决定是否扩展：

- Adaptive Retimer 完全关闭；
- 每个 replan 请求真实 VLA；
- fresh-action watchdog 必须观察到真实 commit；
- 记录 completed cycles、task success、abort reason；
- observation-to-commit P50/P95/P99 ≤ 100 ms；
- 不允许把 zero-cycle、stale hold 或 paused diagnostic 记为成功。

当前已知阻塞：`swept_path_preflight_failed`、`completed_cycles=0`、`task_success=false`、commit age 119.94 ms。

### G5：Single-chunk paired causal test

同一保存 chunk 分别进入 OFF 和 ON：

- action samples byte-identical；
- 只有 timestamps 可以不同；
- Near 目标约 0.50×；
- Far 目标不超过已注册上限；
- Mixed 必须包含 free-space → precision transition；
- 任一 safety context 不完整时进入 hold，不允许加速。

### G6：Adaptive-ON closed-loop pilot

只有 G4 和 G5 全部通过后执行：

- 5 个 paired seeds × Near/Far/Mixed；
- A/B 顺序使用 AB/BA 平衡；
- 同一 seed、初始 state、checkpoint、prompt、预处理和随机化参数；
- 记录每个 replan 的 observation hash、chunk hash、context、timestamps 和 Gate 决策。

### G7：Formal A/B

只有 pilot 预注册指标通过后扩展：

- 每个场景至少 30 个 paired seeds；
- 最多 2 个 Slurm array shards 并行；
- shard 与 A/B 顺序平衡；
- 不因中间结果更改主要 endpoint、threshold 或排除规则；
- 所有失败 trial 保留并进入统计。

## 7. 指标与统计

### 主要指标

1. **Task success rate**：B 不得低于 A 的预注册非劣界；
2. **完整任务时间**：不能用 single-chunk duration 代替；
3. **安全失败**：forbidden collision、limit、watchdog 和 stale commit 不得增加；
4. **完整延迟**：observation-to-commit P50/P95/P99，P99 和最大值单独报告。

### 次要指标

- policy inference latency；
- tunnel round-trip；
- IK time；
- collision/preflight time；
- hold count 和原因；
- Near/Far/Mixed coverage；
- tracking error；
- timestamp monotonicity；
- Retimer action-sample identity；
- GPU peak VRAM、utilization 和 server restart count。

### 统计原则

- paired analysis 优先；
- 同时报告 effect size、置信区间和原始 trial 数；
- 成功 trial 的 duration 与失败率分开报告，避免 survivor bias；
- 增加 capped-time / restricted-mean completion-time 结果；
- local mechanism、single-chunk、完整任务和硬件资格分层报告；
- 不把瞬时 GPU utilization=0% 解释为 GPU 空闲。

## 8. 预注册通过标准

Adaptive-ON 只有同时满足以下条件才可标记为 Simulation-qualified：

1. Adaptive-OFF 至少完成预注册数量的稳定闭环；
2. OFF 和 ON 使用同一冻结 checkpoint；
3. single-chunk action samples byte-identical；
4. Mixed transition coverage 通过；
5. B 的任务成功率不劣于 A；
6. B 的完整任务时间具有正向改善和可报告置信区间；
7. collision、limits、stale、watchdog 和 tracking 不劣；
8. observation-to-commit ≤ 100 ms；
9. 无未解释 exclusion、server restart 或数据丢失；
10. `g1_execution_enabled=false` 和 `hardware_execution_performed=false` 保持不变。

任一项失败：Adaptive-ON 继续锁定，不升级硬件 Gate。

## 9. 结果数据契约

每个 trial 至少保存：

```json
{
  "schema_version": "lgg100_adaptive_ab_trial_v1",
  "condition": "OFF_or_ON",
  "scenario": "near_far_or_mixed",
  "seed": 0,
  "checkpoint_revision": "pinned",
  "contract_sha256": "pinned",
  "control_hz": 15.0,
  "exec_steps": 0,
  "prefetch_lead_steps": 5,
  "blend_steps": 5,
  "time_alignment": true,
  "initial_state_sha256": "...",
  "observations": [],
  "chunk_fingerprints": [],
  "retimer_action_identity_passed": true,
  "task_success": false,
  "task_duration_s": null,
  "abort_reason": null,
  "latency_ms": {},
  "safety": {},
  "gpu": {},
  "g1_execution_enabled": false,
  "hardware_execution_performed": false
}
```

汇总文件必须包含：

- 计划版本与预注册 hash；
- 完整 trial manifest；
- 所有 included/excluded trial 及原因；
- A/B effect sizes 和 confidence intervals；
- Gate decision；
- 代码 commit、checkpoint revision、contract hash 和结果 hash。

不得包含内网地址、SSH identity、hostname、本地路径、Token 或凭据。

## 10. React 结果应用

最终生成一个静态构建、可公开脱敏的 React app。React app 只读取版本化 JSON，不直接连接 GPU、SSH、机器人或内部 API。

### 页面结构

1. **Executive Status**：当前 Gate、最终结论、作者 Shihua Yu；
2. **Workflow**：本机 ↔ SSH tunnel ↔ L40S ↔ 本机 MuJoCo 数据流；
3. **GPU Allocations**：Q0/Q1/Q2 请求、实际时长、peak VRAM、退出原因；
4. **Experiment Matrix**：场景、seed、AB/BA 顺序和完成度；
5. **A/B Results**：任务成功率、完整任务时间和 paired effect；
6. **Latency**：policy、network、IK、safety 和 commit P50/P95/P99；
7. **Safety**：collision、limits、hold、watchdog、stale 和 tracking；
8. **Trial Explorer**：纵向响应式 trial cards，不使用横向宽表；
9. **Evidence**：JSON/NPZ/report hashes、代码 commit 和预注册版本。

### UI 与公开要求

- React + TypeScript 静态构建；
- 桌面左侧导航，移动端横向导航；
- 不使用横向滚动宽表；
- 术语 hyperlink + hover/focus tooltip；
- 支持键盘、ARIA、reduced motion 和 print/PDF；
- CSP、HSTS、nosniff、no-referrer 和 restrictive permissions policy；
- 明确区分计划、进行中、通过、部分通过和阻塞；
- 不把 GPU allocation、局部速度倍率或 single-chunk 结果描述成任务成功；
- 所有公开内容必须脱敏。

## 11. 实施顺序

1. 冻结本计划并计算 SHA-256；
2. 将现有 GPU gate 改为 Slurm 可移植、fail-closed 版本；
3. 新增 allocation/status JSON 与本地 notification watcher；
4. 完成本地 32-step sequential IK + swept-path 回归；
5. 提交 Q0：1 × L40S、最多 1 小时；
6. 通过 G1/G2 后建立受控 tunnel；
7. 完成 G3/G4 Adaptive-OFF；
8. 完成 G5 single-chunk paired test；
9. 通过后提交 Q1：1 × L40S、最多 4 小时；
10. Pilot 通过后按需提交 Q2：最多 2 张 L40S；
11. 冻结 trial manifest 和统计结果；
12. 构建、测试、脱敏并部署 React results app；
13. 所有硬件 Gate 继续保持关闭。

## 12. 当前正式决策

- Q0 已使用 **1 张完整、Slurm 分配的 L40S** 并在 60 秒后正常释放；
- Formal 阶段并行上限：**2 张 L40S**；
- 当前 GPU 用途：**冻结 LGG100 checkpoint inference**；
- 当前不执行：训练、LoRA、checkpoint 更新、真机动作、GPU 抢占；
- 正式 deployment cadence 已固定为 **15 Hz**；
- 下一工程任务：**在 15 Hz manifest 下完成 G3 完整 sequential preflight**；
- 下一次 GPU 作业：只有完成前置 Gate 后才允许提交 **Q1 Adaptive-OFF Pilot**。

## 13. Q0 实际执行结果（2026-08-24）

Q0 最终在一张由 Slurm 正式分配、启动时无 resident compute process 的 L40S 上完成：

- 最大 wall time：1 小时；实际运行：60 秒并提前释放；
- GPU preflight：46,068 MiB total、45,458 MiB free；
- strict restore server：成功进入 ready；
- 3-call smoke：完成；
- 30-call formal：30/30 finite `[32,16]`；
- warm latency：P50 80.78 ms、P95 81.87 ms、max 82.48 ms；
- peak memory used：8,681 MiB；
- training、MuJoCo dynamics 和硬件动作：均未执行。

需要分层解释结果：

- 冻结 checkpoint output-only inference 通过；
- raw quaternion exact-unit 为 0/30；Yuhao 公开 consumer 要求在 IK 前归一化预测 quaternion；
- 官方 consumer post-processing 通过 30/30，raw quaternion norm error 范围为 0.00626–0.00776；
- 因此 quaternion gap 不再阻塞 Adaptive-OFF，`g1_contract_verified=true`；但 `g1_sim_eligible=false`，因为 sequential IK、swept-path、latency 和 closed-loop Gate 仍未通过。

前两次调度尝试在 GPU compute 开始前因 compute node 不共享 submission workspace 而失败。修复方式是在经过只读 GPU inventory 确认的空闲调度节点上，固定并 staging 代码、OpenPI 环境、checkpoint 和 observation；最终 bounded job 正常完成并释放。该问题属于 Slurm workspace portability，不是 checkpoint inference failure。

证据：`results/lgg100_slurm_q0_output_only_20260824.json`。

## 14. 独立 Q0.5 四小时复杂推理作业（运行中）

Q0.5 是与 Q0 分离的 Slurm job，目标是通过真实 inference 调查长时间稳定性、raw quaternion 分布和官方 consumer post-processing 可用率，不是用 dummy process 占用 GPU。

- GPU：1 张启动时无 resident compute process 的 L40S；
- target inference duration：14,400 秒；
- maximum wall time：15,600 秒，用于 strict restore、JAX compile 和 fail-closed cleanup；
- target inference rate：5 Hz；
- 场景：Near、Mixed、Far 按固定顺序轮换；
- 每次调用：记录 finite shape、raw exact-unit diagnostic、官方 consumer post-processing、quaternion norm error、latency 和 raw/canonical hash；
- 每 300 次调用：保存一个 quarantined action sample；
- 输出：压缩 per-call JSONL、sparse NPZ、场景级 P50/P95/P99/max、peak VRAM 和 cleanup evidence；
- 禁止：训练、MuJoCo dynamics、Adaptive-ON、真机动作和修改其他 GPU process。

启动后第一个公开 heartbeat：93.42 秒内完成 301 次调用，301/301 finite、0/301 raw exact-unit、301/301 官方 consumer post-processing 可用；allocated GPU memory used 8,665 MiB，瞬时 utilization 68%。这是由旧 runner 字段保存的运行中状态，最终同步时将进行无损字段迁移；作业将在完成四小时真实 inference 后正常释放，或在任一异常时提前 fail-closed。

证据：`results/lgg100_slurm_q05_soak_status_20260824.json`。
