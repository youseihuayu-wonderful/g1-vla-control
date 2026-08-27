# G1 VLA 速度优化与真机验证计划

**负责人：** Shihua Yu

**目标：** 冻结 Yuhao/LGG100 checkpoint 和 action samples，仅通过部署计算、phase-aware timestamps 和安全调度，让完整任务在成功率与安全不下降的前提下快于 Yuhao 固定 15 Hz baseline。

**GitHub 分支：** `work/lgg100-semantic-speed-gates`

## 永久保留的方法

```text
MuJoCo：继续用于算法优化和安全筛选
真机只读Shadow：联网后尽快开始
真机动作：H1–H5全部通过后才开始
任务速度结论：最终必须以真机paired A/B为准
```

最终验证用真机更有说服力，但不放弃 MuJoCo，也不直接跳到真机运动。

## 今天已经完成、应推送 GitHub 的工作

### S1 · 单 fixture timestamp-only 参数搜索

- 搜索 270 组 near/far 参数；前 12 个进入完整 MuJoCo dynamics；
- 候选：Near `0.95×`、Far `1.65×`、Near threshold `1.5 cm`、Far threshold `8 cm`；
- deterministic 17 cm fixture：`4.068 s → 3.602 s`；
- duration reduction `11.46%`，effective speedup `12.94%`；
- action samples byte-identical；endpoint regression `+1.822 mm`；
- 只证明单一 deterministic fixture，不是 LGG100 任务结论。

证据：`results/g1_phase_scale_optimizer_20260827.json`

### S2 · Fast sequential IK + swept collision microbenchmark

- Legacy 定义：项目 MuJoCo fixed `250 iterations/target`，不是 Yuhao Pinocchio；
- Fast：warm start、最多 60 iterations、内部 early-stop `4 mm/2.5°`；
- 外部正式 Gate 仍为 `5 mm/3°`；
- arm/finger swept interpolation 仍为 `0.01 rad/0.001 m`；
- 50 个 interleaved pairs，5 次 warmup；
- Legacy P50 `1134.21 ms`，Fast P50 `35.01 ms`；
- Fast P95 `36.73 ms`，小于 333 ms prefetch window；
- paired median speedup `32.47×`，95% CI `[32.29, 32.55]×`；
- 50/50 两分支均接受，32 targets / 146 configurations 相同。

证据：`results/g1_fast_preflight_paired_benchmark_20260827.json`

### S3 · Zero-waist → measured-waist adapter

- canonical policy action/hash 保持不变；
- 生成独立 kinematic IK target；
- uncompensated target 在 60 iterations 后 residual 约 `6.19 mm`，拒绝；
- compensated 32/32 hold targets 在 0 iterations 内通过；
- mapping error约 `1.2e-8 m / 3.42e-6°`；
- physical EEF parity 仍未验证。

证据：`results/g1_waist_compensation_diagnostic_20260827.json`

### S4 · 五距离 development paired A/B

- 五个 lift offsets：10、13、15、17、19 cm；
- OFF/ON action hashes 全部相同，只改变 timestamps；
- duration reduction 范围 `9.14%–11.85%`；
- effective speedup 范围 `10.06%–13.44%`；
- 只有 2/5 同时通过时间、endpoint 和 jerk 非劣标准；
- 当前单 fixture 候选没有泛化，`production_adaptive_enabled=false`。

证据：`results/g1_deterministic_speed_paired_ab_20260827.json`

### S5 · 稳定完成时间与Fast正确性corpus

旧 `simulated_duration = retimed path + 固定2秒` 只能作为schedule proxy，不能作为任务完成时间。后续速度搜索冻结为：最终action生效后，双EEF在原始 `5 mm/3°` Gate内连续保持250 ms，最多等待3秒。

- 当前候选在五距离稳定完成测试中仅1/5通过；
- stable duration reduction范围 `-34.19%–+18.68%`，median仅 `+2.42%`；
- 因此此前单fixture `11.46%` 不升级为任务速度结论；
- Fast correctness development corpus：7/7 verdict cases、3/3 schema faults通过；
- Legacy/Fast acceptance concordance `100%`，危险Fast accept为0；
- NaN/Inf action和NaN timestamp现在在`EEFActionChunk`边界立即拒绝。

证据：

- `results/g1_stable_completion_validation_20260827.json`
- `results/g1_fast_preflight_correctness_corpus_20260827.json`

### S6 · Yuhao/项目分层计算对照

相同zero-waist 32-target输入，50个pairs：

- Yuhao Pinocchio IK-only P50 `2.71 ms`、P95 `3.59 ms`，全部通过项目外层 `5 mm/3°`；
- 项目Fast MuJoCo IK-only P50 `24.49 ms`、P95 `35.31 ms`，全部通过；
- 已链接完整Fast IK+swept P50 `35.01 ms`、P95 `36.73 ms`；
- Yuhao层不包含swept collision，且其生产loop每tick选择目标，因此不报告Yuhao IK-only到项目完整preflight的总速度比；
- 这说明Yuhao原始Pinocchio IK本身很快，项目Fast工作的价值是把新增的32-step完整安全preflight压进333 ms，而不是声称IK比Yuhao快。

证据：`results/g1_yuhao_fast_layered_benchmark_20260827.json`

### S7 · Waist-compensated离线Shadow集成

已把zero-waist canonical policy action与measured-waist kinematic IK target分离接入完整离线dataflow：

- canonical action adapter前后SHA-256相同，输入未被修改；
- compensated hold IK 32/32通过、0 iterations、最大位置误差约 `7.1e-9 m`；
- Fast swept fail-closed拒绝保存姿态的`initial_configuration_collision`；
- mock sink 32/32均hold，原因为`collision_failure`和`waist_divergence`；
- synthetic cameras、deterministic hold policy、缺失腿关节仍不能升级为真实Shadow。

证据：`results/g1_offline_compensated_shadow_replay_20260827.json`

### S8 · Stable-completion多场景候选筛选

- 生成811组timestamp-only参数；
- 463组理论上可能在五距离均达到10%；
- 从maximin理论排序中确定性覆盖30组，并保留可行历史anchor；
- 先在13 cm瓶颈执行真实dynamics + continuous-250-ms Gate；
- 0/30达到10%，观察到的最大stable reduction仅 `2.11%`；
- 因瓶颈无通过项，没有候选进入完整五距离评估；搜索不是穷举。

因此当前distance-only Near/Far retimer没有合格的多场景stable candidate，保持`production_adaptive_enabled=false`。13 cm诊断进一步表明：

- 旧候选减少path时间 `0.423 s`，但settling增加 `1.297 s`，最终反而慢 `34.19%`；
- 30-screen内最佳候选减少path时间 `0.252 s`，settling增加 `0.198 s`，净完成时间只降低 `2.11%`；
- 瓶颈不是schedule长度本身，而是跟踪误差和settling。

下一步不能继续追逐schedule缩短，需要研究控制跟踪/settling与更细的phase-aware策略，并最终在真实LGG100 chunks上验证。

证据：

- `results/g1_stable_completion_optimizer_20260827.json`
- `results/g1_settling_bottleneck_diagnostic_20260827.json`

### S9 · Fast preflight 30-trajectory corpus

扩展为30条确定性trajectory：18条0–20 cm reachable vertical、8条unreachable、3条gripper transition、1条保存姿态initial collision。

- 30/30 Legacy/Fast预期判定通过；
- acceptance concordance `100%`；
- dangerous Fast accept `0`；
- Fast P50 `15.33 ms`、P95 `37.55 ms`；
- randomized near-limit与真实完整29关节collision仍未覆盖。

证据：`results/g1_fast_preflight_30_trajectory_corpus_20260827.json`

## 仍需继续的离线工作

### O1 · 多场景鲁棒 scale 优化

不能继续使用只对17 cm最优的参数。第一批30候选瓶颈筛选已完成且0/30通过；搜索并非穷举。下一轮必须使用冻结的 `5 mm/3° continuous 250 ms hold`，并从简单distance-only Near/Far升级为控制跟踪与语义phase共同优化：

- minimum 30完整多场景 deterministic/randomized pairs；
- Far/Approach/Near/Grasp/Lift/Place/Retreat 独立 scale与settling-aware目标；
- 所有 pair stable completion duration reduction `≥10%`；
- endpoint regression `≤5 mm`；
- jerk/contact/limits 非劣；
- action samples byte-identical。

### O2 · 多场景 IK/collision 正确性 corpus

30条确定性trajectory以及NaN/Inf schema faults已通过。仍需增加randomized near-limit、collision-boundary fuzzing和完整29关节真实姿态。继续要求危险Fast accept为0、accepted residual在内部 `4 mm/2.5°`、fault全部hold、P95小于333 ms。

### O3 · Yuhao生产loop和真实计算机复测

离线同输入三层对照已经完成；仍需在部署计算机复测，并区分Yuhao实际selected-tick生产loop与项目32-step安全preflight。不能把Yuhao缺少collision Gate当成公平的总延迟。

### O4 · 真实 Shadow harness 集成

waist compensation、Fast IK、swept path和mock sink的离线fixture集成已经完成；仍需接入真实streaming LowState、三相机freshness、DDS watchdog和冻结LGG100 inference。全过程仍无Publisher。

## 什么时候需要真机器

| 阶段 | 是否需要真机 | 是否允许运动 | 目的 |
|---|---|---|---|
| 当前 O1–O3 | 否 | 否 | MuJoCo优化、正确性corpus、计算benchmark |
| H1/H3只读采集 | 是，联网后立即 | 否 | 完整29关节、三相机、instrumented DDS |
| 真实计算benchmark | 是，只用机器人计算机 | 否 | 测量真实CPU上的Pinocchio/Fast preflight latency |
| 真实15 Hz Policy Shadow | 是 | 否 | LowState+相机→LGG100→IK→mock sink |
| 第一次free-space动作 | 是 | 仅H1–H5全部通过后单独审批 | 支撑、单臂、极低速、小位移、无接触 |
| 任务级paired A/B | 是，最后 | 经独立审批 | 最终证明相对Yuhao的真实任务速度 |

### 联网后的第一次真机工作

只读执行：

1. capture完整29关节，替换当前缺失的12个腿部关节；
2. 重判 Dex1/wrist 与 hip 的初始碰撞；
3. 读取 head-left、left wrist、right wrist；
4. instrument Reader exception、tick和monotonic gap；
5. 在机器人计算机运行无Publisher latency benchmark。

输出必须继续满足：

```text
publisher_created=false
robot_command_sent=false
motion_authorized=false
```

## 真机动作的必要条件

H1–H5 必须全部通过：真实相机、physical EEF/frame、完整29关节碰撞、DDS freshness、真实15 Hz Shadow、Fast IK/swept corpus、integrated fault injection、官方硬件limits和独立adapter审核。

任何一项失败，真机动作保持关闭。

## 最终速度结论的唯一合格形式

```text
Adaptive-OFF：Yuhao fixed 15 Hz
Adaptive-ON：相同checkpoint、observation、raw/canonical actions与初态，只改变timestamps
```

至少报告30个开发pairs和预注册正式pairs的成功率、P50/P95/P99任务时间、endpoint、掉落、碰撞、hold、jerk与tail latency。只有真机paired A/B满足完整任务时间降低且安全/成功率非劣，才能声称“比Yuhao快”。
