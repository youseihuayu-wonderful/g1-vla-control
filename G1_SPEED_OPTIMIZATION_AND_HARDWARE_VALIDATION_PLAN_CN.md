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

## 仍需继续的离线工作

### O1 · 多场景鲁棒 scale 优化

不能继续使用只对17 cm最优的参数。下一轮将5个距离联合进入目标函数，并增加：

- minimum 30 deterministic/randomized pairs；
- Far/Approach/Near/Grasp/Lift/Place/Retreat 独立 scale；
- 所有 pair duration reduction `≥10%`；
- endpoint regression `≤5 mm`；
- jerk/contact/limits 非劣；
- action samples byte-identical。

### O2 · 多场景 IK/collision 正确性 corpus

增加 reachable、near-limit、unreachable、collision、gripper transition、NaN/Inf。要求：

- Fast 危险地接受 Legacy rejection 的次数为0；
- Fast residual 始终在内部 `4 mm/2.5°`；
- fault 全部 hold；
- P95 仍小于333 ms。

### O3 · Yuhao 三分支计算对照

```text
A：Yuhao Pinocchio IK-only
B：项目 Legacy MuJoCo IK+swept
C：项目 Fast MuJoCo IK+swept
```

必须分开报告 IK-only 与完整安全 preflight，不能把 Yuhao 缺少 collision Gate 当成公平的总延迟。

### O4 · 真实 Shadow harness 集成

将 waist compensation、Fast IK、swept path、DDS watchdog、三相机 freshness 和 mock sink 合并；仍无 Publisher。

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
