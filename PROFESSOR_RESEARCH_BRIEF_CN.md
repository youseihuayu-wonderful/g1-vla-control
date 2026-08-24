# LGG100 × Unitree G1：基于安全门控自适应时间重参数化的双臂操作研究

**Research Overview & Experimental Framework**

**项目负责人：Shihua Yu**

**当前阶段：Simulation Validation**
**硬件状态：`g1_execution_enabled=false`；尚未执行真机动作**

---

## 1. 研究目标与核心问题

本研究面向 Unitree G1 双臂操作，研究在不修改 Vision-Language-Action（VLA）模型输出路径几何的前提下，能否通过安全门控的 Adaptive Speed Module 对动作时间进行重参数化，使机器人在自由空间阶段更快、在接近与接触阶段更保守，并最终缩短完整任务时间，同时不降低任务成功率或安全性。

研究对象为 LGG100 `pi05_g1_eef` checkpoint。模型输出为 32-step、16-D、pelvis-frame、absolute EEF action chunk。速度模块只能修改 action timestamps，不能修改 action samples，也不能绕过 contract、IK、swept-path collision、watchdog 或硬件安全限制。

研究目标不是证明“局部倍率可以超过 1.0”，而是建立以下完整证据链：

```text
冻结的数据契约
→ 真实 LGG100 输出与语义
→ G1 IK 与连续碰撞预检
→ Adaptive-OFF 稳定任务基线
→ Adaptive-ON/OFF 配对实验
→ 随机化与故障注入
→ Zero-motion Shadow/HIL
→ 受保护的低速真机验证
```

---

## 2. 核心研究问题

### RQ1：路径不变的时间重参数化能否产生真实任务收益？

当 LGG100 输出的 EEF action samples 完全保持 byte-identical，仅改变 timestamps 时，Adaptive-ON 是否能够显著降低完整抓取/堆叠时间？

### RQ2：局部速度决策能否同时利用任务阶段和安全状态？

phase、clearance、contact、EEF/gripper tracking error、observation age、policy age、IK/joint margin 与 pelvis stability 能否共同形成可审计、fail-closed 的速度决策？

### RQ3：局部“远快近慢”是否能够转化为端到端 speedup？

自由空间加速与接近/抓取减速在单 chunk 中成立后，是否仍能在 multi-chunk closed loop 中缩短总体完成时间，而不是因保守阶段、tracking lag 或重新规划增加总时长？

### RQ4：VLA 输出能否在 G1 上通过完整运动学与实时性 Gate？

32-step action chunk 能否连续通过双臂 IK、5 mm/3° 误差门限、joint limits 与 swept-path collision，并在 observation-to-commit ≤100 ms 的约束内提交？

### RQ5：Simulation 证据如何安全迁移到真实 G1？

如何通过 subscriber-only LowState、真实相机/EEF/Dex1 标定、zero-motion Shadow/HIL、watchdog 与 deterministic command adapter，避免将 MuJoCo 成功错误解释为硬件安全？

---

## 3. 核心学术贡献

以下贡献分为“已实现的方法贡献”和“仍需实验确认的经验结论”。

### 3.1 已实现的方法贡献

1. **Path-invariant Adaptive Retiming**
   建立只改变 timestamps、不改变 action samples 的 VLA action retimer，并使用 byte-identical 检查阻止路径漂移。

2. **多信号安全 Context**
   将任务 phase、EEF-to-object clearance、contact、tracking error、freshness、IK/joint margin 与 pelvis stability 纳入统一的 `AdaptiveSafetyContext`。

3. **Fail-closed 速度决策**
   对 NaN/Inf、未知 phase、network timeout、stale observation/policy、IK/collision/limits failure、低 clearance 和低 stability 强制 hold，而不是退化为默认高速。

4. **局部机制与任务级收益分层验证**
   将“局部远快近慢”“single-chunk duration 变化”“完整任务时间收益”作为三个独立 Gate，避免把局部倍率直接写成任务级 speedup。

5. **完整 observation-to-commit 实时指标**
   不只报告 neural inference latency，而是记录 render、inference、context、preflight、commit 与 observation age，防止用局部延迟掩盖 stale-action 风险。

6. **Simulation-to-Hardware 分阶段安全路径**
   定义 read-only inventory、subscriber-only LowState、物理标定、zero-motion Shadow/HIL、受保护确定性动作和任务闭环的递进式硬件协议。

### 3.2 仍需实验确认的经验贡献

1. Adaptive-ON 在相同 action、初态和随机 seed 下，能否比 Adaptive-OFF 更快；
2. 提速是否在成功率、碰撞率、接触风险和 tail latency 不劣于基线的条件下成立；
3. phase-aware retiming 是否能够跨 multi-chunk 连续覆盖 free-space → approach → grasp → lift/place；
4. Simulation 中的速度收益是否能够在真实 G1 的控制频率、反馈延迟与硬件限制下保持。

当前证据不能支持“任务级提速已经实现”或“Adaptive-ON 已具备真机资格”的结论。

---

## 4. 研究假设与主要评价指标

### H1：局部阶段行为

在安全 free-space 中，目标 scale 应大于 1.0；在 approach、grasp、place、contact 或低 clearance 中，scale 应受到保守上限约束。

### H2：任务级时间收益

在相同 action path、相同初态与相同随机 seed 下：

```text
Task Duration(Adaptive-ON) < Task Duration(Adaptive-OFF)
```

且任务成功率不下降。

### H3：安全非劣性

Adaptive-ON 不得增加：

- forbidden collision；
- 非预期 contact；
- IK/swept-path rejection；
- EEF/joint motion-envelope violation；
- stale-action commit；
- watchdog abort。

### 主要指标

| 指标类型 | 指标 | 当前用途 |
|---|---|---|
| 任务结果 | 完整抓取/堆叠成功率 | Adaptive-OFF 基线与 ON/OFF 主比较 |
| 速度 | 完整任务完成时间 | 主要 speedup estimand |
| 控制效率 | action chunks、committed prefixes、控制周期数 | 区分推理慢与路径效率低 |
| 实时性 | observation-to-commit P50/P95/P99 | 判断 stale-action 风险 |
| 运动学 | EEF position/orientation error、joint margin | IK Gate |
| 安全 | forbidden contact、collision、hold/abort reason | 非劣性与 fail-closed 验证 |
| 模块机制 | phase、clearance、scale profile、path identity | 解释速度决策机制 |

---

## 5. 研究系统架构

| 模块 | 功能 | 当前状态 |
|---|---|---|
| G1 Policy Contract | 冻结三相机、EEF-16、pelvis frame、xyzw、30 Hz、horizon 32 | 已完成 |
| MuJoCo Observation Bridge | 生成三路 RGB 与 FK-based EEF state | 已完成 |
| LGG100 Policy | 真实 checkpoint output-only 推理 | 离线通过 |
| Semantic Validator | 判断 absolute/delta、四元数顺序和左右手顺序 | 离线通过 |
| Adaptive Safety Context | phase/clearance/contact/tracking/freshness/margin/stability | 本地通过 |
| Context-aware Retimer | 只修改 timestamps，并执行 motion-envelope 限制 | 本地通过 |
| G1 Dual-arm IK | EEF target → 14 arm joints | 部分通过；完整路径待验证 |
| Swept-path Preflight | 连续路径 collision/joint/IK 检查 | 已实现；当前拒绝首个 closed-loop target |
| Quarantined Closed Loop | observation→policy→preflight→commit/hold | 已实现；任务未完成 |
| Unitree LowState Adapter | subscriber-only 真实状态读取 | 本地实现；真机尚未运行 |
| Hardware Command Adapter | 受保护动作发送 | 未实现且当前禁止 |

---

## 6. 当前已验证结果

| 项目 | 当前结果 | 证据边界 |
|---|---|---|
| LGG100 output shape | 50/50 为有限 `[32,16]` | 只证明输出结构，不证明任务成功 |
| Action semantics | `absolute_xyzw_lr` 获得最强支持 | 仍需真实 observation 与闭环任务验证 |
| Warm inference | P50 约 80.52 ms | 不等于完整控制链延迟 |
| IK 30 iterations | 姿态误差约 6.89°/6.29°，未过 3° Gate | 正确执行 hold |
| IK 60 iterations | 姿态误差约 1.74°/1.59°，位置误差 <1 mm | 只覆盖 sampled targets |
| IK 参数扫描 | 15 组候选通过两个 sampled targets | 未覆盖完整 32-step/multi-chunk |
| Near speed behavior | 32/32 approach，scale=0.50× | 局部保守减速成立 |
| Far speed behavior | single-chunk duration 改善 6.90% | 不是完整任务 speedup |
| Mixed transition | 表面 duration 改善 25.44%，但 Gate 失败 | 缺少 grasp phase，不得宣称有效 |
| Deterministic fixture | Far median 1.30×、Near 0.50× | 整体反而比 baseline 慢 0.422 s |
| Context measurement | 本地 median 11.96→0.61 ms | L40S 端完整链尚未复测 |
| Adaptive-OFF closed loop | `completed_cycles=0`、`task_success=false` | 未建立任务基线 |
| Complete commit age | 119.94 ms，Gate 为 100 ms | 当前超时 19.94 ms |
| Hardware execution | `hardware_execution_performed=false` | 未执行任何真机动作 |

---

## 7. Adaptive Speed Module：已做到什么程度

### 7.1 已验证

- action samples 在 retiming 前后保持 byte-identical；
- safe free-space 允许 scale >1.0；
- approach 上限 0.70×；
- grasp/place 与 contact/low-clearance 场景限制为 ≤0.50×；
- lift 上限 0.80×，retreat 上限 1.00×；
- scale increase 限制为每秒 2.0；
- EEF speed/acceleration/jerk、angular speed 与 gripper speed 均进入 simulation envelope；
- stale、timeout、non-finite、IK/collision/limits failure 全部 fail-closed hold；
- MuJoCo context 可测量 clearance、contact、joint margin 与 pelvis stability；
- action-independent scene measurement 已改为每 chunk 共享 snapshot。

### 7.2 部分通过

- 真实 LGG100 Near/Far single-chunk 行为通过；
- Mixed transition coverage 未通过；
- deterministic far-to-near fixture 证明局部远快近慢，但未产生总体时间收益；
- motion envelope 来自 simulation/training-target P99，不是官方 G1 hardware limits。

### 7.3 尚未完成

- Adaptive-OFF 稳定抓取/堆叠基线；
- multi-chunk free-space→approach→grasp→lift/place 连续覆盖；
- 同 action、同初态、同 seed 的 Adaptive-OFF/ON 配对；
- 多 seed 成功率和时间统计；
- 接触力、碰撞、tail latency 与 recovery 非劣性；
- 真实 G1 controller rate、limits 与反馈下的速度验证。

---

## 8. 当前主要阻塞项

1. **S4 IK 与 Swept Path**：尚缺完整 32-step、多 chunk、连续性、随机目标和碰撞负例回归；
2. **S5 Adaptive-OFF**：cycle 0 被 preflight 拒绝，没有形成稳定任务基线；
3. **S6 实时性**：119.94 ms 超过 100 ms Gate；
4. **S7 随机化与故障注入**：尚未完成多 seed robustness suite；
5. **L40S 资源**：2026-08-23 已恢复认证 shell；8 张卡每卡约有 11.0 GiB free，但均有约 34.2 GiB resident workload，仍需完全空闲或明确分配的 GPU；
6. **物理标定**：真实三相机、EEF、Dex1、桌面和方块尚未同步标定；
7. **硬件安全链**：官方 limits、watchdog、E-stop、balance/stance 与 command adapter 未完成。

---

## 9. 下一阶段实验计划

```text
P1  完整 32-step sequential IK + swept-path regression
P2  已知 collision/unreachable 负例与随机可达目标
P3  Fresh-action watchdog 与 fault injection
P4  Adaptive-OFF 多轮 MuJoCo closed loop
P5  observation-to-commit ≤100 ms profile
P6  相同 action/初态/seed 的 OFF/ON paired evaluation
P7  camera/object/friction/jitter/stale randomization
P8  subscriber-only LowState 与物理标定
P9  zero-motion Shadow/HIL
P10 所有 Gate 通过后，单独评审低速真机动作
```

Adaptive-ON 只有在以下条件同时满足后才可进入下一阶段：

```text
Adaptive-OFF 稳定成功
AND Mixed transition coverage passed
AND Task duration improved
AND Success/safety non-inferior
AND Commit age ≤100 ms
AND Watchdog validated with fresh committed actions
```

---

## 10. 真机阶段与安全边界

真机阶段按以下顺序递进：

1. Read-only inventory；
2. Subscriber-only LowState；
3. 真实相机/EEF/Dex1/场景标定；
4. LowState→FK→current-pose IK 的 zero-motion Shadow；
5. watchdog、E-stop、communication-loss 与 hardware limits 验证；
6. deterministic command adapter 独立审查；
7. 受保护低速单臂动作；
8. 双臂、桌面、轻物体、Adaptive-OFF；
9. Adaptive-ON 最后评估。

当前只允许 Simulation、read-only、subscriber-only 与 zero-motion 工作。任何缺失证据、timeout、stale state、IK/collision failure 或 mode mismatch 都必须触发 hold。

---

## 11. 预期研究输出

1. 一套版本化 G1 VLA EEF action contract；
2. 一套 path-invariant、context-aware、fail-closed Adaptive Retimer；
3. 一套区分局部速度行为与任务级 speedup 的评测方法；
4. Adaptive-OFF/ON paired benchmark 与多 seed robustness profile；
5. observation-to-commit latency decomposition；
6. Simulation→Shadow/HIL→hardware 的安全部署协议；
7. 对“何时 VLA manipulation 可以安全加速、何时必须减速或 hold”的可复核实验结论。

---

## 12. 当前结论

当前工作已经证明：

- LGG100 能稳定输出符合基础 G1 EEF contract 的 action chunk；
- Adaptive Speed Module 能保持 action path 不变，并根据 phase 与安全 context 做出局部远快近慢决策；
- Near/Far single-chunk 机制具备真实 LGG100 evidence；
- fail-closed safety boundary 能在 IK/preflight 失败时阻止动作。

当前工作尚未证明：

- Adaptive-OFF 能稳定完成抓取/堆叠；
- Adaptive-ON 能缩短完整任务时间；
- 速度收益在多 seed 下具有统计稳定性；
- 该系统已经具备真机动作资格。

因此，现阶段最准确的研究结论是：**速度模块的局部机制与安全不变量已得到验证，但完整任务提速、实时闭环和硬件可部署性仍处于待验证阶段。**
