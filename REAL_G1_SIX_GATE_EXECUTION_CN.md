# 真实 G1 六阶段验证清单

**负责人：** Shihua Yu

**冻结顺序：** H1 → H2 → H3 → H4 → H5 → H6
**当前范围：** 真实机器人 read-only / zero-motion Shadow；禁止 Publisher 和运动命令

## 不可跳过的安全边界

- 操作员已确认 Damping、机械支撑和 E-stop 就绪；
- 本项目不调用 `ai` mode、MotionSwitcher 或任何模式切换；
- H1–H5 期间 `publisher_created=false`、`robot_command_sent=false`；
- 前一阶段未通过时，不得用后续结果覆盖或绕过失败；
- H6 只是第一次运动的独立评审，不代表自动授权。

## 当前总状态

| Gate | 问题 | 当前状态 | 当前证据 | 通过后权限 |
|---|---|---|---|---|
| H1 | 真实三相机 | **阻塞** | 离线 BGR/head-left/RGB224 boundary 已实现；image server 未运行，真实三帧仍为 `None` | 可建立真实视觉 observation；仍不可运动 |
| H2 | 真实关节 FK 对照 | **离线核心完成** | zero-waist 软件模型一致；真实 waist 造成最大 `12.12 mm / 2.93°` 偏差 | 可冻结双 FK view；物理 EEF parity 仍不可运动 |
| H3 | DDS freshness | **离线 forensic 完成** | 汇总 500 samples；候选 `20/50/200 ms`，根因和最终 threshold 仍待 live capture | 可预注册 watchdog；仍不可运动 |
| H4 | 15 Hz Policy Shadow | **离线 replay 完成，真实 Shadow 锁定** | 32-step plumbing 完成；IK residual、初始碰撞和 waist divergence 均正确 hold | 可保存离线建议动作；仍不可运动 |
| H5 | Watchdog + fail-closed mock adapter | **离线 8/8，通过但未集成** | 所有 fault 均 `publish_allowed=false`、`hold=true`、`robot_command_sent=false` | 仍须绑定真实 H4；不自动运动 |
| H6 | 第一次真机动作评审 | **离线 checklist 完成，评审锁定** | 所有签字、limits 和 arm token 均保持未通过 | 仅决定是否允许单次、低速、free-space 试验 |

当前真实 Gate 完整解决数仍为 **0/6**；计划的离线工作包为 **5/5 完成**。离线完成不得提升 live Gate 或动作权限。

## H1 · 真实三相机

### 输入

- Yuhao `g1_client/camera_client.py` 作为协议参考；
- teleimager image server；
- head-left、left wrist、right wrist。

### 必须证明

1. 三路 frame 均非空、finite、带独立 SHA-256；
2. 明确源颜色为 BGR，并在 policy boundary 转成 RGB；
3. 明确源尺寸、head 是否 binocular、left-half 选择；
4. 保存每路接收 timestamp 和顺序采集跨度；
5. 转换成冻结的三路 `224×224 uint8 RGB` observation；
6. 不导入 Unitree DDS、不创建 Arm/Gripper Controller。

### 当前阻塞

`g1_camera_observation.py` 已离线实现 Yuhao 协议边界：BGR、binocular head-left 裁剪、BGR→RGB、`224×224 uint8`、三路 timestamp/hash 和顺序跨度验证。`results/g1_three_camera_readonly_probe_20260824.json` 仍证明实际 runtime 未启动；连接恢复后必须读取真实帧，不能用 synthetic fixture 通过 H1。

## H2 · Yuhao Pinocchio FK × 项目 MuJoCo FK

### 已完成

- 真实 waist 3 + arms 14 均存在且落在 MuJoCo model range；
- pelvis/world frame round-trip 最大 position error `6.05e-9 m`；
- perturbation-seeded current-pose IK 最终最大 position error `4.46e-5 m`；
- Yuhao Pinocchio 与项目 MuJoCo 在 `waist=0` 时最大误差约 `0.000000621 m / 0.0000836°`；
- measured waist 与 zero-waist EEF 最大差异约 `0.012116 m / 2.9294°`。

### 冻结决定

- Policy observation 使用 Yuhao zero-waist view，保持训练/部署 contract；
- safety、IK 和 swept geometry 使用 measured-waist view；
- 两个 view 的位置差异超过 `5 mm` 时必须 hold；当前样本触发 hold；
- 仍须物理测量 EEF site，不能只依赖两个软件模型互相同意。

## H3 · DDS freshness

### 离线 forensic

- 三批合计 500 samples、17 duplicate ticks、0 decreases；
- Reader errors 至少 26 次；最大 receive gap `3.98534 ms`；
- pinned SDK 用裸 `except` 打印通用错误，现有证据没有保存真实异常类型；
- pinned IDL 只定义 `tick:uint32`，没有定义 tick period/unit；
- duplicate tick 只能保守解释为“没有新的 source progress”，不能判定是 DDS 重复还是机器人 tick 语义。

### 候选 watchdog（PROVISIONAL）

- warning `20 ms`；stale hold `50 ms`；disconnect `200 ms`；
- stale 后要求连续 2 个 unique ticks 才恢复；
- deadline 使用 local monotonic receive time，tick 仅用于 progress 和 uint32 rollover；
- 最终值仍须 instrumented live capture 后冻结。

## H4 · 15 Hz 真实 Policy Shadow

```text
真实 LowState + 三相机
→ frozen LGG100
→ official quaternion-only canonicalization
→ sequential IK
→ swept path / limits
→ mock sink
```

每个周期保存 observation/raw/canonical/joint hashes、延迟、prefetch boundary、hold reason 和 current-joint delta。禁止创建 Unitree Publisher。

离线 replay 已完成 32-step、15 Hz、5-step prefetch boundary 和 mock sink。它使用 synthetic cameras、deterministic hold-policy fixture 和历史 Q0 latency，只验证 plumbing，不是 LGG100 inference。结果因 `6.54 mm` IK residual、`initial_configuration_collision` 和 waist divergence fail-closed。

## H5 · Watchdog + fail-closed mock adapter

必须注入 LowState 停止、相机冻结、Policy timeout、NaN/Inf、IK/collision failure 和 DDS 断线。每项都必须得到：

```text
publish_allowed=false
hold=true
robot_command_sent=false
```

离线 mock suite 已执行 8 个 case（NaN 和 Inf 分开、IK 和 collision 分开），8/8 通过。adapter 不含 Unitree SDK 或 hardware transport；只有绑定真实 H4 后才能将 H5 标为集成通过。

## H6 · 第一次动作评审

H1–H5 全部通过后才评审：支撑、单臂、free-space、极低速、小位移、无抓取、无桌面接触、E-stop 操作员持续就绪。`G1_FIRST_MOTION_REVIEW_CHECKLIST_CN.md` 已完成离线模板，但所有 trial-day 检查、官方 limits、双签字和一次性 arm token 均为 false。H6 之前禁止运行 Yuhao 的 `main_eef.py`、`main.py`、`replay.py`。
