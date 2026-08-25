# G1 第一次真机动作评审 Checklist

**负责人：** Shihua Yu

**用途：** H1–H5 全部通过后，对单次、支撑状态下的极小 free-space 单臂动作进行独立评审。本文档本身不授予动作权限。

## 当前决定

```text
motion_review_passed = false
motion_authorized = false
publisher_created = false
robot_command_sent = false
```

离线准备完成不等于 H6 通过。当前真实相机、DDS freshness、真实 15 Hz Shadow、集成 watchdog、物理 EEF 和硬件 limits 均未完成。

## A. 前置 Gate（全部必选）

- [ ] H1 三路真实相机：head-left、left wrist、right wrist 均提供 fresh timestamp 和 `224×224 RGB` observation。
- [ ] H2 source parity：zero-waist policy view 与 measured-waist safety/IK view 均已冻结；偏差超限必 hold。
- [ ] H2 physical parity：真实 EEF site 测量与左右手 frame 已验证。
- [ ] H3 DDS freshness：Reader error 有类型化证据，duplicate/tick jump 已解释，硬件 stale timeout 已冻结。
- [ ] H4 真实 15 Hz Policy Shadow：真实 LowState、真实相机和 frozen LGG100 inference 完成，无 Publisher。
- [ ] H4 sequential IK：完整 32-step、多 chunk 均满足 `≤5 mm / ≤3°`。
- [ ] H4 swept path：当前姿态、插值姿态及目标姿态均无禁止碰撞。
- [ ] H5 六类 fault injection 在真实 Shadow 集成层全部 fail-closed。
- [ ] 完整 observation-to-commit latency contract 已通过。

任一项未选中：停止评审，保持 Damping，不建立硬件 Publisher。

## B. 硬件和操作员确认（试验当天重新确认）

- [ ] 机器人机械支撑承载能力已由现场负责人确认。
- [ ] 安全绳不会进入机械臂 swept volume。
- [ ] 操作员全程手持并已测试 E-stop。
- [ ] 第二观察员可以口头要求立即停止。
- [ ] 机器人周围人员清场，非参与者不能进入安全区。
- [ ] 工作空间中移除桌面、方块和其他接触物；目标完全位于 free-space。
- [ ] 电池、电源、网络和机器人温度处于官方允许范围。
- [ ] 当前控制模式、authority ownership 和退出到 Damping 的步骤已由 Unitree 文档/负责人确认。

## C. 官方限制和 adapter 审核

- [ ] 官方 joint position limits 已按机器人版本冻结。
- [ ] 官方 joint velocity/acceleration limits 已冻结。
- [ ] 官方 torque/current/temperature shutdown limits 已冻结。
- [ ] command-loss watchdog timeout 已在目标控制器上验证。
- [ ] LowState stale 与 fresh-commit watchdog 已验证。
- [ ] fail-closed Unitree adapter 已通过独立代码审核。
- [ ] adapter 默认不创建 Publisher；只有一次性 operator token 和全部 Gate 才能 arm。
- [ ] 任何异常都先停止发送并进入经审核的安全 hold/Damping 路径。
- [ ] 禁止直接运行 Yuhao `main_eef.py`、`main.py` 或 `replay.py`。

## D. 首次动作 manifest

以下值必须在评审前填写、hash 并签字；空值或临时 CLI override 一律拒绝。

| 字段 | 当前值 |
|---|---|
| trial ID | 未填写 |
| 执行手臂 | 未选择（只能左或右单臂） |
| 起始 joint/EEF snapshot hash | 未填写 |
| 目标 joint/EEF hash | 未填写 |
| 最大 EEF 位移 | 未冻结；必须为评审批准的极小值 |
| 最大 joint 位移 | 未冻结 |
| 最大 EEF 速度 | 未冻结；必须为评审批准的极低值 |
| 最大 joint 速度 | 未冻结 |
| 最大持续时间 | 未冻结 |
| 允许接触 | `none` |
| gripper command | `disabled` |
| torso/waist command | `disabled` |
| locomotion | `disabled` |
| retry | `0`（首次试验禁止自动重试） |

## E. 运行前 dry-run

- [ ] 完全相同 manifest 在 mock sink 重放，hash 与评审值一致。
- [ ] 单臂未选侧保持当前关节 target，不产生 drift。
- [ ] 当前 real waist 进入 safety/IK，不被静默清零。
- [ ] action raw/canonical hashes 已保存，只有 quaternion 字段可改变。
- [ ] 每个建议 joint target 与当前姿态 delta 均在批准 envelope 内。
- [ ] fresh commit age 在允许窗口内。
- [ ] Publisher 尚未创建，dry-run 的 `robot_command_sent=false`。

## F. 必须立即 abort 的条件

- LowState、任一路相机、Policy 或 commit 超时；
- duplicate tick 持续至 stale threshold，tick backward/reset，DDS 断线；
- NaN/Inf、shape/hash/contract 不匹配；
- IK residual、joint limit、swept collision 或 waist divergence 超限；
- torque/current/temperature 超限；
- 机器人、支撑、安全绳或未选侧手臂出现非预期运动；
- 操作员/观察员发出 stop；
- E-stop、Damping fallback 或 watchdog 状态不确定。

Abort 后禁止自动重试，必须保存证据并重新走评审。

## G. 双重授权

- [ ] 操作员签名和时间：未签署。
- [ ] 独立安全审核人签名和时间：未签署。
- [ ] 负责人对 manifest SHA-256 的批准：未签署。
- [ ] 单次 Publisher arm token：未生成。

只有 A–G 全部通过，才可以形成“建议批准”。真正创建 Publisher 和执行动作仍需要当次明确命令，不由本文档自动触发。
