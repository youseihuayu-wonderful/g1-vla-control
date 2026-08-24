# Yuhao `g1-client` 部署代码审计

## 固定来源

- 仓库：<https://github.com/leihao100/g1-client>
- 审计 commit：`1422e8d6ef674aa047cfb2878bc7dae54b118fbe`
- EEF OpenPI 入口：`openpi/main_eef.py`
- FK/IK：`openpi/eef_kinematics.py`
- OpenPI transport：`openpi/openpi_policy.py`
- Robot publisher：`g1_client/arm_controller.py`、`g1_client/gripper_controller.py`

审计时仓库根目录未发现 LICENSE 文件，因此当前只固定引用和复现行为，不复制其运动控制代码进入本项目。

## 结论

该仓库确认 `pi05_g1_eef` 是已经训练好的 EEF-space OpenPI policy，并提供了完整的部署参考链：

```text
三相机 + measured arm/gripper state
→ pelvis-frame EEF-16 observation
→ OpenPI inference [H,16]
→ quaternion normalization
→ warm-start dual-arm IK
→ joint-space boundary blend/time alignment
→ Unitree arm_sdk + Dex1 publishers
```

因此，本项目不需要重新训练 LGG100。此前将 `raw quaternion exact-unit=0/N` 表述为模型 contract 失败是不准确的：作者训练/部署代码都明确要求消费者在 IK 前归一化预测四元数。

## 已确认的 EEF contract

`openpi/main_eef.py` 明确声明：

```text
state/action = [left xyz+qx qy qz qw,
                right xyz+qx qy qz qw,
                left gripper,
                right gripper]
frame        = pelvis
EEF          = wrist_yaw + local X 0.05 m
```

`openpi/eef_kinematics.py::_xyzquat_to_se3()`：

1. 读取 `pose7[3:7]`；
2. 拒绝 norm `<1e-6`；
3. 执行 `q / ||q||`；
4. 再送入 Pinocchio IK。

该行为与 `openpi-fintune` 中 `UnitreeG1EEFOutputs` 的注释一致：预测四元数不保证单位范数，consumer 应在 IK 前归一化左右 quaternion。

## 部署闭环策略

`openpi/main_eef.py` 使用 stateless receding-horizon inference：

- 首个 chunk 同步 inference；
- 后续 chunk 在当前 chunk 尾部异步 prefetch；
- `exec_steps=0` 的代码默认含义是执行当前剩余 horizon；
- `prefetch_lead=5`；
- 新 chunk 根据 inference 期间已经经过的 steps 做 time alignment；
- `blend_steps=5`，在 IK 后的 joint space 做 cross-fade；
- IK 从上一时刻 joint solution warm-start，保持冗余肘关节分支连续。

代码参数默认 `control_hz=15.0`，但 help text 写“default 30”。项目负责人已于 2026-08-24 确认正式部署 cadence 为 **15 Hz**；因此 formal Simulation manifest 必须显式写入 `control_hz=15.0`，30 Hz help text 作为过期文档记录保留，不再作为实验分支。32-step chunk 的标称跨度为 `31/15 ≈ 2.067 s`，prefetch lead 5 的调度窗口为 `5/15 ≈ 333 ms`。

## IK 实现差异

作者部署参考使用 Pinocchio 14-DOF reduced model：

- 腰、腿和手指锁定为 0；
- 双臂联合 DLS IK；
- `max_iters=20`、`tol=1e-5`、`damping=1e-8`；
- quaternion 在 `_xyzquat_to_se3()` 中归一化；
- 返回最坏双手 position residual。

但部署入口对 `IK_WARN_M=0.02` 只打印 warning，随后仍会 dispatch；没有完整 orientation threshold，也没有 swept-path collision preflight。这与本项目 fail-closed 的 `5 mm / 3°` Gate 不等价。作者 IK/调度应作为 parity reference，本项目不能直接删除更严格的拒绝逻辑。

## Robot publisher 风险边界

该仓库不是只读工具。以下代码会产生真实动作：

- `ArmController` 创建 `ChannelPublisher("rt/arm_sdk", LowCmd_)`；
- `motor_cmd[29].q=1.0` 获取 arm_sdk authority；
- 腿和腰被锁在 startup pose；
- 双臂以 50 Hz 发布；
- Dex1 以 200 Hz 向左右 command topic 发布；
- 默认包含 ready-pose move、gripper close/open 和 gravity feedforward。

其前置模式是 operator 设置 `ai` mode 并站立，不是当前已确认的 Damping 状态。因此：

```text
禁止在当前真实 G1 shell 运行 README 中的 main_eef.py、main.py、replay.py。
```

当前只允许静态审计、离线移植 contract/IK/scheduling reference，以及 mock-sink Simulation。

## 对 Adaptive-OFF 的修正

应使用两级 action boundary：

```text
Raw neural audit
- finite [32,16]
- correct channel/range
- nonzero bounded quaternion
- preserve raw SHA-256

Official consumer boundary
- normalize only q[3:7] and q[10:14]
- preserve position/gripper bytes
- validate canonical unit quaternions
- preserve canonical SHA-256
```

因此 Q0/Q0.5 的旧字段：

```text
raw_contract_passes=0/N
bounded_analysis_available=N/N
```

应解释为：

```text
raw_quaternion_exact_unit_passes=0/N
official_consumer_postprocess_passes=N/N
```

这不再是 Adaptive-OFF 的 quaternion blocker，15 Hz cadence 也已固定。剩余 Simulation blocker 是完整 multi-chunk IK/swept-path、15 Hz 下的完整调度/延迟验证和 fresh-commit watchdog。
