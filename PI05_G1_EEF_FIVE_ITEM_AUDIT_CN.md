# `pi05_g1_eef` 五项关键信息审计

## 审计对象

- GitHub：<https://github.com/leihao100/openpi-fintune>
- 审计 commit：`29030046fd6a2810201db67b9804f243e0af3218`
- LGG100 checkpoint：`LGG100/stack-cube-eef-24k`
- Checkpoint revision：`cced7a7ff7b454fdcac555457a1a2a3dc262ac77`

## 作者后续确认

项目负责人通过与 Yuhao 的直接沟通获得了以下回复：

```text
model config name: pi05_g1_eef
action_horizon: 32
discrete_state_input: False
```

Yuhao 同时表示 horizon 48 可以试验，但 32 应作为严格复现的 canonical 配置，48 只能作为实验性对照。该沟通尚没有对应的公开 commit 或完整 TrainConfig，因此在结果中必须明确标注来源。

## 总体结论

公开仓库与作者回复已经足以重建核心 output-only/MuJoCo 推理配置，但尚不足以解锁 G1 真机：

- 模型属于 π0.5 的 `pi05_g1_eef`；
- canonical action horizon 为 32；
- `discrete_state_input=False`；
- 输入为三路 RGB 相机和 16-D pelvis-frame EEF state；
- 输出为 16-D absolute quaternion EEF action；
- π0.5 使用 quantile normalization；
- 四元数在送入 IK 前由消费者归一化；
- `g1-client` 已公开 full-remaining-horizon execution、prefetch、time-alignment 和 joint-space blend 参考；但代码默认 15 Hz 与 help text 30 Hz 冲突。

## 五项信息状态

| 项目 | 当前结论 | 状态 | 仍缺少的信息 |
|---|---|---|---|
| 1. OpenPI 版本和模型配置 | `pi05_g1_eef`；horizon 32；continuous state；π0.5 | 核心参数完成 | 精确 OpenPI commit、完整历史 TrainConfig 和训练命令 |
| 2. 相机输入和图像处理 | 3 个 RGB 相机；480×640；high/left wrist/right wrist；`resize_with_pad(224,224)` | 基本完成 | 相机内外参、畸变、时间同步和真实安装标定 |
| 3. 16 维 state 定义 | 左/右 EEF xyz+xyzw、左右夹爪；pelvis frame；EEF 为 wrist-yaw 沿 +X 50 mm | 基本完成 | 夹爪真机标定仍需正式验证 |
| 4. 16 维 action 定义 | 16-D absolute quaternion EEF；canonical horizon 32 | 基本完成 | 真机 controller 合同和 action timing |
| 5. normalization/post-processing/replanning | checkpoint quantile norm；IK 前归一化；full-horizon + prefetch/time-alignment/blend reference | 基本完成 | 15/30 Hz 冲突需显式实验或作者确认；项目安全 Gate 仍需补齐 |

## 公开代码证据

### 输入输出定义

`unitree_eef_policy.py` 明确给出：

```text
[0:3]   左 EEF xyz，pelvis frame，单位 m
[3:7]   左 EEF quaternion，qx qy qz qw
[7:10]  右 EEF xyz，pelvis frame，单位 m
[10:14] 右 EEF quaternion，qx qy qz qw
[14]    左夹爪
[15]    右夹爪
```

相机映射为：

```text
cam_left_high  -> base_0_rgb
cam_left_wrist -> left_wrist_0_rgb
cam_right_wrist -> right_wrist_0_rgb
```

来源：

<https://github.com/leihao100/openpi-fintune/blob/29030046fd6a2810201db67b9804f243e0af3218/src/openpi/policies/unitree_eef_policy.py#L1-L103>

### EEF 定义

EEF 是 wrist-yaw frame 沿局部 +X 偏移 0.05 m；数据转换时 waist、legs、fingers 锁定在 0。

来源：

<https://github.com/leihao100/openpi-fintune/blob/29030046fd6a2810201db67b9804f243e0af3218/scripts/joint_to_eef.py#L1-L115>

### 图像处理与 normalization

标准 model transform 使用 `ResizeImages(224,224)`，其实现是保持长宽比并补零，而不是中心裁剪。π0.5 使用 quantile normalization。

来源：

<https://github.com/leihao100/openpi-fintune/blob/29030046fd6a2810201db67b9804f243e0af3218/src/openpi/training/config.py#L115-L198>

### 四元数后处理

公开 policy 明确说明预测四元数不保证单位范数，消费者应在送入 IK 前归一化 `q[3:7]` 和 `q[10:14]`。原始输出仍必须保留 hash，零范数、非有限值和超出安全边界的输出继续 fail-closed。

## 公开仓库与作者配置的差异

当前公开 `main` 中的 `pi05_g1_eef` 写的是 horizon 48，并指向 `sort-tools-eef / 20k`。作者确认 `stack-cube-eef-24k` 使用 horizon 32。这说明 checkpoint 使用的历史或本地 TrainConfig 没有完整公开。

因此：

- 32 是 canonical 严格复现配置；
- 48 仅作为作者允许尝试的实验性对照；
- 不能把 48 的结果当成 checkpoint 原始配置结果；
- 不能根据参数树成功恢复来推断 horizon，因为 horizon 不编码在权重叶子形状中。

## 对此前实验的影响

此前实验使用：

```text
action_horizon=50
discrete_state_input=True
center crop -> 224×224
```

正确核心配置为：

```text
action_horizon=32
discrete_state_input=False
resize_with_pad -> 224×224
```

所以此前的输出语义结果可以作为历史诊断，但任务质量、实时性和 MuJoCo 闭环必须重新运行，不能继续用旧结果判断 checkpoint 本身的能力。

## 仍未解决

1. 精确 OpenPI commit 和完整历史 TrainConfig；
2. `g1-client` 的 `control_hz=15` 代码默认值与“default 30”help text 冲突；
3. 部署参考对 IK residual 只 warning 后继续 dispatch，缺少本项目的 orientation/swept collision Gate；
4. 三相机正式标定和时间同步；
5. Dex1 真机数值映射；
6. G1 controller、watchdog 和 E-stop 合同。

## 下一步

1. 以 horizon 32、continuous state、`resize_with_pad` 重建样本；
2. 重新运行真实 checkpoint output-only inference；
3. 保留原始输出，并按公开 policy 与 deployment IK 对非零四元数做有界、可审计归一化；
4. 使用 source attestation 将 `g1_contract_verified` 与 `g1_sim_eligible` 分离；
5. 固定 15/30 Hz cadence manifest 后完成 multi-chunk IK/swept-path，再运行 Adaptive-OFF MuJoCo 基线；
6. horizon 48 仅进行固定 observation/seed 的对照；
7. 所有真机 gate 继续保持关闭。
