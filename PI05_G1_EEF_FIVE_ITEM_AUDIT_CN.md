# `pi05_g1_eef` 五项关键信息审计

## 审计对象

- GitHub：<https://github.com/leihao100/openpi-fintune>
- 审计 commit：`29030046fd6a2810201db67b9804f243e0af3218`
- LGG100 checkpoint：`LGG100/stack-cube-eef-24k`
- Checkpoint revision：`cced7a7ff7b454fdcac555457a1a2a3dc262ac77`

## 总体结论

该仓库可以补齐大部分 LGG100 输入输出定义，但尚不能单独证明 `pi05_g1_eef` 就是 `stack-cube-eef-24k` checkpoint 的精确训练配置。

主要原因是：

- `pi05_g1_eef` 使用 `action_horizon=48`；
- `pi05_g1_eef` 配置的数据集是 `sort-tools-eef`；
- 配置训练步数是 20,000；
- 当前 checkpoint 名称和 asset 是 `stack-cube-eef-24k`；
- checkpoint 没有 README、config 文件或 OpenPI commit 信息。

因此，五项信息中有三项基本补齐，两项仍需 Yuhao/模型作者确认。

## 五项信息状态

| 项目 | 仓库提供的信息 | 状态 | 仍缺少的信息 |
|---|---|---|---|
| 1. OpenPI 版本和模型配置 | π0.5；`action_horizon=48`；`discrete_state_input=False`；PaliGemma 2B LoRA；300M action expert | 部分完成 | 需确认该 checkpoint 使用的精确 commit/config、是否确实为 48 horizon、24k 的训练配置 |
| 2. 相机输入和图像处理 | 3 个 RGB 相机；480×640；high/left wrist/right wrist；最终使用 `resize_with_pad(224,224)` | 基本完成 | 相机内外参、图像翻转和真实安装标定不在仓库中 |
| 3. 16 维 state 定义 | 左 EEF xyz+xyzw、右 EEF xyz+xyzw、左右夹爪；pelvis frame；EEF 为 wrist-yaw 沿 +X 50 mm | 基本完成 | 夹爪真机单位、开闭方向及控制标定仍需确认 |
| 4. 16 维 action 定义 | 与 state 相同的 16 维 EEF 格式；`pi05_g1_eef` 默认 absolute；输出顺序明确 | 基本完成 | checkpoint 对应的 horizon、每次执行多少 action、实际 replanning 周期仍需确认 |
| 5. normalization/post-processing/replanning | π0.5 使用 quantile normalization；checkpoint 内有 16 维 `norm_stats`；输出取前 16 维；IK 前应归一化两个四元数 | 部分完成 | 缺少 G1 部署 client；没有明确执行 chunk 前几个点、控制频率及 replanning 策略 |

## 仓库中确认的具体定义

### 模型配置

`pi05_g1_eef` 配置为：

```text
pi05=True
action_horizon=48
discrete_state_input=False
paligemma_variant=gemma_2b_lora
action_expert_variant=gemma_300m
```

来源：

<https://github.com/leihao100/openpi-fintune/blob/29030046fd6a2810201db67b9804f243e0af3218/src/openpi/training/config.py#L940-L993>

### 相机输入

训练数据使用：

```text
observation.images.cam_left_high   480×640 RGB
observation.images.cam_left_wrist  480×640 RGB
observation.images.cam_right_wrist 480×640 RGB
```

模型内部映射为：

```text
base_0_rgb
left_wrist_0_rgb
right_wrist_0_rgb
```

图像最终通过 `resize_with_pad` 缩放到 224×224，而不是先做中心裁剪。

来源：

<https://github.com/leihao100/openpi-fintune/blob/29030046fd6a2810201db67b9804f243e0af3218/src/openpi/policies/unitree_eef_policy.py#L1-L91>

### State/Action 定义

16 维顺序为：

```text
[0:3]   左 EEF xyz，pelvis frame，单位 m
[3:7]   左 EEF quaternion，qx qy qz qw
[7:10]  右 EEF xyz，pelvis frame，单位 m
[10:14] 右 EEF quaternion，qx qy qz qw
[14]    左夹爪
[15]    右夹爪
```

EEF 定义为 wrist-yaw frame 沿局部 +X 偏移 0.05 m。数据转换时 waist、legs、fingers 锁定在 0。

来源：

<https://github.com/leihao100/openpi-fintune/blob/29030046fd6a2810201db67b9804f243e0af3218/scripts/joint_to_eef.py#L1-L115>

### 四元数后处理

仓库明确写明：模型预测的四元数不保证单位范数，消费者应在送入 IK 前归一化：

```text
normalize actions[:, 3:7]
normalize actions[:, 10:14]
```

来源：

<https://github.com/leihao100/openpi-fintune/blob/29030046fd6a2810201db67b9804f243e0af3218/src/openpi/policies/unitree_eef_policy.py#L94-L103>

### Normalization

π0.5 使用 checkpoint 中的 quantile normalization。当前 checkpoint 的 state/action norm stats 均为 16 维，位置统计值是绝对 EEF 位置分布，因此支持 quaternion 16-D absolute EEF 表示，而不是 20-D 6D-rotation 表示。

## 对当前项目的重要影响

当前实验性推理配置与该仓库存在以下差异：

| 项目 | 当前旧设置 | 仓库设置 | 影响 |
|---|---|---|---|
| Action horizon | 50 | 48 | 可能改变模型采样轨迹和执行时间 |
| State tokenizer | `discrete_state_input=True` | `False` | 可能显著影响模型对机器人状态的理解 |
| 图像处理 | 640×480 中心裁剪后缩放 | 保比例缩放并黑边 padding 到 224×224 | 可能造成训练与推理视觉分布不一致 |
| 四元数 | 小范数误差被视为仅可隔离分析 | 官方 policy 说明消费者应归一化后再送 IK | 需要建立明确、可审计的官方后处理合同 |
| EEF offset | wrist +X 0.05 m | wrist +X 0.05 m | 当前实现与仓库一致 |
| 坐标和顺序 | pelvis、absolute、xyzw、left-right | 相同 | 当前语义判断得到进一步支持 |

这些差异可能是此前 MuJoCo 闭环没有完成抓取的重要原因之一。在按照仓库配置重新运行前，不能用旧结果判断 LGG100 checkpoint 本身无效。

## 仍需向 Yuhao 确认的问题

1. `stack-cube-eef-24k` 的精确 OpenPI commit 是什么？
2. 该 checkpoint 对应的 config 名称是什么？是否就是本仓库的 `pi05_g1_eef` 的本地修改版本？
3. `action_horizon` 是否为 48，而不是 50？
4. `discrete_state_input` 是否确定为 `False`？
5. 训练 action 是否为 absolute quaternion EEF，而不是 delta 或 6D rotation？
6. 24k 表示 checkpoint step 24,000 吗？训练时的 dataset 是否为 `stack-cube-eef`？
7. G1 客户端每次执行 action chunk 的几个点？控制频率和 replanning 频率是多少？
8. 左右夹爪数值的单位、开闭方向和真机映射是什么？
9. 是否可以提供 `main_eef.py`、`eef_kinematics.py` 或实际部署 client？
10. 三个相机的内参、外参、安装方向和图像翻转设置是什么？

## 下一步建议

1. 先由 Yuhao 确认上述 checkpoint/config 对应关系；
2. 固定正确的 OpenPI commit；
3. 使用 48 horizon、continuous state 和官方 `resize_with_pad` 重新做 output-only 推理；
4. 按仓库说明归一化四元数，并保留原始输出与后处理 hash；
5. 重新运行公开 episode 的语义/质量测试；
6. 重新生成 exact t=0 MuJoCo observation；
7. 先运行 Adaptive OFF 的闭环基线；
8. 基线成功后再评估 Adaptive OFF/ON。

在完成上述重新验证前，`g1_sim_eligible` 和 `g1_execution_enabled` 必须继续保持 `false`。
