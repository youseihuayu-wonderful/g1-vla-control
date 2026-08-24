# LGG100 作者确认核心配置重验证

## 1. 范围

本报告记录 `LGG100/stack-cube-eef-24k` 在作者确认核心配置下的 output-only 重验证。整个过程没有执行 MuJoCo dynamics，也没有向 Unitree G1 发送动作。

作者通过直接沟通确认：

```text
config name: pi05_g1_eef
action_horizon: 32
discrete_state_input: False
```

公开 `openpi-fintune` 代码与 pinned `g1-client` deployment code 同时支持：

- 三路 480×640 RGB 相机；
- `resize_with_pad(224,224)`，而不是中心裁剪；
- 16-D pelvis-frame absolute EEF action；
- quaternion 顺序 `xyzw`；
- π0.5 quantile normalization；
- 消费者在送入 IK 前归一化非零四元数。

完整历史 TrainConfig 和精确 OpenPI commit 仍未公开；但训练 checkpoint、EEF transform、quaternion consumer boundary 和部署参考已经恢复。

## 2. 固定版本

- 项目代码：`6d29f2ae431550e239bfd60d3e877a843a1c7bb4`
- Checkpoint revision：`cced7a7ff7b454fdcac555457a1a2a3dc262ac77`
- Dataset revision：`c468ef259ff8bfad1b3b0e7b2c1f45efbb0b30ba`
- GPU：NVIDIA L40S，固定使用 GPU 5
- 自动化测试：59/59 通过
- GitHub Actions：<https://github.com/youseihuayu-wonderful/g1-vla-control/actions/runs/31848520564>

## 3. Output-only 推理结果

在五个公开 episode 上各抽取 10 个 observation，共 50 个同步样本。

| 指标 | 结果 |
|---|---:|
| 输出形状和有限值 | 50/50，通过，均为 `[32,16]` |
| Raw quaternion exact-unit diagnostic | 35/50 |
| 官方 consumer post-processing | 50/50 |
| 唯一 raw output hash | 50/50 |
| 冷启动首调用 | 14,611.1 ms |
| 热调用 P50 | 80.52 ms |
| 热调用 P95 | 81.38 ms |
| 热调用 P99 | 83.19 ms |
| 热调用最大值 | 83.75 ms |

冷启动时间包含 JAX 编译，不能混入稳态延迟结论。热调用延迟只包含当前 output-only policy 调用，不包含完整的 render、网络、IK、swept preflight 和 command commit，因此不能据此声明实时 watchdog 已通过。

35/50 raw 输出满足项目当前 `1e-3` exact-unit diagnostic。其余 15 个样本通过有界官方 consumer boundary；公开 `pi05_g1_eef` training transform 与 deployment IK 都明确要求消费者在 IK 前归一化四元数。原始输出与 postprocessed 输出分别保留 hash。该结果验证 action contract，但不解锁 Simulation dynamics 或真机执行。

## 4. 语义与离线质量结果

最佳解释仍为：

```text
absolute_xyzw_lr
```

即 pelvis frame、absolute EEF、`xyzw`、左手在前、右手在后。

| 指标 | 结果 |
|---|---:|
| 可用样本 | 50/50 |
| 最佳假设逐样本胜率 | 100% |
| 相对第二名 margin | 97.42% |
| EEF position RMSE | 3.20 mm |
| EEF position error norm P95 | 9.68 mm |
| quaternion geodesic mean | 0.765° |
| quaternion geodesic P95 | 1.773° |
| gripper RMSE | 0.0754 rad |
| first-target jump P95 | 17.25 mm |
| 模型综合 score | 0.1740 |
| Hold baseline score | 0.9621 |

结果为：

```text
semantic_identification_supported=true
offline_single_draw_policy_quality_passed=true
```

这里的 `offline_single_draw_policy_quality_passed` 仅表示预注册的“模型离线 score 优于 hold”标准通过。它不等于抓取成功、堆叠成功或闭环任务成功。

## 5. 与旧候选配置的关系

此前使用：

```text
action_horizon=50
discrete_state_input=True
center crop -> 224×224
```

这些设置与作者确认配置不一致。旧结果现仅作为历史诊断，不能用于评价 checkpoint 的任务能力。正确核心配置显著改善了公开 episode 上的动作一致性，因此 Adaptive OFF 基线必须重新建立。

## 6. Gate 状态

本次通过：

- 作者确认核心配置的严格参数恢复；
- 50/50 `[32,16]` finite output；
- 语义识别；
- pinned official quaternion consumer post-processing；
- 单次采样离线质量标准；
- 59/59 自动化测试。

仍未通过：

```text
policy_task_quality_passed=false
physical_scene_calibration_verified=false
real_time_watchdog_validated=false
g1_contract_verified=true
g1_sim_eligible=false
g1_execution_enabled=false
```

下一步应按 contract v1.3 重新生成或无损迁移 horizon-32 artifacts，显式固定 deployment code 中存在歧义的 15/30 Hz cadence，再完成 full multi-chunk IK/swept-path；通过后首先运行 Adaptive-OFF 基础闭环。不能复用旧 horizon-50 artifacts。

## 7. 远端证据

证据均保存在：

```text
/home/user1/workspace/shihua/results
/home/user1/workspace/shihua/logs
```

主要文件及 SHA-256：

```text
lgg100_author32_semantic_samples.npz
  e0d7e57c859aa6bbf53792fbf9e14df2cd0a7833319a914325d03174b3487d83

lgg100_author32_semantic_samples_manifest.json
  3e7f283b4b9cc79636a65d1857added17035b4b27562fa480248fcefd6009ee4

lgg100_author32_semantic_outputs_quarantined.npz
  92f360ef059617c915d4dabcb90adafa88531528f7cde2d426ff2e5958882173

lgg100_author32_semantic_inference.json
  1c065b3602e7060d6479561b3e05c0f6cf90fd44cce5cb8cc423636ba19538e0

lgg100_author32_semantic_validation.json
  9774003434643ad64d96d69bfc8af0bb18aec3614d87343526792fc06b6d71f0

lgg100_author32_revalidation_summary.json
  bbf4b6c24f64bfb231a1bc2a976c7a0ab4bbaa87774503db3bf12dcd601f278e
```

测试结束后没有留下本项目 policy server；没有终止或修改服务器上已有的其他进程。
