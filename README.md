# G1 EDU Dual-Dex1 VLA Control

本仓库只服务一个目标：

```text
Unitree G1 Education + 左右 Dex1-1
固定站立的双臂视觉语言操作
MuJoCo 验证通过后迁移到同一台 G1 EDU
```

不兼容 G1 16-D contract 的机器人 action、过渡模型和 transport mock 不属于主 simulation。

## 唯一 Sim-to-Real 链路

```text
cam_left_high + cam_left_wrist + cam_right_wrist
                    +
pelvis-frame 16-D EEF/Dex1 state + canonical prompt
                    ↓
LGG100/stack-cube-eef-24k · pinned revision cced7a…
                    ↓
pelvis-frame 16-D absolute EEF/Dex1 action chunk
                    ↓
contract validator → stale/IK/collision preflight → EEF limits
                    ↓
G1 dual-arm IK → ordered 14-joint limits + Dex1 mapping
                    ↓
MuJoCo boundary / Unitree G1 EDU hardware boundary
```

第一版 VLA 不控制腿、行走、躯干或平衡。它们继续由 Unitree 官方控制器负责。

## 安装

```bash
git clone --recurse-submodules https://github.com/danielchen26/g1-vla-control.git
cd g1-vla-control
conda create -n mujoco python=3.12 -y
conda activate mujoco
python -m pip install -r requirements.txt
```

已有 clone 补齐官方模型：

```bash
git submodule update --init --recursive
```

## 冻结的 G1 Policy Contract

单一来源：[`g1_policy_contract.yaml`](g1_policy_contract.yaml)

```text
contract_id: g1_edu_dual_dex1_eef_v1
camera preprocessing: 640×480 RGB → aspect-preserving resize with zero padding → 224×224
policy/deployment rate: 15 Hz (confirmed; explicit in formal manifests)
action horizon: 32 (author-confirmed canonical; 48 is experimental only)
action semantics: absolute target
action frame: pelvis
action dimension: 16
quaternion: xyzw
EEF: wrist_yaw_link local +X 50 mm
Dex1: 0 rad closed / 5.5 rad open
```

Action 顺序：

```text
0:3    left EEF XYZ, m, pelvis
3:7    left quaternion xyzw
7:10   right EEF XYZ, m, pelvis
10:14  right quaternion xyzw
14:16  left/right Dex1 motor, rad
```

Policy 之后只输出双臂 14 个 absolute joint-position target，顺序也在 contract 中冻结。真机 adapter 不允许重新排列、补零、镜像或猜测 frame。

## 可直接迁移到 G1 的模块

| 文件 | 职责 | 真机迁移方式 |
|---|---|---|
| `g1_policy_contract.yaml` | 唯一 observation/action/timing 定义 | 原样复用 |
| `g1_policy_contract.py` | schema、finite、quaternion、Dex1、metadata validator | 原样复用 |
| `action_schema.py` | 16-D chunk、pelvis/world、xyzw/wxyz、SLERP | 原样复用 |
| `safety_governor.py` | IK/collision preflight、EEF/Joint command filters | 用官方 limits 标定后复用 |
| `g1_dual_arm_ik.py` | G1 双臂 14-joint IK | 用真机模型和反馈核对后复用 |
| `dex1_gripper.py` | Dex1 0–5.5 rad 映射 | 接 Unitree Dex1 接口 |
| `g1_mujoco_bridge.py` | MuJoCo 图像/state 和 pelvis/world 边界 | 由 G1 hardware bridge 替换 |
| `run_simulation.py` | 完整 production-path regression | 每次真机发布前继续运行 |

## 运行 G1 Contract Simulation

```bash
conda activate mujoco
cd ~/g1_vla_control
python run_validation.py
open validation_report.html#simulation
```

单独运行：

```bash
python run_simulation.py --baseline --output results/baseline.json
python run_simulation.py --output results/adaptive.json
python render_camera_observations.py
```

`run_simulation.py` 的输入是 deterministic **G1 contract trajectory fixture**，不是 VLA。它只验证未来 policy 下游会实际复用的链路：

```text
16-D pelvis action validation
→ phase-aware preflight
→ EEF command filter
→ pelvis-to-world boundary
→ G1 dual-arm IK
→ 14-joint command filter
→ Dex1 mapping
→ G1/Dex1 MuJoCo dynamics
```

当前 baseline：

- exact 50-step horizon 的 50/50 targets 通过 grasp-phase preflight；
- EEF 与 joint command limits 通过；
- endpoint error 约 1.3 mm；
- forbidden manipulation contact rate 为 0；
- pelvis 保持稳定。

这些结果只证明接口与命令链，不证明 VLA 会叠积木，也不证明真机动力学安全。MuJoCo position actuator 的 actual jerk 仍有明显瞬态。

## Adaptive Speed

Adaptive 仍是可选模块，不是默认生产路径。证据分为两层：

- 8 cm precision fixture 没进入 16 cm far zone，scale 只有 `0.5–1.0×`，整体慢约 29.3%；它没有覆盖“该快时快”。
- 新增 17 cm G1 lift coverage：50/50 preflight，far 区 scale 中位 `1.30×`、最大 `1.60×`，near 区 `0.50×`。它证明确定性 G1 路径局部做到了“该快时快、该慢时慢”，但整体仍比 baseline 慢约 0.422 s。

这两项都没有使用真实 LGG100 chunk，因此 production adaptive 继续关闭。只有满足以下配对条件才可进入 G1：

- 使用同一个 G1 neural chunk；
- task success 不下降；
- forbidden collision/drop 不增加；
- actual joint dynamics 不恶化；
- 完成时间分布真实改善；
- G1 官方 limits、低层反馈和 watchdog 已接入。

## 生产 VLA：LGG100

生产模型确定为：

```text
repo: LGG100/stack-cube-eef-24k
revision: cced7a7ff7b454fdcac555457a1a2a3dc262ac77
```

不再切换到其他 VLA。Yuhao 已直接确认核心配置为 `pi05_g1_eef`、`action_horizon=32`、`discrete_state_input=False`。按该配置和公开代码中的 `resize_with_pad(224,224)` 重新验证后，50/50 输出为有限 `[32,16]`，语义识别和预注册的单次采样离线质量标准均通过。

Yuhao 的 pinned `g1-client` deployment code 进一步确认：该 checkpoint 已训练，预测 quaternion 应由 consumer 在 IK 前归一化，并提供 full-horizon prefetch/time-alignment/joint-space blend 参考。项目负责人已确认正式 deployment cadence 为代码默认的 15 Hz；30 Hz help text 视为过期文档。这仍然不是本项目的闭环抓取或堆叠成功证据；完整 sequential IK/swept-path、15 Hz 调度延迟、实时 watchdog 和硬件安全合同尚未通过，因此严格标记：

```text
policy_task_quality_passed=false
g1_contract_verified=true
g1_sim_eligible=false
g1_execution_enabled=false
```

下一步是重新建立 Adaptive OFF 的 horizon-32 MuJoCo 基线；不能复用旧 horizon-50 artifacts，也不能因为离线误差较低就直接执行。

- Yuhao deployment 审计：[`YUHAO_G1_CLIENT_DEPLOYMENT_AUDIT_CN.md`](YUHAO_G1_CLIENT_DEPLOYMENT_AUDIT_CN.md)
- 最新 author-32 重验证：[`LGG100_AUTHOR32_REVALIDATION_CN.md`](LGG100_AUTHOR32_REVALIDATION_CN.md)
- GPU 安装与真实权重命令：[`LGG100_REAL_VLA.md`](LGG100_REAL_VLA.md)
- 真正 VLA + Adaptive Module 完整 Gate：[`REAL_LGG100_ADAPTIVE_WORKFLOW.md`](REAL_LGG100_ADAPTIVE_WORKFLOW.md)

## MuJoCo → G1 EDU Gates

1. **D0 Contract**：冻结 YAML、validators、round-trip fixtures。
2. **D1 Hardware parity**：确认 G1 EDU firmware/SDK/control mode、相机标定、EEF site、joint IDs、官方 limits。
3. **D2 Dataset**：100 episodes 转换为冻结 16-D contract，episode-level split，无 frame leakage。
4. **D3 LGG100 Restore**：Ubuntu NVIDIA strict Orbax restore，固定 HF/OpenPI revision，恢复并验证 config/transform/golden sample。
5. **D4 Offline**：真实 LGG100 chunk 只保存，不执行；schema/timing/IK/collision 全部通过。
6. **D5 MuJoCo closed loop**：multi-chunk、任务成功率、随机化、网络 jitter、stale/断线。
7. **D6 Shadow/HIL**：真机 observation 输入 policy，机器人保持 hold，只记录建议 action。
8. **D7 Staged hardware**：E-stop 和支撑下逐步执行单臂、双臂、桌面、轻物体、已训练任务。

上一 Gate 不通过，下一 Gate 不得获得 action 权限。

## 当前真机阻塞项

- G1 EDU 准确 firmware、Unitree SDK 与 control mode；
- 三相机型号、安装位姿、内外参和硬件时间同步；
- 真机 pelvis 与左右 50 mm EEF site 标定；
- 官方逐关节 position/velocity/torque/current limits；
- 低层反馈、watchdog、hold、通信中断和 E-stop；
- G1 hardware adapter 与 Shadow/HIL logs；
- 通过冻结 contract 训练并验证的真实 neural policy。

在这些项目完成前：

```text
g1_hardware_execution_enabled=false
```

## 深度验证

```bash
python run_deep_validation.py
```

现有证据包括：100 episodes / 95,966 帧数据审计、episode-0 回放、250/250 数据内 IK、1000 OOD workspace、Dex1 抓持边界、30 秒稳定性、20 组随机化、扰动和视觉域差异。它们继续用于 G1 场景与安全链验证，但不会被描述成 neural VLA 或真机成功。

## 第三方资源

- `third_party/mujoco_menagerie/`：Google DeepMind MuJoCo Menagerie
- `third_party/dex1_1_service/`：Unitree Dex1-1 URDF/STL，BSD-3-Clause

仓库不包含 VLA checkpoint。第三方资源继续适用其上游许可证。
