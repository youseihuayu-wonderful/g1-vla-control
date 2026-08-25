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
| H1 | 真实三相机 | **阻塞** | 配置存在且三路 `enable_zmq=true`，但没有 image-server 进程/端口，三帧均为 `None` | 可建立真实视觉 observation；仍不可运动 |
| H2 | 真实关节 FK 对照 | **部分通过** | 300-sample LowState 可映射到 MuJoCo；frame/current-pose IK 数值 round-trip 通过 | 可冻结 zero-waist/waist-aware FK 选择；仍不可运动 |
| H3 | DDS freshness | **部分通过** | 300/300、无 tick decrease；18 Reader errors、9 duplicates、最大正 tick jump 15 | 可冻结 stale/watchdog 时间阈值；仍不可运动 |
| H4 | 15 Hz Policy Shadow | **锁定** | 等待 H1–H3 | 可保存真实 observation 下的建议动作；仍不可运动 |
| H5 | Watchdog + fail-closed mock adapter | **锁定** | 已有 Simulation 单元机制，但未绑定本轮真实 Shadow | 可进入 H6 评审；仍不自动运动 |
| H6 | 第一次真机动作评审 | **锁定** | H1–H5 未全部通过 | 仅决定是否允许单次、低速、free-space 试验 |

当前完整解决数：**0/6**。H2/H3 的部分结果不得表述为完整通过。

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

`results/g1_three_camera_readonly_probe_20260824.json` 证明配置存在但 runtime 未启动。下一动作是审计 camera-only server 启动入口；只有确认其不包含机器人 Publisher/模式切换后，才能启动并重跑固定三帧 probe。

## H2 · Yuhao Pinocchio FK × 项目 MuJoCo FK

### 已完成

- 真实 waist 3 + arms 14 均存在且落在 MuJoCo model range；
- pelvis/world frame round-trip 最大 position error `6.05e-9 m`；
- perturbation-seeded current-pose IK 最终最大 position error `4.46e-5 m`；
- 最终最大 orientation error约 `4.30e-5 deg`。

### 尚缺

1. 对同一真实 `q14` 运行 pinned Yuhao Pinocchio FK；
2. 与 MuJoCo `waist=0` 结果比较；
3. 与 MuJoCo measured-waist 结果比较；
4. 量化真实非零 waist 对左右 EEF 的 position/orientation 影响；
5. 物理测量 EEF site，不能只依赖两个软件模型互相同意。

## H3 · DDS freshness

### 当前测量

- 300/300 samples；
- callback overflow 0；
- median gap `1.397231 ms`，max gap `3.228279 ms`；
- 291 unique ticks、9 duplicates、0 decreases；
- positive tick increment `1–15`；
- 18 次 Reader error。

### 通过标准

- 重复多个固定 capture，保存 Reader error 与 tick/gap 联合分布；
- 解释 duplicate/tick jump 语义；
- 冻结 warning、hold、disconnect timeout；
- offline fault injection 证明 stale 时只进入 hold。

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

## H5 · Watchdog + fail-closed mock adapter

必须注入 LowState 停止、相机冻结、Policy timeout、NaN/Inf、IK/collision failure 和 DDS 断线。每项都必须得到：

```text
publish_allowed=false
hold=true
robot_command_sent=false
```

## H6 · 第一次动作评审

H1–H5 全部通过后才评审：支撑、单臂、free-space、极低速、小位移、无抓取、无桌面接触、E-stop 操作员持续就绪。H6 之前禁止运行 Yuhao 的 `main_eef.py`、`main.py`、`replay.py`。
