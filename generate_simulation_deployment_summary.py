#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the single-table Simulation → G1 deployment summary."""

from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "simulation_deployment_summary.html"
PUBLIC_OUTPUT = ROOT / "vercel_public" / "index.html"
PUBLIC_GITHUB_BASE = (
    "https://github.com/youseihuayu-wonderful/g1-vla-control/"
    "blob/work/lgg100-semantic-speed-gates/"
)

TERM_INFO = {
    "IK": ("Inverse Kinematics，逆运动学：根据双手目标位置/姿态求解各关节角。本项目用它把 LGG100 的 EEF action 转成 G1 双臂 14 关节目标；不收敛时必须 hold。", "https://en.wikipedia.org/wiki/Inverse_kinematics"),
    "FK": ("Forward Kinematics，正运动学：由当前关节角计算末端执行器位置和姿态。本项目用真实 LowState 重建双手 EEF state，并验证 IK 结果。", "https://en.wikipedia.org/wiki/Forward_kinematics"),
    "EEF": ("End Effector，末端执行器：机械臂最末端用于抓取的参考点/坐标系。本项目定义在左右 wrist yaw link 前方的特定 site，真机必须实测一致。", "https://en.wikipedia.org/wiki/Robot_end_effector"),
    "VLA": ("Vision-Language-Action 模型：根据图像、语言指令和机器人状态预测动作。本项目的 VLA 是 LGG100；输出必须先经过 contract、IK 和安全预检。", "https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/"),
    "LGG100": ("本项目使用的高层 VLA policy/checkpoint。它输出 32×16 的双手 EEF action chunk，但输出 shape 正确不等于抓取任务或硬件执行已经安全。", "https://huggingface.co/LGG100/stack-cube-eef-24k"),
    "GEN-1.5": ("Generalist AI 于 2026-08-19 发布的 embodied foundation model。官方称其可用 3–12 秒 sensorimotor demonstration 做 one-shot physical prompting；目前未公开权重、代码、API 或 action schema。", "https://generalistai.com/blog/gen-1.5"),
    "physical prompting": ("把同步的传感器观测与动作轨迹示例放入模型 context window，让模型无需梯度更新就推断并执行新任务。该能力不能假定适用于 LGG100。", "https://generalistai.com/blog/gen-1.5#one-shot-in-context"),
    "in-context learning": ("上下文学习：模型根据当前输入上下文中的示例临时表现出新能力，而不修改权重。GEN-1.5 将 sensorimotor demonstration 作为 physical prompt。", "https://generalistai.com/blog/gen-1.5#one-shot-in-context"),
    "one-shot": ("单示例学习：只给一个演示就尝试新任务。GEN-1.5 官方报告平均成功率 59%，并明确说明短时任务且比微调模型更脆弱。", "https://generalistai.com/blog/gen-1.5"),
    "sim-to-real": ("从仿真到真实迁移。GEN-1.5 声称可把模拟 rollout 作为 real robot 的 physical prompt；这不消除真实标定、动力学差异或安全 Gate。", "https://generalistai.com/blog/gen-1.5#sim2real"),
    "MuJoCo": ("用于机器人动力学与接触仿真的物理引擎。本项目在 MuJoCo 中验证 observation、IK、碰撞和 retiming；仿真结果不能替代真机标定。", "https://mujoco.org/"),
    "Quaternion": ("四元数：无奇异表示三维旋转的四个数。必须明确分量顺序并归一化；本项目 policy contract 使用 xyzw。", "https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation"),
    "quaternion": ("四元数：无奇异表示三维旋转的四个数。必须明确分量顺序并归一化；本项目 policy contract 使用 xyzw。", "https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation"),
    "xyzw": ("四元数分量顺序 x、y、z、w。顺序错误会产生完全错误的手腕姿态，因此属于 fail-closed contract 字段。", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html"),
    "pelvis-frame": ("以机器人骨盆为参考的坐标系。LGG100 的双手位置/姿态都在该 frame 中表达，真机 FK、相机和 EEF 必须转换到同一 frame。", "https://en.wikipedia.org/wiki/Frame_of_reference"),
    "29-DOF": ("G1 29 自由度配置：腿、腰和双臂共 29 个关节。当前 policy 只控制双臂索引 15–28，不控制腿和腰。", "https://github.com/unitreerobotics/unitree_sdk2"),
    "DOF": ("Degrees of Freedom，自由度：机器人可独立运动的关节维度数量。", "https://en.wikipedia.org/wiki/Degrees_of_freedom_(mechanics)"),
    "Dex1": ("Unitree Dex1 夹爪/手部执行器。本项目 state/action 各包含左右 Dex1 一个标量；零点、方向和范围必须在真机标定。", "https://github.com/unitreerobotics/dex1_1_service"),
    "contract": ("版本化的数据与安全契约：规定输入/输出 shape、顺序、坐标系、单位、时间和 Gate。任何不匹配都应拒绝，而不是自动猜测。", "g1_policy_contract.yaml"),
    "Observation": ("Policy 的一次观测输入：三路 RGB、16-D 机器人状态和语言指令。必须有同步时间戳并在最大 age 内提交。", "g1_policy_contract.yaml"),
    "observation": ("Policy 的一次观测输入：三路 RGB、16-D 机器人状态和语言指令。必须有同步时间戳并在最大 age 内提交。", "g1_policy_contract.yaml"),
    "resize_with_pad": ("保持宽高比缩放后用零填充到 224×224 的图像预处理。真机必须与训练/OpenPI 处理一致，不能改成 center crop。", "https://github.com/Physical-Intelligence/openpi"),
    "action chunk": ("Policy 一次生成的未来动作序列。本项目 canonical shape 是 32×16；实际执行只允许经过审核的 committed prefix。", "g1_policy_contract.yaml"),
    "Action horizon": ("一次 policy 输出中未来动作步数。本项目固定为 32；改变 horizon 会改变时序与动作语义。", "g1_policy_contract.yaml"),
    "horizon": ("一次 policy 输出覆盖的未来动作步数。本项目 canonical action horizon 固定为 32。", "g1_policy_contract.yaml"),
    "Adaptive-OFF": ("关闭自适应调速的基础闭环。必须先证明它能稳定抓取，之后才能与 Adaptive-ON 做公平配对。", "REAL_LGG100_ADAPTIVE_WORKFLOW.md"),
    "Adaptive-ON": ("开启 phase/clearance-aware retiming 的实验组。只改变时间安排，不应改变 LGG100 action path。", "REAL_LGG100_ADAPTIVE_WORKFLOW.md"),
    "Adaptive": ("根据距离、阶段、接触风险和 stale 状态动态改变动作 timing 的模块。远处可加速，接近抓取时应减速。", "adaptive_speed_context.py"),
    "retiming": ("在不改变几何路径的前提下重新分配动作时间戳，从而加速自由空间段并减速接触段。", "adaptive_retimer.py"),
    "Swept Path": ("Swept-path preflight：检查从当前关节状态连续移动到目标期间整个扫掠体，而不仅检查终点是否碰撞。", "swept_path_preflight.py"),
    "swept-path": ("连续路径扫掠检查：验证中间插值状态的碰撞、关节限制和 IK，而不是只验证终点。", "swept_path_preflight.py"),
    "collision": ("碰撞检查：检测机器人自身、桌面、方块或禁入几何之间的接触/穿透。任何 forbidden collision 都应拒绝执行。", "swept_path_preflight.py"),
    "fail-closed": ("故障关闭原则：缺少证据、超时、异常或验证失败时默认 hold/拒绝，而不是继续执行。", "https://en.wikipedia.org/wiki/Fail-safe"),
    "commit": ("动作提交：安全预检全部通过后，控制链接受某个新鲜 action prefix 的时刻。commit age 是 observation 采集到该时刻的完整延迟。", "lgg100_quarantined_closed_loop.py"),
    "P50": ("第 50 百分位（中位数）延迟；一半样本更快、一半更慢。不能单独代表尾部实时风险。", "https://en.wikipedia.org/wiki/Percentile"),
    "P95": ("第 95 百分位延迟；95% 样本不超过该值。用于观察尾部延迟。", "https://en.wikipedia.org/wiki/Percentile"),
    "P99": ("第 99 百分位延迟；用于暴露少量但危险的长尾停顿。", "https://en.wikipedia.org/wiki/Percentile"),
    "watchdog": ("看门狗：持续检查 observation/action freshness、控制周期和通信状态；超时必须触发 hold。无动作执行不能算验证通过。", "https://en.wikipedia.org/wiki/Watchdog_timer"),
    "stale": ("数据过期：observation、feedback 或 action 的 age 超过注册阈值。stale 数据不能用于新动作提交。", "lgg100_quarantined_closed_loop.py"),
    "LowState": ("Unitree SDK2 的低层只读状态消息，包含 mode、tick、IMU 和 motor_state。本项目只允许 subscriber-only 读取。", "https://github.com/unitreerobotics/unitree_sdk2_python"),
    "SDK2": ("Unitree 第二代官方开发套件。项目固定官方 Python commit，并禁止直接运行含运动 Publisher 的示例。", "https://github.com/unitreerobotics/unitree_sdk2_python"),
    "SDK": ("Software Development Kit，厂商提供的接口、消息定义和示例。示例可能包含真实运动，必须逐项审计。", "https://github.com/unitreerobotics/unitree_sdk2_python"),
    "DDS": ("Data Distribution Service，Unitree 使用的实时发布/订阅通信中间件。读取 LowState 和发送 LowCmd 都通过 DDS，但权限和风险完全不同。", "https://www.omg.org/omg-dds-portal/"),
    "CycloneDDS": ("Eclipse 的 DDS 实现，Unitree SDK2 Python 的底层依赖。网卡、domain 和版本必须与机器人环境匹配。", "https://cyclonedds.io/"),
    "IDL": ("Interface Definition Language，定义 DDS 消息字段和类型。G1 使用 unitree_hg IDL。", "https://www.omg.org/spec/IDL/"),
    "IMU": ("惯性测量单元：提供姿态、角速度和加速度。四元数顺序、坐标系和单位必须在真机确认。", "https://en.wikipedia.org/wiki/Inertial_measurement_unit"),
    "ChannelSubscriber": ("Unitree SDK2 的 DDS 订阅端，只读取指定 topic。本项目只读 adapter 允许使用它。", "https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/core/channel.py"),
    "ChannelPublisher": ("Unitree SDK2 的 DDS 发布端，可向机器人 topic 写消息。当前项目硬件 Gate 未通过，禁止创建。", "https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/core/channel.py"),
    "LowCmd": ("Unitree 低层电机命令消息，可直接影响关节。当前没有获准的 command adapter，禁止发送。", "https://github.com/unitreerobotics/unitree_sdk2_python"),
    "arm_sdk": ("Unitree G1 高层手臂控制 topic。官方示例会启用并发布手臂轨迹；当前项目禁止使用。", "https://github.com/unitreerobotics/unitree_sdk2_python/tree/master/example/g1"),
    "Shadow": ("影子模式：真实传感器进入完整 pipeline，但建议动作只记录、不下发，用于验证 parity、时序和安全拒绝。", "https://en.wikipedia.org/wiki/Shadow_system"),
    "HIL": ("Hardware-in-the-Loop，硬件在环：让真实硬件/控制器参与测试，但保持受控边界和可验证停止条件。", "https://en.wikipedia.org/wiki/Hardware-in-the-loop_simulation"),
    "E-stop": ("Emergency Stop，独立急停装置。必须由现场操作员可立即触发，且软件 watchdog 不能替代它。", "https://en.wikipedia.org/wiki/Kill_switch"),
    "Damping": ("Unitree 的阻尼安全模式：关节保持阻尼/柔顺而非执行任务轨迹。SSH 登录和只读检查不得改变该模式。", "https://support.unitree.com/"),
    "ControlMaster": ("OpenSSH 连接复用：多个 SSH 会话共享一条主 TCP 连接，减少不稳定的重复握手。它不提供机器人动作安全。", "https://man.openbsd.org/ssh_config#ControlMaster"),
    "tunnel": ("SSH 端口转发：本地 127.0.0.1:8000 映射到 L40S loopback。隧道存在不代表远端 policy server 正在监听。", "https://man.openbsd.org/ssh#L"),
    "strict restore": ("严格 checkpoint 恢复：参数树、shape、revision 和 normalization 必须完全匹配；不允许静默忽略 missing/extra 参数。", "lgg100_candidate_server.py"),
    "metadata/hash": ("元数据与哈希绑定：把 checkpoint、contract、norm、代码和证据文件固定到可复核标识，防止混用结果。", "lgg100_candidate_server.py"),
    "task success": ("任务成功：完整抓取/放置达到预注册条件。输出有限、IK 通过或没有碰撞都不能单独等价为任务成功。", "REAL_LGG100_ADAPTIVE_WORKFLOW.md"),
    "closed loop": ("闭环：每周期读取新反馈、重新推理/规划、执行并再次验证；与只播放预存动作的 open-loop 不同。", "https://en.wikipedia.org/wiki/Closed-loop_controller"),
    "RTT": ("Round-Trip Time，网络往返延迟。高 RTT/jitter 会降低远程观测与控制的 freshness。", "https://en.wikipedia.org/wiki/Round-trip_delay"),
    "GPU": ("图形处理器；L40S 用于 LGG100 推理。瞬时 utilization 低不等于显存和设备未被其他任务占用。", "https://www.nvidia.com/en-us/data-center/l40s/"),
    "L40S": ("NVIDIA 数据中心 GPU；本项目用它承载 LGG100 推理。只有完全空闲或管理员明确分配的设备才能启动服务。", "https://www.nvidia.com/en-us/data-center/l40s/"),
    "Monte Carlo": ("蒙特卡洛测试：对物体、相机、摩擦、延迟等随机变量运行大量可复现 seed，用统计结果评估成功率和安全边界。", "https://en.wikipedia.org/wiki/Monte_Carlo_method"),
    "Simulation/Replay": ("Simulation 运行物理仿真；Replay 确定性重放记录的 observation/state/action，用于在不控制真机时验证相同 pipeline。", "https://en.wikipedia.org/wiki/Robotics_simulator"),
    "replay": ("确定性重放已记录的传感器和动作数据，使解析、FK、IK、安全和 watchdog 回归可以复核。", "https://en.wikipedia.org/wiki/Record_and_replay"),
    "mock sink": ("模拟动作接收端：记录或在 MuJoCo 中执行获准目标，但不创建 Unitree Publisher，也不能向真机发送命令。", "https://en.wikipedia.org/wiki/Mock_object"),
    "ground truth": ("已知真值：合成测试中预先知道的相机、EEF 或场景参数，用来量化标定算法能否无偏恢复参数。", "https://en.wikipedia.org/wiki/Ground_truth"),
    "Supervisor": ("安全监督器：独立检查 freshness、limits、碰撞、模式和停止条件；任何检查失败都必须阻断动作。", "https://en.wikipedia.org/wiki/Supervisory_control"),
    "Physical-prompt": ("GEN-1.5 风格的短时 sensorimotor demonstration 片段；当前只建立记录/replay 格式，不声称 LGG100 支持该能力。", "https://generalistai.com/blog/gen-1.5"),
}
_TERM_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    + "|".join(re.escape(term) for term in sorted(TERM_INFO, key=len, reverse=True))
    + r")(?![A-Za-z0-9_])"
)


def linked_text(value: str) -> str:
    """Escape text and wrap registered terminology in accessible hyperlinks."""
    output: list[str] = []
    cursor = 0
    for match in _TERM_PATTERN.finditer(value):
        output.append(html.escape(value[cursor:match.start()]))
        term = match.group(0)
        tip, url = TERM_INFO[term]
        output.append(
            '<a class="term" href="{}" target="_blank" rel="noopener noreferrer" '
            'data-tip="{}" aria-label="{}：{}">{}</a>'.format(
                html.escape(url, quote=True),
                html.escape(tip, quote=True),
                html.escape(term, quote=True),
                html.escape(tip, quote=True),
                html.escape(term),
            )
        )
        cursor = match.end()
    output.append(html.escape(value[cursor:]))
    return "".join(output)


def load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text())


def bullet_cell(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{linked_text(item)}</li>" for item in items) + "</ul>"


def status_pill(status: str, label: str) -> str:
    return f'<span class="pill {html.escape(status)}"><i></i>{html.escape(label)}</span>'


def table_row(row: dict[str, Any]) -> str:
    return f"""
    <tr class="{html.escape(row['status'])}">
      <td class="stage"><span class="domain">{html.escape(row['domain'])}</span><b>{html.escape(row['id'])}</b><strong>{linked_text(row['title'])}</strong></td>
      <td class="state">{status_pill(row['status'], row['label'])}</td>
      <td>{bullet_cell(row['done'])}</td>
      <td>{bullet_cell(row['result'])}</td>
      <td>{bullet_cell(row['meaning'])}</td>
      <td>{bullet_cell(row['missing'])}</td>
      <td>{bullet_cell(row['next'])}</td>
      <td class="evidence">{bullet_cell(row['evidence'])}</td>
    </tr>
    """.strip()


PRE_REAL_SIMULATION_PRIORITIES = [
    ("1", "S4 IK / Swept Path", "把候选 IK 参数扩展到完整 32-step、连续多 chunk、随机可达目标；检查关节连续性、限制、自碰撞和桌面碰撞；加入已知碰撞负例", "合法路径全部满足 ≤5 mm/≤3°；禁入路径全部拒绝；不放宽阈值"),
    ("2", "S5 Adaptive-OFF", "IK 通过后，用真实 LGG100 chunk 跑完整 observation→policy→IK→preflight→commit→physics 闭环", "不再是 0 cycles；稳定完成抓取/堆叠；fresh commit 后 watchdog 有真实证据"),
    ("3", "S6 实时性", "分别优化 render、context、IK、collision 和 commit；注入 GPU/network jitter；测 P50/P95/P99", "完整 observation-to-commit 不超过 100 ms，stale action 必须 hold"),
    ("4", "S3 Adaptive Timing", "修复 Mixed transition coverage；在完全相同 seed、初态和 action path 上做 OFF/ON 配对", "Adaptive-ON 缩短任务时间，同时不降低成功率、不增加碰撞和接触风险"),
    ("5", "S7 随机化", "随机 cube pose/mass/friction、桌面、光照、相机偏移；运行多 seed Monte Carlo", "预注册成功率、碰撞率、IK rejection 和完成时间阈值后再运行"),
    ("6", "S7 故障注入", "模拟图像丢帧、冻结、乱序、NaN、LowState stale、policy timeout、断线和 watchdog 超时", "所有危险故障都只能进入 hold，不能 commit 新动作"),
    ("7", "Simulation Shadow/HIL", "用 MuJoCo 生成假的 Unitree LowState，走 LowState→FK→current-pose IK→safety 全链；使用 mock sink，禁止 Publisher", "current-pose round-trip 正确；断线/缺电机/错误 mode 时 fail-closed"),
    ("8", "标定工具预验证", "用已知 ground truth 的合成相机/EEF/桌面数据验证标定算法、单位、frame 和不确定度", "能恢复已知外参并正确拒绝高残差数据；但不能把它标为真实物理标定通过"),
    ("9", "安全 Supervisor", "在模拟 command sink 中验证 velocity/acceleration/joint/contact limits、E-stop、通信丢失和 mode mismatch", "每种故障都有确定的 hold/abort 状态和可回放证据；不实现真实 SDK Publisher"),
]

PRE_REAL_IMMEDIATE = [
    "S4 完整 IK 和碰撞回归",
    "随机目标与已知碰撞负例",
    "故障注入框架",
    "Synthetic LowState Shadow/HIL",
    "标定算法的 synthetic ground-truth 测试",
    "模拟安全 supervisor 和 no-command sink",
]

PRE_REAL_WAIT_L40S = [
    "重新生成完整真实 LGG100 action chunks",
    "Adaptive-OFF 多轮闭环",
    "OFF/ON 成对比较",
    "完整端到端 P95/P99 延迟测试",
]

PRE_REAL_NOT_REPLACEABLE = [
    "三相机真实内外参和同步",
    "Dex1 零点、方向和范围",
    "真实 EEF offset",
    "Unitree LowState 真值和 DDS freshness",
    "torque/current/temperature 限制",
    "E-stop、balance、stance 和真实通信丢失",
    "左腕相机物理完整性",
]


def numbered_cell(items: list[str]) -> str:
    return "<ol>" + "".join(f"<li>{linked_text(item)}</li>" for item in items) + "</ol>"


def stage_table(rows: list[dict[str, Any]], label: str) -> str:
    return f"""
    <div class="table-shell" role="region" aria-label="{html.escape(label)}" tabindex="0">
      <table class="detail-table">
        <thead><tr><th>阶段</th><th>状态</th><th>已完成工作</th><th>客观结果 / 数据</th><th>证据含义</th><th>未完成项</th><th>后续动作</th><th>证据</th></tr></thead>
        <tbody>{''.join(table_row(row) for row in rows)}</tbody>
      </table>
    </div>
    """.strip()


def speed_evidence_cards(rows: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    sections = (
        ("done", "已实现"),
        ("result", "当前验证数据"),
        ("missing", "还剩什么"),
        ("meaning", "证据含义"),
        ("next", "后续动作"),
        ("evidence", "证据文件"),
    )
    for row in rows:
        content = "".join(
            f'<section class="speed-detail-section {html.escape(key)}">'
            f'<h4>{label}</h4>{bullet_cell(row[key])}</section>'
            for key, label in sections
        )
        cards.append(
            f'<article class="speed-evidence-card {html.escape(row["status"])}" '
            f'data-speed-row="{html.escape(row["id"])}">'
            '<header class="speed-card-header">'
            f'<span class="speed-id">{html.escape(row["id"])}</span>'
            f'<div><small>{html.escape(row["domain"])}</small><h3>{linked_text(row["title"])}</h3></div>'
            f'{status_pill(row["status"], row["label"])}'
            '</header>'
            f'<div class="speed-card-grid">{content}</div></article>'
        )
    return "".join(cards)


def next_steps_panel() -> str:
    priority_rows = "".join(
        "<tr>"
        f"<td><span class=\"priority-number\">{linked_text(priority)}</span></td>"
        f"<td><b>{linked_text(stage)}</b></td>"
        f"<td>{linked_text(work)}</td>"
        f"<td>{linked_text(gate)}</td>"
        "</tr>"
        for priority, stage, work, gate in PRE_REAL_SIMULATION_PRIORITIES
    )
    execution_order = [
        "S4 完整 IK",
        "fresh-action watchdog",
        "Adaptive-OFF 成功闭环",
        "延迟 ≤100 ms",
        "随机化/故障注入",
        "simulated Shadow/HIL",
        "真实只读 LowState 和物理标定",
    ]
    flow = "".join(
        f'<div class="flow-step"><span>{index}</span><b>{linked_text(item)}</b></div>'
        for index, item in enumerate(execution_order, start=1)
    )
    return f"""
    <div class="section-heading"><div><span class="kicker">PRE-REAL EXECUTION PLAN</span><h2>真机前执行计划</h2><p>所有项目按证据依赖关系排序；计划项不代表对应 Gate 已通过。</p></div><span class="owner-chip">Owner · Shihua Yu</span></div>
    <div class="table-shell roadmap-shell" role="region" aria-label="真机前 Simulation 优先级" tabindex="0">
      <table class="roadmap-table"><thead><tr><th>优先级</th><th>对应阶段</th><th>可继续完成的工作</th><th>通过标准</th></tr></thead><tbody>{priority_rows}</tbody></table>
    </div>
    <div class="split-grid">
      <article class="info-card ready"><span class="card-label">AVAILABLE NOW</span><h3>当前可执行 · 不依赖空闲 GPU</h3>{numbered_cell(PRE_REAL_IMMEDIATE)}</article>
      <article class="info-card waiting"><span class="card-label">COMPUTE BLOCKED</span><h3>必须等 L40S 空闲</h3>{numbered_cell(PRE_REAL_WAIT_L40S)}<p class="card-note">启动条件：设备完全空闲或管理员明确分配；不得终止或挤占其他任务。</p></article>
    </div>
    <article class="flow-card"><span class="card-label">DEPENDENCY ORDER</span><h3>客观执行顺序</h3><div class="flow">{flow}</div><p>S4 完整 32-step sequential IK + swept-path 回归是当前最高优先级；该工作不涉及任何真机命令。</p></article>
    """.strip()


def objective_text(value: str) -> str:
    replacements = {
        "我们做了什么": "已完成工作",
        "这意味着什么": "证据含义",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return linked_text(value)


def objective_bullet_cell(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{objective_text(item)}</li>" for item in items) + "</ul>"


def updates_panel(updates: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    labels = (
        ("done", "已完成"),
        ("conclusions", "结论"),
        ("fixes", "修复"),
        ("remaining", "未完成"),
        ("next", "后续动作"),
        ("evidence", "证据"),
    )
    for index, update in enumerate(reversed(updates)):
        sections = "".join(
            f'<div class="update-block"><h4>{heading}</h4>{objective_bullet_cell(update.get(key, []))}</div>'
            for key, heading in labels
            if update.get(key)
        )
        open_attr = " open" if index == 0 else ""
        cards.append(
            f'<details class="update-card"{open_attr}>'
            f'<summary><span class="update-dot {html.escape(update["status"])}"></span>'
            f'<span><time>{html.escape(update["time"])}</time><b>{objective_text(update["title"])}</b></span>'
            '<span class="expand-icon" aria-hidden="true">＋</span></summary>'
            f'<div class="update-content"><p class="update-summary">{objective_text(update["summary"])}</p>'
            f'<div class="update-grid">{sections}</div></div></details>'
        )
    return "".join(cards)


def build() -> str:
    author = load("lgg100_author32_revalidation_summary.json")
    phase = load("lgg100_author32_t0_phase_speed_sweep_validation.json")
    closed = load("lgg100_author32_closed_loop_t0_offset008_baseline_realtime.json")
    ik = load("lgg100_author32_online_ik_parameter_sweep.json")
    tests = load("test_summary.json")
    sdk = load("unitree_g1_sdk2_readonly_audit_20260818.json")
    gpu = load("l40s_gpu_availability_20260819.json")
    gen15 = load("gen_1_5_relevance_review_20260820.json")
    deterministic_speed = load("g1_adaptive_phase_validation.json")
    retiming_safety = load("retiming_safety_validation.json")
    current_connectivity = load("current_robot_gpu_readonly_preflight_20260822.json")
    current_robot = load("g1_robot_connection_live_status_20260824.json")
    lowstate = load("g1_lowstate_readonly_capture_20260824.json")
    current_l40s = load("current_l40s_login_gpu_inventory_20260823.json")
    ab_gpu_plan = load("lgg100_adaptive_ab_gpu_experiment_plan.json")
    q0 = load("lgg100_slurm_q0_output_only_20260824.json")
    q05 = load("lgg100_slurm_q05_soak_status_20260824.json")
    updates = load("development_updates.json")

    inference = author["inference"]
    semantic = author["semantic_validation"]
    record = closed["records"][0]
    scenarios = {item["role"]: item for item in phase["scenarios"]}
    commit_age = record["observation_age_at_commit_ms"]
    maximum_age = record["maximum_observation_age_ms"]
    passing_ik = len(ik["passing_both_trials"])
    generated = datetime.now().astimezone().isoformat(timespec="seconds")

    rows = [
        {
            "domain": "SIMULATION", "id": "S0", "title": "G1 数据契约", "status": "pass", "label": "通过",
            "done": [
                "冻结 pelvis-frame state/action[16] 和 xyzw 四元数。",
                "冻结双臂 14 关节、双 Dex1、30 Hz 和 horizon 32。",
                "实现 contract metadata/hash 与 fail-closed validator。",
            ],
            "result": [
                "Contract ID: g1_edu_dual_dex1_eef_v1。",
                "Action semantics: absolute EEF target。",
                "错误 shape、NaN/Inf、错误 quaternion 会被拒绝。",
            ],
            "meaning": [
                "仿真、模型 client 和未来真机 adapter 有统一接口。",
                "其他机器人 schema 不能误入主链。",
            ],
            "missing": [
                "真实 LowState、相机和 Dex1 尚未证明满足同一 contract。",
                "真实 EEF site 与单位仍需硬件验证。",
            ],
            "next": ["保持 contract 冻结；任何硬件 profile 必须绑定相同 ID/SHA。"],
            "evidence": ["g1_policy_contract.yaml", "g1_policy_contract.py"],
        },
        {
            "domain": "SIMULATION", "id": "S1", "title": "MuJoCo Observation", "status": "pass", "label": "通过",
            "done": [
                "建立 G1 29-DOF、双臂、Dex1、桌面和方块场景。",
                "生成三路 480×640 RGB，经 resize_with_pad 到 224×224。",
                "从 MuJoCo FK 生成 pelvis-frame EEF state[16]。",
            ],
            "result": ["仿真 observation schema 与冻结 contract 一致。", "共享场景测量减少每个 chunk 的重复计算。"],
            "meaning": ["模型输入不再依赖临时 fixture 或其他机器人 action schema。"],
            "missing": ["真实相机域、时间戳、EEF 与仿真 parity 未验证。", "物理场景标定不能由 MuJoCo 代替。"],
            "next": ["用真实同步测量建立 hardware observation parity。"],
            "evidence": ["g1_mujoco_bridge.py", "g1_sim_speed_context.py", "MuJoCo tests"],
        },
        {
            "domain": "SIMULATION", "id": "S2", "title": "真实 LGG100 输出与语义", "status": "pass", "label": "离线通过",
            "done": [
                "固定 pi05_g1_eef、checkpoint revision 和 author horizon 32。",
                f"采集 {inference['samples']} 次真实神经输出并验证 shape/finite。",
                "比较 absolute/delta、xyzw/wxyz、左右顺序等语义假设。",
            ],
            "result": [
                f"{inference['finite_shape_passes']}/{inference['samples']} 为有限 [32,16]。",
                f"最佳语义为 {semantic['best_hypothesis']}，usable samples={semantic['usable_samples']}。",
                f"Warm inference P50={inference['warm_latency_ms']['p50']:.2f} ms，P95={inference['warm_latency_ms']['p95']:.2f} ms。",
            ],
            "meaning": ["模型能稳定输出符合基础 shape 的 EEF action chunk。", "这仍是离线证据，不证明闭环抓取成功。"],
            "missing": ["真实 observation 下的 task success。", "当前没有可安全启动 server 的完全空闲 L40S。"],
            "next": ["GPU 空闲后 strict restore，并先做 output-only metadata/hash probe。"],
            "evidence": ["results/lgg100_author32_revalidation_summary.json"],
        },
        {
            "domain": "SIMULATION", "id": "S3", "title": "Adaptive Timing", "status": "partial", "label": "部分通过",
            "done": ["实现 clearance/phase/stale-aware retiming。", "验证 near、mixed、far 三种距离场景。"],
            "result": [
                f"Near: {scenarios['near']['scale_range'][0]:.2f}×，保守接近。",
                f"Far: chunk duration 改善 {abs(scenarios['far']['duration_change_percent']):.1f}%。",
                f"Mixed: 最大 {scenarios['mixed']['scale_range'][1]:.2f}×，但 transition coverage=false。",
            ],
            "meaning": ["调速器展示了远处加速、近处减速。", "只改变 timing，不改变 LGG100 action path。"],
            "missing": ["Mixed grasp transition coverage。", "完整抓取总时间下降且成功率不降低的任务级证据。"],
            "next": ["先让 Adaptive-OFF 稳定抓取，再进行相同 action/初态的 OFF/ON 配对。"],
            "evidence": ["results/lgg100_author32_t0_phase_speed_sweep_validation.json"],
        },
        {
            "domain": "SIMULATION", "id": "S4", "title": "IK 与 Swept Path", "status": "blocked", "label": "阻塞",
            "done": [
                "实现双臂 IK、5 mm/3° Gate、关节限制和 sequential swept collision。",
                "对 committed target 做 0–250 iteration 收敛跟踪。",
                "扫描 damping 和 numerical step。",
            ],
            "result": [
                "30 iterations: 姿态误差约 6.89°/6.29°，超过 3°。",
                "60 iterations: 约 1.74°/1.59°，位置 <1 mm，无关节触限。",
                f"{passing_ik} 组候选在两个 sampled target 上通过。",
            ],
            "meaning": ["目标不是物理不可达，主要问题是数值收敛预算。", "当前 hold 是正确的安全拒绝。"],
            "missing": ["完整 32-step、多 chunk、连续性和随机目标回归。", "已知碰撞目标必须继续被拒绝。"],
            "next": ["完成 candidate IK sequential swept-path 全回归，不放宽 5 mm/3°。"],
            "evidence": ["results/lgg100_author32_online_ik_convergence_trace.json", "results/lgg100_author32_online_ik_parameter_sweep.json"],
        },
        {
            "domain": "SIMULATION", "id": "S5", "title": "Adaptive-OFF 闭环", "status": "blocked", "label": "未通过",
            "done": [
                "建立 observation→LGG100→preflight→commit/hold 的 quarantined closed loop。",
                "记录 action hash、IK rejection、stage latency 和 observation age。",
                "修复无新鲜动作也判 watchdog 通过的问题。",
            ],
            "result": [
                f"completed_cycles={closed['completed_cycles']}。",
                f"task_success={str(closed['task_success']).lower()}。",
                f"abort_reason={closed['abort_reason']}，cycle 0 未执行动作。",
            ],
            "meaning": ["Fail-closed 链有效。", "任务能力和稳定抓取尚未得到证明。"],
            "missing": ["Adaptive-OFF 稳定完成抓取/堆叠。", "fresh action commit 后的 watchdog 证据。"],
            "next": ["IK Gate 通过后重跑 instrumented Adaptive-OFF。"],
            "evidence": ["results/lgg100_author32_closed_loop_t0_offset008_baseline_realtime.json"],
        },
        {
            "domain": "SIMULATION", "id": "S6", "title": "端到端实时性", "status": "blocked", "label": "超时",
            "done": ["测量 render、inference、context、preflight 与 commit。", "将重复场景测量改为每周期一次。"],
            "result": [
                f"Warm inference P50={inference['warm_latency_ms']['p50']:.2f} ms。",
                f"Complete observation-to-commit={commit_age:.2f} ms。",
                f"Gate ≤{maximum_age:.0f} ms，当前超出 {commit_age-maximum_age:.2f} ms。",
            ],
            "meaning": ["不能只用推理延迟代表控制链延迟。", "过期 action 不得 commit。"],
            "missing": ["远端实测 context 优化后的完整链。", "继续减少约 9–20 ms，并验证 P95/P99。"],
            "next": ["GPU 可用后做分阶段 profile，并保持 watchdog fail closed。"],
            "evidence": ["closed-loop timing record", "g1_sim_speed_context benchmark"],
        },
        {
            "domain": "SIMULATION", "id": "S7", "title": "随机化与故障注入", "status": "todo", "label": "未完成",
            "done": ["已有 stale/hold、collision 和部分安全单元测试。"],
            "result": [f"完整本地自动测试 {tests['passed']}/{tests['total']} 通过。"],
            "meaning": ["代码回归通过不等于任务在随机物理环境中稳定。"],
            "missing": ["相机、物体、摩擦、网络 jitter、断线和传感器 stale 随机化。", "预注册成功率和安全阈值。"],
            "next": ["Adaptive-OFF 基线通过后运行 randomized/fault suite。"],
            "evidence": ["results/test_summary.json", "tests/"],
        },
        {
            "domain": "MODEL RESEARCH", "id": "M0", "title": "Generalist AI GEN-1.5 启发", "status": "partial", "label": "仅研究",
            "done": [
                "读取 Generalist AI 官方 GEN-1.5 发布全文及引用边界。",
                "核对 one-shot physical prompting、few-step adaptation、sim-to-real 与 physical generalization claims。",
                "比较其公开信息与当前 LGG100/G1 contract、延迟和安全链。",
            ],
            "result": [
                f"官方称 3–12 秒单次 demonstration 的 one-shot 平均成功率 {gen15['official_claims']['one_shot_average_success_percent']}%±{gen15['official_claims']['one_shot_stddev_percent']}%。",
                f"10 gradient steps/5 分钟数据平均 {gen15['official_claims']['ten_step_five_minute_average_success_percent']}%±{gen15['official_claims']['ten_step_five_minute_stddev_percent']}%。",
                "30 秒多模态 memory，输出 100 Hz action trajectories（不等于 100 Hz neural inference）。",
                "目前没有公开 weights、code、API、action schema、latency 或 Unitree G1 support。",
            ],
            "meaning": [
                "最重要启发是把同步 sensorimotor demonstration 当作 runtime context，而不只是训练集。",
                "Simulation rollout 未来可能成为 physical prompt，但只有 contract-matched 成功轨迹才有资格。",
                "更强的 improvisation 会产生新接触/transition，因此更需要 IK、swept collision 和 watchdog。",
            ],
            "missing": [
                "官方模型/API/schema/license 和独立复现。",
                "当前数据 recorder 还没有 3–12 秒 prompt segment、30 秒 context 和 phase/contact/failure 标签。",
                "GEN-1.5 无法证明与当前 bimanual EEF-16 contract drop-in compatible。",
            ],
            "next": [
                "GEN-1.5 保留为研究记录，不建设相关数据基础设施，也不纳入当前执行计划。",
                "若官方开放访问，只记录接口事实；不替换 LGG100，不连接真机动作。",
            ],
            "evidence": ["GEN_1_5_RELEVANCE.md", "results/gen_1_5_relevance_review_20260820.json", "https://generalistai.com/blog/gen-1.5"],
        },
        {
            "domain": "REAL ROBOT", "id": "H0", "title": "网络与认证 Shell", "status": "partial", "label": "仅连接",
            "done": [
                "机器人由操作员保持 Damping、安全绳/支撑和 E-stop 就绪。",
                "Mac→开发机 192.168.1.13→机器人 192.168.123.164 登录成功。",
                "建立独立 L40S ControlMaster、keeper 和 loopback tunnel。",
            ],
            "result": ["2026-08-24 当前复查：operator gateway 与机器人内部 authenticated nested shell 均已建立。", "只读 inventory：aarch64、Ubuntu 22.04、Tegra 5.15；隔离 subscriber-only Python environment 已按 pinned SDK/adapter 建立。", f"真实 LowState 固定读取：{lowstate['capture']['captured_samples']}/{lowstate['capture']['requested_samples']} samples，Publisher/command=false。", f"robot_connection_available={str(current_robot['decision']['robot_connection_available']).lower()}。"],
            "meaning": ["当前具备真实机器人内部 read-only shell 和真实反馈证据。", "连接与 LowState 成功仍不等于 pelvis-frame FK、zero-motion Shadow 或动作权限；g1_execution_enabled=false。"],
            "missing": ["处理 reader take-sample error、duplicate ticks 与 mode/unit semantics。", "补充 waist state 后完成 LowState→FK→current-pose IK→safety zero-motion evidence。", "机器人内部实时 supervisor 与稳定链路测试。"],
            "next": ["先扩展 subscriber-only capture contract，加入 waist state 并离线回归；不得创建 Publisher。"],
            "evidence": ["results/g1_lowstate_readonly_capture_20260824.json", "results/g1_robot_connection_live_status_20260824.json", "UNITREE_G1_SDK_READONLY.md"],
        },
        {
            "domain": "REAL ROBOT", "id": "H1", "title": "Unitree SDK / LowState", "status": "partial", "label": "本地完成",
            "done": [
                f"固定官方 unitree_sdk2_python {sdk['official_sources']['python']['commit'][:10]}…。",
                "确认 G1 使用 unitree_hg、rt/lowstate，双臂索引 15–28。",
                "实现 subscriber-only adapter 和 AST 禁写测试。",
            ],
            "result": [f"真实 rt/lowstate 读取 {lowstate['capture']['captured_samples']}/{lowstate['capture']['requested_samples']} samples；callback overflow={lowstate['capture']['callback_overflow']}。", f"接收 gap median/max={lowstate['capture']['timing_gap_ms']['median']:.3f}/{lowstate['capture']['timing_gap_ms']['max']:.3f} ms；duplicate ticks={lowstate['capture']['tick']['duplicate_sample_count']}。", "观察到一次 reader take-sample error；Publisher、mode change、robot command 均为 false。"],
            "meaning": ["subscriber-only 真实反馈链已首次闭合。", "DDS freshness 和 state contract 仍是部分通过，不能升级 zero-motion 或动作权限。"],
            "missing": ["官方 units、IMU quaternion order、mode semantics 的物理确认。", "waist motor state 与 pelvis-frame FK parity。", "重复批次、stale/断线 fault injection。"],
            "next": ["先修改并本地测试 subscriber schema，加入 waist state；再进行第二个固定只读 capture。"],
            "evidence": ["results/g1_lowstate_readonly_capture_20260824.json", "UNITREE_G1_SDK_READONLY.md", "g1_unitree_lowstate.py"],
        },
        {
            "domain": "REAL ROBOT", "id": "H2", "title": "相机、Dex1 与物理标定", "status": "blocked", "label": "缺数据",
            "done": ["冻结真实 observation 应满足的三相机、EEF 与 Dex1 schema。", "记录左腕相机损坏并粘回的风险。"],
            "result": ["physical_scene_calibration_verified=false。", "没有真实同步相机/关节/EEF/桌面/方块测量。"],
            "meaning": ["当前不能证明真机输入与训练/仿真一致。", "有画面也不代表外参和时序正确。"],
            "missing": ["三相机内外参和时间同步。", "pelvis、EEF offset、桌面、方块、Dex1 零点/方向。", "左腕相机完整性验证。"],
            "next": ["采集真实同步 calibration dataset，并生成 hash-bound calibration report。"],
            "evidence": ["g1_policy_contract.yaml hardware blockers", "operator-provided G1 guide"],
        },
        {
            "domain": "REAL ROBOT", "id": "H3", "title": "L40S Policy 服务", "status": "blocked", "label": "无空卡",
            "done": ["建立 127.0.0.1:8000 loopback-only tunnel。", "只读监控 8 张 L40S 和 resident process。"],
            "result": [
                "2026-08-23 当前复查：L40S 认证 shell 已在 Herdr w1:p3 建立并保持活动。",
                f"8 张 L40S 均为 46,068 MiB total；实时 free range={current_l40s['summary']['minimum_free_memory_mib']/1024:.1f}–{current_l40s['summary']['maximum_free_memory_mib']/1024:.1f} GiB。",
                "每卡各有 1 个既有 compute process，约占 34.2 GiB；瞬时 utilization=0% 不等于空闲。",
                "本地 8000 tunnel 已监听，但远端 policy port 8000 关闭，LGG100 server 不可用。",
                f"Q0 已在 1 张 Slurm 分配的空闲 L40S 上完成：{q0['output_only_probe']['finite_shape_passes']}/{q0['output_only_probe']['formal_draws']} finite [32,16]，实际运行 {q0['scheduler']['actual_runtime_s']} 秒并提前释放。",
                f"Q0 warm latency P50/P95={q0['output_only_probe']['latency_ms']['p50']:.2f}/{q0['output_only_probe']['latency_ms']['p95']:.2f} ms；peak VRAM={q0['output_only_probe']['peak_memory_used_mib']/1024:.2f} GiB。",
                f"raw quaternion exact-unit={q0['output_only_probe']['raw_quaternion_exact_unit_passes']}/{q0['output_only_probe']['formal_draws']}；官方 consumer post-processing={q0['output_only_probe']['official_consumer_postprocess_passes']}/{q0['output_only_probe']['formal_draws']}。",
                f"独立 Q0.5 当前 RUNNING：1 张 L40S，Near/Mixed/Far，{q05['workload']['target_rate_hz']} Hz，目标真实 inference {q05['scheduler']['target_inference_duration_s']/3600:.0f} 小时。",
                f"首个 heartbeat：{q05['latest_heartbeat']['completed_calls']} 次调用，{q05['latest_heartbeat']['finite_shape_passes']}/{q05['latest_heartbeat']['completed_calls']} finite，GPU utilization={q05['latest_heartbeat']['allocated_gpu_utilization_percent']}%。",
            ],
            "meaning": ["冻结 checkpoint strict restore 与 Q0 output-only inference 通过。", "Yuhao pinned deployment code 明确在 IK 前归一化预测 quaternion；Q0 官方 consumer boundary 30/30 通过。", "Q0.5 用真实多场景 inference 调查 endurance，不是 dummy occupancy；运行中 heartbeat 不是最终结果，g1_sim_eligible=false。"],
            "missing": ["Q0.5 完成四小时或 fail-closed 后的最终汇总。", "固定 deployment code 中 15 Hz 默认值与 30 Hz help text 冲突的实验 manifest。", "完整 32-step、多 chunk sequential IK、swept-path 和 commit latency ≤100 ms。"],
            "next": ["不干预正在运行的 bounded job；完成后冻结结果，再继续 G3，前置 Gate 通过后才申请 Q1。"],
            "evidence": ["results/yuhao_g1_client_deployment_audit_20260824.json", "results/lgg100_slurm_q05_soak_status_20260824.json", "results/lgg100_slurm_q0_output_only_20260824.json", "LGG100_ADAPTIVE_AB_GPU_EXECUTION_PLAN_CN.md"],
        },
        {
            "domain": "REAL ROBOT", "id": "H4", "title": "Zero-motion Shadow / HIL", "status": "todo", "label": "首个允许测试",
            "done": ["定义 current-pose hold：LowState→FK→当前 EEF→IK→safety，不 publish。", "定义 no-command 证据字段。"],
            "result": ["publisher_created=false 为必须条件。", "robot_command_sent=false，mode_changed=false。", "尚未实际运行。"],
            "meaning": ["这是把仿真安全链连接到真实数据的最低风险步骤。"],
            "missing": ["真实 LowState 与 FK parity。", "真实 stale/断线/watchdog no-command 测试。"],
            "next": ["完成 H1/H2 后运行 zero-motion Shadow，并独立审阅报告。"],
            "evidence": ["results/g1_first_motion_readiness_decision_20260818.json"],
        },
        {
            "domain": "REAL ROBOT", "id": "H5", "title": "Hardware Command 与安全限制", "status": "blocked", "label": "未实现",
            "done": ["注册 hardware_execution_allowed=false。", "明确禁止官方 low-level/arm7 运动示例和任何未经审查 Publisher。"],
            "result": ["Unitree command adapter 不存在。", "官方 torque/current/temperature/controller limits 未接入。", "real_time_watchdog_validated=false。"],
            "meaning": ["当前没有可审计方式把关节目标安全发送给 G1。", "从 Damping 切模式本身也是未授权动作。"],
            "missing": ["正确的官方控制接口与 hardware profile。", "本地 watchdog、E-stop、通信丢失、stance/balance、feedback。"],
            "next": ["Shadow/HIL 全通过后，单独设计并审查 deterministic command adapter。"],
            "evidence": ["g1_policy_contract.yaml", "results/unitree_g1_sdk2_readonly_audit_20260818.json"],
        },
        {
            "domain": "REAL ROBOT", "id": "H6", "title": "真机闭环与抓取加速", "status": "blocked", "label": "禁止执行",
            "done": ["定义 staged path：单臂→双臂→桌面→轻物体→Adaptive-OFF→Adaptive-ON。"],
            "result": [
                "g1_contract_verified=false。",
                "g1_sim_eligible=false。",
                "g1_execution_enabled=false。",
                "hardware_execution_performed=false。",
            ],
            "meaning": ["当前只能继续 read-only、subscriber-only 和 zero-motion 工作。"],
            "missing": ["S4–S7、H1–H5 全部通过。", "Adaptive-OFF 稳定任务基线和相同 action/初态的 ON/OFF 对比。"],
            "next": ["先完成 IK 全路径与 Unitree 阶段 A inventory；不要发送动作。"],
            "evidence": ["validation_report.html", "simulation_deployment_summary.html"],
        },
    ]

    deterministic_coverage = deterministic_speed["coverage"]
    deterministic_comparison = deterministic_speed["comparison"]
    speed_rows = [
        {
            "domain": "SPEED MODULE", "id": "V0", "title": "模块边界与不变量", "status": "pass", "label": "已验证",
            "done": [
                "速度模块只重写 timestamps，不修改 LGG100 EEF action samples 或几何路径。",
                "对 retimed action 使用 np.array_equal 检查 byte-identical。",
                "Adaptive 模块不授予 Simulation 或硬件执行权限。",
            ],
            "result": ["path_actions_byte_identical=true。", "检测到 action sample 改动时返回 hold。"],
            "meaning": ["已证明 retiming 与路径生成相互隔离。", "速度变化不能绕过 contract、IK、碰撞或 watchdog。"],
            "missing": ["真实控制器插值和时间戳消费方式尚未验证。"],
            "next": ["保持路径不变量；未来 controller adapter 必须重复验证。"],
            "evidence": ["adaptive_speed_context.py", "tests/test_adaptive_speed_context.py"],
        },
        {
            "domain": "SPEED MODULE", "id": "V1", "title": "Context 输入与阶段识别", "status": "pass", "label": "已验证",
            "done": [
                "输入 task phase、EEF-to-cube clearance、contact、EEF/gripper tracking error、observation/policy age。",
                "输入 IK/joint margin、pelvis stability、network timeout 和 preflight/limits 状态。",
                "MuJoCo context 将 free_space、approach、grasp 和 place 映射为可审计 evidence。",
            ],
            "result": ["参考场景识别为 approach。", "方块移远后识别为 free_space。", "夹爪闭合意图识别为 grasp。"],
            "meaning": ["速度决策已由单一距离规则扩展为 phase/clearance/tracking/contact/freshness 联合决策。"],
            "missing": ["lift/retreat 的真实任务级 transition coverage。", "真实传感器与 contact 信号。"],
            "next": ["在完整 Adaptive-OFF 轨迹上验证连续 phase transition。"],
            "evidence": ["g1_sim_speed_context.py", "tests/test_g1_sim_speed_context.py"],
        },
        {
            "domain": "SPEED MODULE", "id": "V2", "title": "Fail-closed Hold 条件", "status": "pass", "label": "已验证",
            "done": ["实现 non-finite、unknown phase、network timeout、stale observation/policy、IK/collision/limits failure 的 hold。", "实现 clearance、tracking error、gripper error、kinematic margin 和 pelvis stability 硬阈值。"],
            "result": ["observation age >100 ms、policy age >500 ms、NaN、timeout 和 failed preflight 均拒绝 retiming。", "不连续且无法满足 envelope 的 path 被拒绝而不是误报通过。"],
            "meaning": ["缺失或危险 context 不会回退为高速执行。"],
            "missing": ["fresh committed action 之后的实时 watchdog 闭环证据。"],
            "next": ["在 S5 成功 commit 后注入 stale/timeout/collision fault。"],
            "evidence": ["adaptive_speed_context.py::decide_speed", "tests/test_adaptive_speed_context.py"],
        },
        {
            "domain": "SPEED MODULE", "id": "V3", "title": "阶段速度策略", "status": "pass", "label": "逻辑通过",
            "done": ["定义 free_space 最大 1.60×、approach 上限 0.70×、grasp/place 0.50×、lift 0.80×、retreat 1.00×。", "只有无 contact 的安全 free_space 允许加速；低 clearance 或 tracking/margin/stability 风险强制保守。", "scale increase 限制为每秒 2.0。"],
            "result": ["单元测试证明 safe free_space >1.0×。", "grasp、place、contact 和 low-clearance 均 ≤0.50×。"],
            "meaning": ["局部规则符合远处加速、近处和接触阶段减速的设计目标。"],
            "missing": ["真实 LGG100 mixed chunk 未覆盖 grasp transition。"],
            "next": ["用包含 free_space→approach→grasp 的同一完整轨迹复测。"],
            "evidence": ["ContextRetimerConfig", "tests/test_adaptive_speed_context.py"],
        },
        {
            "domain": "SPEED MODULE", "id": "V4", "title": "运动包络与平滑限制", "status": "pass", "label": "仿真通过",
            "done": ["计算双手 EEF speed/acceleration/jerk、angular speed 和 gripper speed。", "迭代缩小 scale，直到所有 simulation MotionEnvelope ratios ≤1。", "跨 sample 限制 scale 上升并跨 chunk 保存 previous_scale。"],
            "result": ["安全 free-space fixture 在保持 action byte-identical 时满足 EEF motion envelope。", "不可行 discontinuity 返回 motion_envelope_infeasible_at_safety_minimum_scale。"],
            "meaning": ["速度模块不会只依据目标倍率而忽略轨迹导数。"],
            "missing": [f"当前 envelope 来源为 {retiming_safety['motion_envelope']['source']}，不是官方 G1 hardware limits。", "真实 controller jerk/torque/current 限制。"],
            "next": ["取得官方 hardware profile 后重新注册 limits；此前仅用于 Simulation。"],
            "evidence": ["adaptive_speed_context.py::_motion_metrics", "results/retiming_safety_validation.json"],
        },
        {
            "domain": "SPEED MODULE", "id": "V5", "title": "MuJoCo 状态测量与 Context 性能", "status": "pass", "label": "本地通过",
            "done": ["测量 Dex1-cube 最小距离、contact、双臂 joint-limit margin 和 pelvis stability。", "新增 SimulationStateSnapshot，使一个 action chunk 共享 action-independent scene measurement。"],
            "result": ["cached snapshot 与逐次直接测量的 context/evidence 完全一致。", "本地 context median 由 11.96 ms 降至 0.61 ms，记录改善约 11.35 ms。"],
            "meaning": ["已移除每个 32-step sample 重复执行 MuJoCo geometry/contact 查询的开销。"],
            "missing": ["L40S 端完整 observation-to-commit profile 复测。"],
            "next": ["GPU 可用后在真实闭环 timing report 中验证该改善。"],
            "evidence": ["g1_sim_speed_context.py::measure_simulation_state", "tests/test_g1_sim_speed_context.py", "results/development_updates.json"],
        },
        {
            "domain": "SPEED MODULE", "id": "V6", "title": "Deterministic far-to-near Coverage", "status": "partial", "label": "局部通过",
            "done": ["用 G1 contract deterministic 17 cm far-to-near fixture 比较 baseline 与 adaptive。", "检查 far/near coverage、path identity、EEF/joint filters 和 endpoint error。"],
            "result": [
                f"Far scale median={deterministic_coverage['far_scale_median']:.2f}×，max={deterministic_coverage['far_scale_max']:.2f}×。",
                f"Near scale median={deterministic_coverage['near_scale_median']:.2f}×；path_actions_byte_identical={str(deterministic_coverage['path_actions_byte_identical']).lower()}。",
                f"Adaptive simulated duration=4.056 s，baseline=3.634 s，反而增加 {deterministic_comparison['duration_delta_s']:.3f} s。",
            ],
            "meaning": ["远快近慢的局部行为成立。", "整体更快没有成立；该 fixture 不是 LGG100 task evidence。"],
            "missing": ["真实 LGG100 multi-chunk、任务成功和总体时间收益。"],
            "next": ["不把 deterministic success 写成 policy speedup；仅保留为局部机制证据。"],
            "evidence": ["results/g1_adaptive_phase_validation.json"],
        },
        {
            "domain": "SPEED MODULE", "id": "V7", "title": "真实 LGG100 单 Chunk Phase Sweep", "status": "partial", "label": "Near/Far 通过",
            "done": ["对 hash-bound LGG100 near、mixed、far single-chunk ensembles 运行 phase-aware retiming。", "Near/Far 与 Mixed transition 分开判定，禁止部分通过升级为整体通过。"],
            "result": [
                f"Near: clearance={scenarios['near']['clearance_m']:.5f} m，32/32 approach，scale=0.50×，duration {scenarios['near']['nominal_chunk_duration_s']:.4f}→{scenarios['near']['retimed_chunk_duration_s']:.4f} s，accepted=true。",
                f"Far: clearance={scenarios['far']['clearance_m']:.5f} m，scale={scenarios['far']['scale_range'][0]:.3f}–{scenarios['far']['scale_range'][1]:.3f}×，duration 改善 {abs(scenarios['far']['duration_change_percent']):.2f}%，accepted=true。",
                f"Mixed: scale={scenarios['mixed']['scale_range'][0]:.3f}–{scenarios['mixed']['scale_range'][1]:.3f}×，表面 duration 改善 {abs(scenarios['mixed']['duration_change_percent']):.2f}%，但 accepted=false。",
            ],
            "meaning": ["Near 保守减速和 Far 局部加速已由真实 LGG100 chunk 支持。", "Mixed 缺少 grasp phase 和 accelerate-then-slow coverage，因此不能声称完整 Adaptive 行为通过。"],
            "missing": ["同一 chunk 中的 free_space→approach→grasp transition。", "observation 与执行 initial state 尚未绑定。"],
            "next": ["等待完整任务 chunk 后重跑 mixed transition，不根据 25.44% 表面缩时宣称收益。"],
            "evidence": ["results/lgg100_author32_t0_phase_speed_sweep_validation.json"],
        },
        {
            "domain": "SPEED MODULE", "id": "V8", "title": "Adaptive-OFF 闭环基线", "status": "blocked", "label": "未建立",
            "done": ["闭环已连接 observation→LGG100→context→preflight→commit/hold，并支持 adaptive_retiming_enabled=false。", "记录 inference、preflight、commit age、action hash 和 abort reason。"],
            "result": [f"completed_cycles={closed['completed_cycles']}，task_success={str(closed['task_success']).lower()}。", f"abort_reason={closed['abort_reason']}；cycle 0 没有执行动作。", f"complete commit age={commit_age:.2f} ms > {maximum_age:.0f} ms。"],
            "meaning": ["Fail-closed refusal 正常。", "Adaptive-OFF 任务能力、fresh-action watchdog 和速度基线均未证明。"],
            "missing": ["S4 完整 IK/swept-path Gate。", "稳定抓取/堆叠和 ≤100 ms commit。"],
            "next": ["S4 通过且 L40S 可用后，先建立 Adaptive-OFF 多轮闭环基线。"],
            "evidence": ["results/lgg100_author32_closed_loop_t0_offset008_baseline_realtime.json"],
        },
        {
            "domain": "SPEED MODULE", "id": "V9", "title": "任务级 Adaptive-ON 资格", "status": "todo", "label": "禁止启用",
            "done": ["定义同 action、同初态、同 seed 的 OFF/ON 配对原则。", "定义只有成功率和安全不下降且完整任务时间改善时才允许启用。"],
            "result": ["controlled_phase_speed_behavior_passed=false。", "policy_task_quality_passed=false。", "production_adaptive_enabled=false。"],
            "meaning": ["当前证据只支持局部速度决策，不支持任务级 speedup 或硬件启用。"],
            "missing": ["Adaptive-OFF 稳定基线。", "Mixed transition coverage。", "多 seed 配对成功率、完成时间、碰撞/contact force 和 tail latency。", "官方硬件 limits、真实标定和 Shadow/HIL。"],
            "next": ["完成 S4→S5→S6→S7；之后才运行 Adaptive-ON 配对实验。"],
            "evidence": ["REAL_LGG100_ADAPTIVE_WORKFLOW.md", "results/lgg100_author32_t0_phase_speed_sweep_validation.json"],
        },
    ]

    simulation_rows = [row for row in rows if row["domain"] == "SIMULATION"]
    model_rows = [row for row in rows if row["domain"] == "MODEL RESEARCH"]
    hardware_rows = [row for row in rows if row["domain"] == "REAL ROBOT"]
    simulation_passed = sum(row["status"] == "pass" for row in simulation_rows)
    hardware_blocked = sum(row["status"] in {"blocked", "todo"} for row in hardware_rows)

    css = """
    :root{--bg:#070b14;--panel:#0e1727;--line:#ffffff17;--text:#eef5ff;--muted:#9aabc1;--cyan:#37d9e8;--green:#45dca1;--amber:#ffc75d;--red:#ff738d;--violet:#b69aff}*{box-sizing:border-box}html{color-scheme:dark}body{margin:0;background:radial-gradient(circle at 8% 0,#173d61 0,transparent 27%),radial-gradient(circle at 92% 0,#332268 0,transparent 25%),var(--bg);color:var(--text);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.5}.shell{width:min(1880px,calc(100% - 28px));margin:auto;padding:26px 0 60px}.hero{display:flex;justify-content:space-between;align-items:end;gap:24px;padding:28px;margin-bottom:16px;border:1px solid var(--line);border-radius:22px;background:#0e1727dd;box-shadow:0 28px 90px #0007;backdrop-filter:blur(18px)}.eyebrow{color:var(--cyan);font-size:11px;font-weight:900;letter-spacing:.16em}.hero h1{font-size:clamp(32px,4vw,58px);line-height:1;margin:10px 0 12px;letter-spacing:-.045em}.hero p{margin:0;color:var(--muted);max-width:1050px}.verdict{text-align:right;min-width:240px}.verdict b{display:block;color:var(--red);font-size:20px}.verdict small{color:var(--muted)}.legend{display:flex;gap:13px;flex-wrap:wrap;padding:12px 18px;color:var(--muted);font-size:12px}.legend span:before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}.legend .pass:before{background:var(--green)}.legend .partial:before{background:var(--amber)}.legend .blocked:before{background:var(--red)}.legend .todo:before{background:var(--violet)}.table-wrap{overflow:auto;max-height:calc(100vh - 210px);border:1px solid var(--line);border-radius:20px;background:#0b1220e8;box-shadow:0 28px 90px #0008}table{width:100%;min-width:1900px;border-collapse:separate;border-spacing:0;font-size:13px}caption{text-align:left;padding:15px 18px;color:var(--muted);border-bottom:1px solid var(--line)}thead{position:sticky;top:0;z-index:8;background:#131e31}th{text-align:left;padding:14px 15px;color:#b8c7db;font-size:11px;letter-spacing:.08em;text-transform:uppercase;border-bottom:1px solid #ffffff24}th:nth-child(1){width:200px}th:nth-child(2){width:100px}th:nth-child(3),th:nth-child(4),th:nth-child(5),th:nth-child(6),th:nth-child(7){width:270px}th:nth-child(8){width:230px}td{padding:16px 15px;vertical-align:top;border-bottom:1px solid #ffffff0d;border-right:1px solid #ffffff09;background:#0d1625aa}tr:hover td{background:#142138}tr.pass td:first-child{box-shadow:inset 4px 0 var(--green)}tr.partial td:first-child{box-shadow:inset 4px 0 var(--amber)}tr.blocked td:first-child{box-shadow:inset 4px 0 var(--red)}tr.todo td:first-child{box-shadow:inset 4px 0 var(--violet)}.stage{position:sticky;left:0;z-index:3;background:#101b2d!important}.stage .domain{display:block;color:var(--cyan);font-size:9px;font-weight:900;letter-spacing:.14em}.stage b{display:block;margin:6px 0;color:#7891af}.stage strong{display:block;font-size:16px}.state{position:sticky;left:200px;z-index:3;background:#101b2d!important}.pill{display:inline-flex;align-items:center;gap:6px;padding:6px 9px;border-radius:99px;font-size:11px;font-weight:850;white-space:nowrap}.pill i{width:6px;height:6px;border-radius:50%}.pill.pass{color:var(--green);background:#45dca116}.pill.pass i{background:var(--green)}.pill.partial{color:var(--amber);background:#ffc75d16}.pill.partial i{background:var(--amber)}.pill.blocked{color:var(--red);background:#ff738d16}.pill.blocked i{background:var(--red)}.pill.todo{color:var(--violet);background:#b69aff16}.pill.todo i{background:var(--violet)}ul{margin:0;padding-left:17px;color:var(--muted)}li+li{margin-top:8px}.evidence li{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:#7890ae;overflow-wrap:anywhere}a.term{color:#6fe6f0;text-decoration-line:underline;text-decoration-style:dotted;text-decoration-color:#6fe6f099;text-underline-offset:3px;font-weight:720;cursor:help}a.term:hover,a.term:focus-visible{color:#b6f8ff;background:#35d9e812;border-radius:4px;outline:none}.roadmap-section td{padding:20px 22px;background:linear-gradient(90deg,#14233b,#111a2e);color:#dcecff}.roadmap-section b{font-size:16px;color:#eef7ff}.roadmap-section h2{margin:0 0 12px;color:var(--cyan);font-size:17px}.roadmap-section p{margin:0 0 10px;color:var(--muted)}.roadmap-section ol{columns:2;column-gap:46px}.roadmap-section pre{display:inline-block;margin:0;padding:15px 18px;border:1px solid #37d9e844;border-radius:12px;background:#07101f;color:#dffcff;font:700 13px/1.75 ui-monospace,SFMono-Regular,Menlo,monospace}.roadmap-header th{position:static;background:#20304a;color:#7de7ef;border-right:1px solid #ffffff18}.roadmap-priority td{background:#101b2d;color:#aebfd4}.roadmap-priority td:first-child{color:var(--cyan);font-size:17px;font-weight:900;text-align:center}.roadmap-priority td:nth-child(2){color:#f1f6ff;font-weight:800}.term-tooltip{position:fixed;z-index:9999;width:min(430px,calc(100vw - 28px));padding:13px 15px;border:1px solid #6fe6f055;border-radius:13px;background:#07101ff5;color:#dcecff;box-shadow:0 18px 60px #000b;font-size:12px;line-height:1.55;pointer-events:none;opacity:0;transform:translateY(5px);transition:opacity .12s ease,transform .12s ease}.term-tooltip.visible{opacity:1;transform:translateY(0)}footer{text-align:center;color:#64758c;padding:22px;font-size:11px}@media(max-width:760px){.shell{width:calc(100% - 12px)}.hero{align-items:start;flex-direction:column;padding:20px}.verdict{text-align:left}.table-wrap{max-height:calc(100vh - 260px)}}
    /* 2026 tabbed product UI */
    :root{--bg:#070a11;--surface:#0d131f;--surface-2:#121b2a;--surface-3:#172236;--line:#ffffff14;--line-strong:#ffffff26;--text:#f4f7fb;--muted:#92a0b4;--cyan:#55d9e6;--blue:#6c8cff;--green:#4bd7a2;--amber:#f7c765;--red:#ff7188;--violet:#b49bff}
    html{scroll-behavior:smooth}body{min-height:100vh;background:radial-gradient(900px 600px at 8% -10%,#194f6966,transparent 68%),radial-gradient(760px 620px at 92% -8%,#4b327155,transparent 65%),linear-gradient(180deg,#080b12,#070a11 45%,#090d15);font-feature-settings:"tnum" 1,"ss01" 1}body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.16;background-image:linear-gradient(#ffffff05 1px,transparent 1px),linear-gradient(90deg,#ffffff05 1px,transparent 1px);background-size:44px 44px;mask-image:linear-gradient(to bottom,#000,transparent 72%)}.shell{position:relative;width:min(1600px,calc(100% - 36px));padding:26px 0 54px}.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;color:var(--muted);font-size:12px}.brand{display:flex;align-items:center;gap:10px;color:var(--text);font-weight:800;letter-spacing:.02em}.brand-mark{display:grid;place-items:center;width:30px;height:30px;border:1px solid #63e1ed55;border-radius:10px;background:linear-gradient(145deg,#4ce0ec22,#7d74ff22);color:var(--cyan);box-shadow:inset 0 0 18px #55d9e611}.top-meta{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}.mini-chip,.owner-chip{padding:7px 11px;border:1px solid var(--line);border-radius:999px;background:#ffffff06;color:#b9c5d5;font-size:11px;font-weight:750}.brief-link{text-decoration:none;transition:.18s ease}.brief-link:hover{border-color:#55d9e655;background:#55d9e60c;color:#c9f8fb}.hero{position:relative;overflow:hidden;align-items:center;min-height:270px;margin-bottom:18px;padding:42px;border-color:#ffffff1c;border-radius:30px;background:linear-gradient(135deg,#101b2bdd,#111425e8 58%,#17152add);box-shadow:0 30px 90px #0008,inset 0 1px #ffffff0b}.hero:after{content:"";position:absolute;width:370px;height:370px;right:-90px;top:-150px;border-radius:50%;background:radial-gradient(circle,#6d8cff38,transparent 66%);pointer-events:none}.hero-copy{position:relative;z-index:1;max-width:1000px}.eyebrow{display:flex;align-items:center;gap:9px;color:#84e7ef}.eyebrow:before{content:"";width:24px;height:1px;background:var(--cyan)}.hero h1{max-width:950px;margin:14px 0 16px;font-size:clamp(38px,5.2vw,78px);line-height:.96;letter-spacing:-.06em}.hero h1 span{background:linear-gradient(100deg,#f7fbff,#a9eaf0 48%,#b9adff);-webkit-background-clip:text;background-clip:text;color:transparent}.hero p{max-width:880px;font-size:15px;line-height:1.75}.hero-side{position:relative;z-index:1;display:flex;flex-direction:column;align-items:flex-end;gap:10px;min-width:260px}.decision-badge{padding:15px 18px;border:1px solid #ff71884a;border-radius:16px;background:#ff71880d;text-align:right}.decision-badge small{display:block;color:#ff9bad;font-size:10px;font-weight:900;letter-spacing:.13em}.decision-badge b{display:block;margin-top:4px;color:#ffd9df;font-size:17px}.author-card{padding:12px 16px;border:1px solid var(--line);border-radius:15px;background:#ffffff06;text-align:right}.author-card small{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.12em}.author-card b{font-size:14px}.workspace{display:grid;grid-template-columns:220px minmax(0,1fr);gap:22px;align-items:start}.tab-content{min-width:0}.tabs{position:sticky;top:18px;z-index:30;display:flex;flex-direction:column;gap:6px;margin:0;padding:9px;border:1px solid var(--line);border-radius:20px;background:#0b111cdb;box-shadow:0 18px 45px #0007;backdrop-filter:blur(20px)}.tab-label{padding:10px 11px 8px;color:#62728a;font-size:9px;font-weight:950;letter-spacing:.16em}.tab{position:relative;display:flex;align-items:center;justify-content:space-between;width:100%;appearance:none;border:0;border-radius:13px;padding:13px 13px 13px 16px;background:transparent;color:#8796aa;text-align:left;font:800 12px/1 inherit;letter-spacing:.01em;white-space:nowrap;cursor:pointer;transition:.18s ease}.tab:before{content:"";position:absolute;left:5px;top:50%;width:3px;height:0;border-radius:9px;background:var(--cyan);transform:translateY(-50%);transition:.18s ease}.tab:hover{color:#dfe8f3;background:#ffffff08}.tab[aria-selected="true"]{color:#f7fbff;background:linear-gradient(135deg,#263751,#202b43);box-shadow:inset 0 1px #ffffff12,0 8px 22px #0005}.tab[aria-selected="true"]:before{height:20px}.tab-count{display:inline-grid;place-items:center;min-width:22px;height:22px;margin-left:auto;padding:0 6px;border-radius:999px;background:#ffffff0b;color:#9debf1;font-size:10px}.tab-panel{display:none;animation:panel-in .24s ease}.tab-panel.active{display:block}@keyframes panel-in{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}.section-heading{display:flex;align-items:end;justify-content:space-between;gap:20px;margin:4px 2px 18px}.section-heading h2{margin:4px 0 4px;font-size:clamp(25px,3vw,38px);letter-spacing:-.035em}.section-heading p{margin:0;color:var(--muted);font-size:13px}.kicker,.card-label{color:var(--cyan);font-size:9px;font-weight:950;letter-spacing:.16em}.metric-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-bottom:14px}.metric-card{position:relative;overflow:hidden;min-height:148px;padding:20px;border:1px solid var(--line);border-radius:20px;background:linear-gradient(145deg,#111a28dd,#0c121ddd);box-shadow:inset 0 1px #ffffff09}.metric-card:after{content:"";position:absolute;width:90px;height:90px;right:-26px;bottom:-40px;border-radius:50%;background:var(--accent,#6c8cff);filter:blur(35px);opacity:.18}.metric-card small{display:block;color:var(--muted);font-size:10px;font-weight:800;letter-spacing:.1em;text-transform:uppercase}.metric-card b{display:block;margin:18px 0 7px;font-size:27px;line-height:1;letter-spacing:-.035em}.metric-card span{color:#8fa0b5;font-size:11px}.metric-card.good{--accent:var(--green)}.metric-card.warn{--accent:var(--amber)}.metric-card.bad{--accent:var(--red)}.metric-card.info{--accent:var(--cyan)}.decision-grid,.split-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:14px 0}.decision-grid{grid-template-columns:1.05fr 1fr 1fr}.info-card,.flow-card,.hardware-callout{padding:22px;border:1px solid var(--line);border-radius:22px;background:linear-gradient(150deg,#111a28d9,#0d131ed9);box-shadow:inset 0 1px #ffffff08}.info-card h3,.flow-card h3,.hardware-callout h3{margin:7px 0 14px;font-size:17px}.info-card ul,.info-card ol,.hardware-callout ul{color:#a9b6c8}.info-card.ready{border-color:#4bd7a22e}.info-card.waiting{border-color:#f7c76533}.info-card.blocker{border-color:#ff718832}.card-note{margin:16px 0 0;padding-top:14px;border-top:1px solid var(--line);color:#8493a7;font-size:11px}.gate-strip{display:flex;align-items:center;justify-content:space-between;gap:14px;margin:14px 0 20px;padding:16px 18px;border:1px solid #ff71882e;border-radius:17px;background:linear-gradient(90deg,#ff71880c,#101722)}.gate-strip b{color:#ffc4ce}.gate-strip code{color:#92a3b8;font-size:11px}.table-shell{overflow:auto;border:1px solid var(--line);border-radius:22px;background:#0b111be8;box-shadow:0 22px 55px #0005}.detail-table{min-width:1800px}.roadmap-table{min-width:1180px}.detail-table,.roadmap-table{border-collapse:separate;border-spacing:0;font-size:12px}.detail-table thead,.roadmap-table thead{position:sticky;top:0;background:#151f30}.detail-table th,.roadmap-table th{padding:14px;color:#9dacbf}.detail-table td,.roadmap-table td{padding:16px 14px;background:#0e1623d9}.detail-table th:nth-child(1){width:185px}.detail-table th:nth-child(2){width:105px}.detail-table th:nth-child(n+3):nth-child(-n+7){width:255px}.detail-table th:nth-child(8){width:225px}.roadmap-table th:nth-child(1){width:80px}.roadmap-table th:nth-child(2){width:190px}.roadmap-table th:nth-child(3),.roadmap-table th:nth-child(4){width:455px}.roadmap-table td:first-child{text-align:center}.priority-number{display:inline-grid;place-items:center;width:30px;height:30px;border:1px solid #55d9e64a;border-radius:10px;background:#55d9e60c;color:#91f0f7;font-weight:900}.stage{left:0}.state{left:185px}.speed-maturity-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:14px 0}.maturity-card{padding:18px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#111a28,#0d141f)}.maturity-card small{display:block;color:#8291a6;font-size:9px;font-weight:950;letter-spacing:.12em}.maturity-card b{display:block;margin:9px 0 7px;font-size:28px;line-height:1}.maturity-card span{color:#8f9eb2;font-size:11px;line-height:1.5}.maturity-card.verified{border-color:#4bd7a233}.maturity-card.verified b{color:var(--green)}.maturity-card.partial{border-color:#f7c76538}.maturity-card.partial b{color:var(--amber)}.maturity-card.blocked{border-color:#ff71883d}.maturity-card.blocked b{color:var(--red)}.speed-evidence-list{display:grid;gap:14px;min-width:0}.speed-evidence-card{min-width:0;overflow:hidden;border:1px solid var(--line);border-radius:22px;background:linear-gradient(150deg,#101925,#0c131e);box-shadow:inset 4px 0 var(--violet),inset 0 1px #ffffff08}.speed-evidence-card.pass{box-shadow:inset 4px 0 var(--green),inset 0 1px #ffffff08}.speed-evidence-card.partial{box-shadow:inset 4px 0 var(--amber),inset 0 1px #ffffff08}.speed-evidence-card.blocked{box-shadow:inset 4px 0 var(--red),inset 0 1px #ffffff08}.speed-card-header{display:grid;grid-template-columns:auto minmax(0,1fr) auto;align-items:center;gap:14px;padding:18px 20px;border-bottom:1px solid var(--line);background:#ffffff03}.speed-card-header small{display:block;color:#73849a;font-size:8px;font-weight:950;letter-spacing:.14em}.speed-card-header h3{margin:3px 0 0;font-size:17px;letter-spacing:-.015em}.speed-id{display:grid;place-items:center;width:40px;height:40px;border:1px solid #55d9e635;border-radius:12px;background:#55d9e609;color:#8cebf2;font:900 12px ui-monospace,SFMono-Regular,Menlo,monospace}.speed-card-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));min-width:0}.speed-detail-section{min-width:0;padding:17px 19px;border-right:1px solid #ffffff0b;border-bottom:1px solid #ffffff0b}.speed-detail-section:nth-child(3n){border-right:0}.speed-detail-section:nth-last-child(-n+3){border-bottom:0}.speed-detail-section h4{margin:0 0 10px;color:#aab8c9;font-size:9px;font-weight:950;letter-spacing:.1em;text-transform:uppercase}.speed-detail-section ul{font-size:11px;line-height:1.55;overflow-wrap:anywhere}.speed-detail-section.result{background:#55d9e604}.speed-detail-section.missing{background:#ff718804}.speed-detail-section.evidence li{color:#74869e;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10px}.flow-card{margin-top:14px}.flow{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:7px;margin:18px 0}.flow-step{position:relative;min-height:92px;padding:14px 10px;border:1px solid var(--line);border-radius:15px;background:#ffffff05}.flow-step:not(:last-child):after{content:"›";position:absolute;right:-8px;top:34px;z-index:2;color:#78dce5;font-size:18px}.flow-step span{display:grid;place-items:center;width:22px;height:22px;margin-bottom:10px;border-radius:7px;background:#55d9e615;color:#7de8f0;font-size:10px;font-weight:900}.flow-step b{font-size:11px;line-height:1.4}.flow-card>p{margin:14px 0 0;color:#a8b6c8}.hardware-callout{margin-bottom:14px;border-color:#ff71883c;background:linear-gradient(145deg,#23131b,#101722 66%)}.hardware-callout h3{color:#ffd7de}.hardware-phases{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px;margin:16px 0 20px}.hardware-phase{padding:16px 13px;border:1px solid var(--line);border-radius:17px;background:#0e1622}.hardware-phase span{display:block;color:#77889f;font-size:9px;font-weight:900;letter-spacing:.1em}.hardware-phase b{display:block;margin:8px 0;color:#e8eff7;font-size:12px}.hardware-phase small{color:#8392a6;font-size:10px;line-height:1.45}.updates-list{display:grid;gap:10px}.update-card{border:1px solid var(--line);border-radius:18px;background:#0d141fca;overflow:hidden}.update-card[open]{border-color:#55d9e62e;background:#0e1724}.update-card summary{display:grid;grid-template-columns:10px 1fr 26px;align-items:center;gap:14px;padding:17px 19px;cursor:pointer;list-style:none}.update-card summary::-webkit-details-marker{display:none}.update-card time{display:block;margin-bottom:3px;color:#7f8fa5;font:700 10px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace}.update-card summary b{font-size:13px}.update-dot{width:8px;height:8px;border-radius:50%;background:var(--violet)}.update-dot.pass{background:var(--green)}.update-dot.partial{background:var(--amber)}.update-dot.blocked{background:var(--red)}.expand-icon{color:#718198;font-size:18px;transition:.2s}.update-card[open] .expand-icon{transform:rotate(45deg);color:var(--cyan)}.update-content{padding:0 19px 20px 43px}.update-summary{margin:0 0 16px;padding:14px 16px;border-left:2px solid #55d9e65c;background:#55d9e607;color:#aab8ca;font-size:12px}.update-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.update-block{padding:13px;border:1px solid #ffffff0c;border-radius:13px;background:#ffffff03}.update-block h4{margin:0 0 9px;color:#a8b7ca;font-size:10px;text-transform:uppercase;letter-spacing:.08em}.update-block ul{font-size:11px}.term-tooltip{backdrop-filter:blur(16px)}footer{margin-top:24px;border-top:1px solid var(--line);text-align:left;display:flex;justify-content:space-between;gap:15px}footer span:last-child{text-align:right}
    @media(max-width:1100px){.workspace{grid-template-columns:190px minmax(0,1fr);gap:16px}.metric-grid{grid-template-columns:repeat(3,1fr)}.decision-grid{grid-template-columns:1fr}.flow{grid-template-columns:repeat(4,1fr)}.hardware-phases{grid-template-columns:repeat(3,1fr)}.update-grid{grid-template-columns:repeat(2,1fr)}.speed-card-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.speed-detail-section:nth-child(3n){border-right:1px solid #ffffff0b}.speed-detail-section:nth-child(2n){border-right:0}.speed-detail-section:nth-last-child(-n+3){border-bottom:1px solid #ffffff0b}.speed-detail-section:nth-last-child(-n+2){border-bottom:0}}
    @media(max-width:760px){.shell{width:calc(100% - 18px);padding-top:12px}.topbar{align-items:flex-start;gap:10px}.top-meta .mini-chip:not(:last-child){display:none}.hero{min-height:0;padding:26px 21px;border-radius:22px;align-items:flex-start}.hero h1{font-size:42px}.hero-side{align-items:flex-start;min-width:0}.decision-badge,.author-card{text-align:left}.workspace{display:block}.tabs{top:6px;flex-direction:row;margin-bottom:18px;border-radius:15px;overflow-x:auto;scrollbar-width:none}.tabs::-webkit-scrollbar{display:none}.tab-label{display:none}.tab{flex:0 0 auto;width:auto;padding:10px 12px}.tab:before{display:none}.metric-grid{grid-template-columns:repeat(2,1fr)}.metric-card{min-height:128px;padding:16px}.metric-card b{font-size:22px}.split-grid,.update-grid{grid-template-columns:1fr}.flow{grid-template-columns:repeat(2,1fr)}.hardware-phases{grid-template-columns:repeat(2,1fr)}.speed-maturity-grid{grid-template-columns:1fr}.speed-card-header{grid-template-columns:auto minmax(0,1fr)}.speed-card-header .pill{grid-column:2}.speed-card-grid{grid-template-columns:1fr}.speed-detail-section,.speed-detail-section:nth-child(2n),.speed-detail-section:nth-child(3n){border-right:0}.speed-detail-section:nth-last-child(-n+2),.speed-detail-section:nth-last-child(-n+3){border-bottom:1px solid #ffffff0b}.speed-detail-section:last-child{border-bottom:0}.section-heading{align-items:flex-start;flex-direction:column}.update-content{padding-left:19px}.roadmap-table{min-width:980px}footer{flex-direction:column}footer span:last-child{text-align:left}}
    """

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="author" content="Shihua Yu"><meta name="description" content="Evidence-bound Unitree G1 VLA simulation and hardware readiness dashboard"><title>G1 VLA · Evidence & Readiness</title><style>{css}</style></head>
<body><main class="shell">
<div class="topbar"><div class="brand"><span class="brand-mark">G1</span><span>VLA Readiness</span></div><div class="top-meta"><a class="mini-chip brief-link" href="https://g1-vla-simulation-readiness.vercel.app/research-brief/">教授版研究概要 ↗</a><span class="mini-chip">Generated {html.escape(generated)}</span><span class="mini-chip">Tests {tests['passed']}/{tests['total']}</span><span class="mini-chip">Shihua Yu</span></div></div>
<header class="hero"><div class="hero-copy"><div class="eyebrow">LGG100 · UNITREE G1 EDU · EVIDENCE-BOUND</div><h1><span>Simulation to Hardware</span><br>Readiness Dashboard</h1><p>页面依据已保存的模型、仿真、延迟与安全证据生成。已验证结果、未完成项和硬件执行边界分别陈述；缺少证据的项目不推定为通过。</p></div><div class="hero-side"><div class="decision-badge"><small>EXECUTION DECISION</small><b>当前不允许真机动作</b></div><div class="author-card"><small>Project owner</small><b>Shihua Yu</b></div></div></header>
<div class="workspace">
<nav class="tabs" role="tablist" aria-label="项目视图" aria-orientation="vertical">
  <div class="tab-label" role="presentation">PROJECT VIEWS</div>
  <button class="tab" id="tab-current" role="tab" aria-selected="true" aria-controls="panel-current" tabindex="0" data-tab="current">当前状态</button>
  <button class="tab" id="tab-next" role="tab" aria-selected="false" aria-controls="panel-next" tabindex="-1" data-tab="next">下一步计划<span class="tab-count">{len(PRE_REAL_SIMULATION_PRIORITIES)}</span></button>
  <button class="tab" id="tab-speed" role="tab" aria-selected="false" aria-controls="panel-speed" tabindex="-1" data-tab="speed">速度模块<span class="tab-count">{len(speed_rows)}</span></button>
  <button class="tab" id="tab-simulation" role="tab" aria-selected="false" aria-controls="panel-simulation" tabindex="-1" data-tab="simulation">Simulation<span class="tab-count">{len(simulation_rows) + len(model_rows)}</span></button>
  <button class="tab" id="tab-hardware" role="tab" aria-selected="false" aria-controls="panel-hardware" tabindex="-1" data-tab="hardware">真机阶段<span class="tab-count">{len(hardware_rows)}</span></button>
  <button class="tab" id="tab-updates" role="tab" aria-selected="false" aria-controls="panel-updates" tabindex="-1" data-tab="updates">开发更新<span class="tab-count">{len(updates)}</span></button>
</nav>
<div class="tab-content">
<section class="tab-panel active" id="panel-current" role="tabpanel" aria-labelledby="tab-current" data-panel="current">
  <div class="section-heading"><div><span class="kicker">CURRENT EVIDENCE</span><h2>当前状态</h2><p>指标均来自版本化结果文件；状态不依据推测或目标值填写。</p></div><span class="owner-chip">Evidence owner · Shihua Yu</span></div>
  <div class="metric-grid">
    <article class="metric-card good"><small>Neural output</small><b>{inference['finite_shape_passes']}/{inference['samples']}</b><span>有限 [32,16] action chunks</span></article>
    <article class="metric-card bad"><small>Closed-loop</small><b>{closed['completed_cycles']} cycles</b><span>task_success=false</span></article>
    <article class="metric-card bad"><small>Commit age</small><b>{commit_age:.2f} ms</b><span>Gate ≤ {maximum_age:.0f} ms</span></article>
    <article class="metric-card good"><small>Local regression</small><b>{tests['passed']}/{tests['total']}</b><span>automated tests passed</span></article>
    <article class="metric-card warn"><small>Simulation stages</small><b>{simulation_passed}/{len(simulation_rows)}</b><span>通过；其余仍需证据</span></article>
  </div>
  <div class="gate-strip"><b>总体验证结论：未达到真机动作条件</b><code>g1_execution_enabled=false · hardware_execution_performed=false</code></div>
  <div class="decision-grid">
    <article class="info-card ready"><span class="card-label">ESTABLISHED</span><h3>已建立的证据</h3>{bullet_cell(["G1 pelvis-frame EEF-16 数据契约已冻结。", "MuJoCo observation schema 与 contract 一致。", "真实 LGG100 输出 50/50 shape/finite 通过，最佳语义为 absolute_xyzw_lr。", "安全链在 IK/preflight 失败时执行 fail-closed hold。"])} </article>
    <article class="info-card blocker"><span class="card-label">ACTIVE BLOCKERS</span><h3>当前阻塞项</h3>{bullet_cell(["S4 尚缺完整 32-step、多 chunk 与随机目标 IK/swept-path 回归。", "Adaptive-OFF completed_cycles=0，稳定抓取尚未证明。", f"完整 commit age 为 {commit_age:.2f} ms，超过 {maximum_age:.0f} ms Gate。", "真实相机、EEF、Dex1 与物理场景尚未标定。"])} </article>
    <article class="info-card waiting"><span class="card-label">EXECUTION BOUNDARY</span><h3>允许范围</h3>{bullet_cell(["允许：本地 Simulation、离线 replay、subscriber-only 代码审计。", "条件允许后：真实 LowState 只读订阅与 zero-motion Shadow。", "禁止：ChannelPublisher、LowCmd、模式切换和任何真机运动。", f"硬件阻塞或未开始阶段：{hardware_blocked}/{len(hardware_rows)}。"])} </article>
  </div>
</section>
<section class="tab-panel" id="panel-next" role="tabpanel" aria-labelledby="tab-next" data-panel="next" hidden>{next_steps_panel()}</section>
<section class="tab-panel" id="panel-speed" role="tabpanel" aria-labelledby="tab-speed" data-panel="speed" hidden>
  <div class="section-heading"><div><span class="kicker">ADAPTIVE SPEED EVIDENCE</span><h2>速度模块验证明细</h2><p>表格区分已验证的局部机制、部分通过的 LGG100 evidence 和尚未建立的任务级收益。</p></div><span class="owner-chip">Adaptive-ON · BLOCKED</span></div>
  <div class="metric-grid">
    <article class="metric-card good"><small>Path invariant</small><b>Byte-identical</b><span>仅修改 timestamps</span></article>
    <article class="metric-card good"><small>Near behavior</small><b>0.50×</b><span>32/32 approach samples</span></article>
    <article class="metric-card good"><small>Far duration</small><b>−{abs(scenarios['far']['duration_change_percent']):.2f}%</b><span>single-chunk accepted</span></article>
    <article class="metric-card warn"><small>Mixed transition</small><b>未通过</b><span>缺少 grasp phase</span></article>
    <article class="metric-card bad"><small>Task-level speedup</small><b>未证明</b><span>Adaptive-OFF 0 cycles</span></article>
  </div>
  <div class="gate-strip"><b>速度模块当前结论：局部决策逻辑有效；完整任务提速和 Adaptive-ON 资格未建立</b><code>controlled_phase_speed_behavior_passed=false</code></div>
  <div class="speed-maturity-grid" aria-label="速度模块成熟度概览">
    <article class="maturity-card verified"><small>VERIFIED MECHANISMS</small><b>{sum(row['status'] == 'pass' for row in speed_rows)}</b><span>模块不变量、context、hold、phase policy、envelope 与 context 性能</span></article>
    <article class="maturity-card partial"><small>PARTIAL EVIDENCE</small><b>{sum(row['status'] == 'partial' for row in speed_rows)}</b><span>Deterministic coverage 与真实 LGG100 single-chunk sweep</span></article>
    <article class="maturity-card blocked"><small>BLOCKED / TODO</small><b>{sum(row['status'] in {'blocked', 'todo'} for row in speed_rows)}</b><span>Adaptive-OFF 基线与任务级 Adaptive-ON 资格</span></article>
  </div>
  <div class="speed-evidence-list" aria-label="速度模块详细验证记录">{speed_evidence_cards(speed_rows)}</div>
</section>
<section class="tab-panel" id="panel-simulation" role="tabpanel" aria-labelledby="tab-simulation" data-panel="simulation" hidden>
  <div class="section-heading"><div><span class="kicker">SIMULATION & MODEL</span><h2>Simulation 详细进展</h2><p>S0–S7 与 GEN-1.5 研究项按完成、结果、缺口、后续动作和证据展开。</p></div><div class="legend"><span class="pass">通过</span><span class="partial">部分完成</span><span class="blocked">阻塞</span><span class="todo">未开始</span></div></div>
  {stage_table(simulation_rows + model_rows, "Simulation 与模型研究详细进展")}
</section>
<section class="tab-panel" id="panel-hardware" role="tabpanel" aria-labelledby="tab-hardware" data-panel="hardware" hidden>
  <div class="section-heading"><div><span class="kicker">HARDWARE STAGING</span><h2>之后的真机阶段</h2><p>真机阶段按只读、零运动、受保护确定性动作和任务闭环递进；任何前置 Gate 失败均停止升级。</p></div><span class="owner-chip">Motion authority · OFF</span></div>
  <article class="hardware-callout"><span class="card-label">NOT REPLACEABLE BY SIMULATION</span><h3>Simulation 无法替代的部分</h3>{bullet_cell(PRE_REAL_NOT_REPLACEABLE)}<p class="card-note">以上项目只能通过真实同步测量、厂商限制或受控硬件验证完成，MuJoCo 数据不得作为物理标定证据。</p></article>
  <div class="hardware-phases">
    <article class="hardware-phase"><span>PHASE A</span><b>只读 Inventory</b><small>Python、CycloneDDS、SDK、网卡、服务与机器人 variant；不初始化 Publisher。</small></article>
    <article class="hardware-phase"><span>PHASE B</span><b>LowState Subscriber</b><small>固定样本读取、freshness、motor index、IMU 顺序与单位；保持 Damping。</small></article>
    <article class="hardware-phase"><span>PHASE C</span><b>物理标定</b><small>三相机、pelvis/EEF、桌面、方块和 Dex1 同步测量与 hash-bound report。</small></article>
    <article class="hardware-phase"><span>PHASE D</span><b>Zero-motion Shadow</b><small>LowState→FK→current pose IK→safety，仅记录建议动作，不 publish。</small></article>
    <article class="hardware-phase"><span>PHASE E</span><b>受保护确定性动作</b><small>仅在前序 Gate、watchdog、E-stop、limits 和 command adapter 全部通过后评审。</small></article>
    <article class="hardware-phase"><span>PHASE F</span><b>任务闭环</b><small>单臂→双臂→桌面→轻物体→Adaptive-OFF；Adaptive-ON 最后评估。</small></article>
  </div>
  {stage_table(hardware_rows, "真机阶段详细进展")}
</section>
<section class="tab-panel" id="panel-updates" role="tabpanel" aria-labelledby="tab-updates" data-panel="updates" hidden>
  <div class="section-heading"><div><span class="kicker">STRUCTURED CHANGELOG</span><h2>开发更新</h2><p>{len(updates)} 条结构化记录；每条记录分别列出完成项、结论、修复、缺口、后续动作与证据。</p></div><span class="owner-chip">Maintainer · Shihua Yu</span></div>
  <div class="updates-list">{updates_panel(updates)}</div>
</section>
</div>
</div>
<div id="term-tooltip" class="term-tooltip" role="tooltip" aria-hidden="true"></div>
<footer><span>G1 VLA evidence dashboard · Shihua Yu</span><span>g1_execution_enabled=false · no hardware action performed</span></footer>
<script>
const tabs=[...document.querySelectorAll('[role="tab"]')];
const panels=[...document.querySelectorAll('[role="tabpanel"]')];
const tablist=document.querySelector('[role="tablist"]');
const narrowTabs=window.matchMedia('(max-width:760px)');
function syncTabOrientation(){{tablist.setAttribute('aria-orientation',narrowTabs.matches?'horizontal':'vertical');}}
syncTabOrientation();narrowTabs.addEventListener('change',syncTabOrientation);
function activateTab(name,moveFocus=false){{
  tabs.forEach(tab=>{{
    const selected=tab.dataset.tab===name;
    tab.setAttribute('aria-selected',String(selected));
    tab.tabIndex=selected?0:-1;
    if(selected&&moveFocus) tab.focus();
  }});
  panels.forEach(panel=>{{
    const selected=panel.dataset.panel===name;
    panel.hidden=!selected;
    panel.classList.toggle('active',selected);
  }});
  if(history.replaceState) history.replaceState(null,'','#'+name);
}}
tabs.forEach((tab,index)=>{{
  tab.addEventListener('click',()=>activateTab(tab.dataset.tab));
  tab.addEventListener('keydown',event=>{{
    let target=null;
    if(event.key==='ArrowRight'||event.key==='ArrowDown') target=(index+1)%tabs.length;
    if(event.key==='ArrowLeft'||event.key==='ArrowUp') target=(index-1+tabs.length)%tabs.length;
    if(event.key==='Home') target=0;
    if(event.key==='End') target=tabs.length-1;
    if(target!==null){{event.preventDefault();activateTab(tabs[target].dataset.tab,true);}}
  }});
}});
const initial=location.hash.slice(1);
if(tabs.some(tab=>tab.dataset.tab===initial)) activateTab(initial);
const tooltip=document.getElementById('term-tooltip');
let activeTerm=null;
function placeTooltip(x,y){{
  const pad=14,width=tooltip.offsetWidth,height=tooltip.offsetHeight;
  let left=Math.max(pad,Math.min(x+15,window.innerWidth-width-pad));
  let top=y+18;
  if(top+height>window.innerHeight-pad) top=Math.max(pad,y-height-15);
  tooltip.style.left=left+'px';tooltip.style.top=top+'px';
}}
function showTooltip(term,x,y){{activeTerm=term;tooltip.textContent=term.dataset.tip;tooltip.classList.add('visible');tooltip.setAttribute('aria-hidden','false');placeTooltip(x,y);}}
function hideTooltip(){{activeTerm=null;tooltip.classList.remove('visible');tooltip.setAttribute('aria-hidden','true');}}
document.querySelectorAll('a.term').forEach(term=>{{
  term.addEventListener('mouseenter',event=>showTooltip(term,event.clientX,event.clientY));
  term.addEventListener('mousemove',event=>{{if(activeTerm===term) placeTooltip(event.clientX,event.clientY);}});
  term.addEventListener('mouseleave',hideTooltip);
  term.addEventListener('focus',()=>{{const r=term.getBoundingClientRect();showTooltip(term,r.left+r.width/2,r.bottom);}});
  term.addEventListener('blur',hideTooltip);
}});
window.addEventListener('scroll',()=>{{if(activeTerm){{const r=activeTerm.getBoundingClientRect();placeTooltip(r.left+r.width/2,r.bottom);}}}},true);
</script>
</main></body></html>"""


def build_public() -> str:
    """Build a public copy with private lab topology and identities redacted."""
    rendered = build()
    rendered = rendered.replace(
        "Mac→开发机 192.168.1.13→机器人 192.168.123.164 登录成功。",
        "Operator host→lab gateway→robot internal compute 登录成功（地址已脱敏）。",
    )
    rendered = rendered.replace(
        "Prompt: unitree@unitree-g1-nx。",
        "已获得认证的机器人内部只读 shell（身份已脱敏）。",
    )
    rendered = re.sub(
        r"\b(?:user1|yixiao|unitree|test)@[A-Za-z0-9_.-]+",
        "[ssh-identity-redacted]",
        rendered,
    )
    rendered = re.sub(
        r"\b(?:shihua-vla(?:-jump|-connect|-keepalive)?|unitree-g1-nx|nnmc65)\b",
        "[private-host-redacted]",
        rendered,
    )
    rendered = re.sub(
        r"(?<![A-Za-z0-9])/(?:Users|home)/[^\s<\"']+",
        "[private-path-redacted]",
        rendered,
    )
    rendered = re.sub(
        r"(?<![0-9])(?:10(?:\.[0-9]{1,3}){3}|192\.168(?:\.[0-9]{1,3}){2}|172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2})(?![0-9])",
        "[private-address-redacted]",
        rendered,
    )

    def public_href(match: re.Match[str]) -> str:
        href = match.group(1)
        if href.startswith(("https://", "http://", "#")):
            return f'href="{href}"'
        return f'href="{PUBLIC_GITHUB_BASE}{quote(href, safe="/")}"'

    rendered = re.sub(r'href="([^"]+)"', public_href, rendered)
    rendered = rendered.replace(
        "EVIDENCE-BOUND",
        "PUBLIC SANITIZED · EVIDENCE-BOUND",
        1,
    )
    rendered = rendered.replace(
        "G1 VLA evidence dashboard ·",
        "G1 VLA public sanitized dashboard · private LAN identities redacted ·",
    )
    return rendered


def main() -> None:
    OUTPUT.write_text(build())
    PUBLIC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_OUTPUT.write_text(build_public())
    print(OUTPUT)
    print(PUBLIC_OUTPUT)


if __name__ == "__main__":
    main()
