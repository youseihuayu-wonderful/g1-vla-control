#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the evidence-bound Simulation → G1 deployment summary."""

from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "simulation_deployment_summary.html"


def load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text())


def number(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def pill(status: str, text: str) -> str:
    return f'<span class="pill {html.escape(status)}"><i></i>{html.escape(text)}</span>'


def list_html(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


def stage_card(stage: dict[str, Any]) -> str:
    return f"""
    <article class="stage-card {stage['status']}">
      <div class="stage-top"><span class="stage-id">{html.escape(stage['id'])}</span>{pill(stage['status'], stage['label'])}</div>
      <h3>{html.escape(stage['title'])}</h3>
      <div class="stage-columns">
        <div><h4>做了什么</h4>{list_html(stage['done'])}</div>
        <div><h4>这意味着什么</h4>{list_html(stage['meaning'])}</div>
        <div><h4>还差什么</h4>{list_html(stage['missing'])}</div>
      </div>
      <p class="evidence"><b>证据：</b>{html.escape(stage['evidence'])}</p>
    </article>
    """


def flow_node(node: dict[str, str]) -> str:
    return f"""
    <div class="flow-node {node['status']}">
      <span>{html.escape(node['id'])}</span>
      <b>{html.escape(node['title'])}</b>
      <small>{html.escape(node['detail'])}</small>
    </div>
    """


def build() -> str:
    author = load("lgg100_author32_revalidation_summary.json")
    phase = load("lgg100_author32_t0_phase_speed_sweep_validation.json")
    closed = load("lgg100_author32_closed_loop_t0_offset008_baseline_realtime.json")
    ik = load("lgg100_author32_online_ik_parameter_sweep.json")
    tests = load("test_summary.json")
    sdk = load("unitree_g1_sdk2_readonly_audit_20260818.json")
    gpu = load("l40s_gpu_availability_20260819.json")

    inference = author["inference"]
    semantic = author["semantic_validation"]
    record = closed["records"][0]
    scenarios = {item["role"]: item for item in phase["scenarios"]}
    commit_age = record["observation_age_at_commit_ms"]
    max_age = record["maximum_observation_age_ms"]
    passing_ik = len(ik["passing_both_trials"])
    generated = datetime.now().astimezone().isoformat(timespec="seconds")

    sim_flow = [
        {"id": "S0", "title": "G1 Contract", "detail": "16-D · pelvis · xyzw", "status": "pass"},
        {"id": "S1", "title": "Observation", "detail": "3 RGB + state[16]", "status": "pass"},
        {"id": "S2", "title": "LGG100 Output", "detail": f"{inference['finite_shape_passes']}/{inference['samples']} finite [32,16]", "status": "pass"},
        {"id": "S3", "title": "Adaptive Timing", "detail": "near/far pass · mixed blocked", "status": "partial"},
        {"id": "S4", "title": "IK / Swept Path", "detail": "cycle 0 rejection", "status": "blocked"},
        {"id": "S5", "title": "Closed Loop", "detail": f"{closed['completed_cycles']} cycles · success=false", "status": "blocked"},
        {"id": "S6", "title": "Realtime", "detail": f"{commit_age:.2f} / {max_age:.0f} ms", "status": "blocked"},
        {"id": "S7", "title": "Randomized", "detail": "fault/jitter not complete", "status": "todo"},
    ]

    sim_stages = [
        {
            "id": "S0–S1", "title": "冻结 G1 数据契约与 MuJoCo Observation", "status": "pass", "label": "已验证",
            "done": [
                "冻结三路 480×640 RGB，经 resize_with_pad 变为 224×224。",
                "冻结 pelvis-frame state/action[16]、xyzw 四元数、双 Dex1 和双臂 14 关节顺序。",
                "MuJoCo observation、policy client 和 action validator 共用同一 contract。",
            ],
            "meaning": [
                "模型、仿真与未来硬件 adapter 有明确的数据边界。",
                "错误 shape、非有限值、错误四元数和未知 contract 会 fail closed。",
            ],
            "missing": [
                "真实三相机、LowState、Dex1 还没有证明能产生相同 observation。",
                "真实 EEF site、相机外参和时间同步尚未标定。",
            ],
            "evidence": "g1_policy_contract.yaml · g1_policy_contract.py · MuJoCo bridge tests",
        },
        {
            "id": "S2", "title": "严格恢复真实 LGG100 并识别输出语义", "status": "pass", "label": "离线通过",
            "done": [
                f"固定 pi05_g1_eef、horizon 32 和 checkpoint revision {author['checkpoint']['revision'][:10]}…。",
                f"采集 {inference['samples']} 次输出；{inference['finite_shape_passes']}/{inference['samples']} 为有限 [32,16]。",
                f"在 {semantic['usable_samples']} 个可用样本上比较语义假设，最佳为 {semantic['best_hypothesis']}。",
            ],
            "meaning": [
                "真实模型能够稳定产生符合基础 shape 的双手 EEF action chunk。",
                "支持 absolute、xyzw、left→right 的解释，但这仍是离线证据。",
            ],
            "missing": [
                "离线输出不能证明任务抓取成功、闭环稳定或硬件安全。",
                "当前 L40S 没有可安全启动 LGG100 的完全空闲 GPU。",
            ],
            "evidence": "results/lgg100_author32_revalidation_summary.json",
        },
        {
            "id": "S3", "title": "Adaptive 远处加速、近处减速", "status": "partial", "label": "部分通过",
            "done": [
                f"Near 场景使用 {scenarios['near']['scale_range'][0]:.2f}×，避免高速接触。",
                f"Far 场景 chunk duration 改善 {abs(scenarios['far']['duration_change_percent']):.1f}%。",
                f"Mixed 场景产生最多 {scenarios['mixed']['scale_range'][1]:.2f}× 的局部加速。",
            ],
            "meaning": [
                "调速器已展示 clearance/phase-aware 的局部速度行为。",
                "速度改变的是 timing，不改变 LGG100 action path。",
            ],
            "missing": [
                "Mixed grasp transition coverage 未通过，因此该场景被正确拒绝。",
                "尚未证明完整抓取总时间下降且成功率不降低。",
            ],
            "evidence": "results/lgg100_author32_t0_phase_speed_sweep_validation.json",
        },
        {
            "id": "S4", "title": "IK、连续路径与碰撞预检", "status": "blocked", "label": "当前阻塞",
            "done": [
                "实现双臂 IK、5 mm/3° 误差 Gate、关节限制和 sequential swept-path collision。",
                "诊断证明 30 次迭代失败主要是收敛预算，而不是目标物理不可达。",
                f"参数扫描中有 {passing_ik} 组候选在两个 committed target 上通过。",
            ],
            "meaning": [
                "系统不会把未经验证的 EEF action 直接变成关节命令。",
                "现有 hold 是安全拒绝，不是任务执行成功。",
            ],
            "missing": [
                "候选参数必须通过完整 32-step、多 chunk、随机可达目标与已知碰撞回归。",
                "不能放宽 5 mm/3° 阈值，也不能只依赖前两个 target。",
            ],
            "evidence": "results/lgg100_author32_online_ik_convergence_trace.json · parameter sweep",
        },
        {
            "id": "S5", "title": "Adaptive-OFF 真实神经闭环", "status": "blocked", "label": "未完成",
            "done": [
                "建立 quarantined closed-loop：observation→LGG100→preflight→commit/hold。",
                "修复没有新鲜动作执行也把 watchdog 判为通过的问题。",
                "记录 inference、observation age、preflight、commit 和 rejection evidence。",
            ],
            "meaning": [
                f"当前在 cycle 0 因 {closed['abort_reason']} 停止。",
                "动作未执行，说明 fail-closed 链有效，但任务能力没有得到证明。",
            ],
            "missing": [
                "Adaptive-OFF 必须先稳定完成抓取/堆叠。",
                "之后才能在同 action、同初态下比较 Adaptive-ON。",
            ],
            "evidence": "results/lgg100_author32_closed_loop_t0_offset008_baseline_realtime.json",
        },
        {
            "id": "S6–S7", "title": "实时性、随机化与故障注入", "status": "blocked", "label": "未达标",
            "done": [
                f"测得 warm inference P50 {inference['warm_latency_ms']['p50']:.2f} ms。",
                f"测得完整 observation-to-commit {commit_age:.2f} ms。",
                "将重复场景测量改为每周期一次，本地 context 构建大幅下降。",
            ],
            "meaning": [
                f"完整链仍超过 {max_age:.0f} ms Gate，不能用推理时间代替端到端时间。",
                "无新鲜 commit 时 watchdog 继续保持未验证。",
            ],
            "missing": [
                "远端重新实测优化后的 render/inference/context/preflight/commit。",
                "完成网络 jitter、stale、断线、相机变化、摩擦和物体随机化。",
            ],
            "evidence": "closed-loop timing records · watchdog regression tests",
        },
    ]

    real_flow = [
        {"id": "H0", "title": "Robot Shell", "detail": "authenticated · Damping", "status": "partial"},
        {"id": "H1", "title": "LowState", "detail": "adapter local only", "status": "todo"},
        {"id": "H2", "title": "3 Cameras", "detail": "calibration missing", "status": "blocked"},
        {"id": "H3", "title": "Real Observation", "detail": "parity unverified", "status": "blocked"},
        {"id": "H4", "title": "L40S Policy", "detail": "no safe GPU/server", "status": "blocked"},
        {"id": "H5", "title": "IK + Safety", "detail": "hardware profile missing", "status": "blocked"},
        {"id": "H6", "title": "Command Adapter", "detail": "not implemented", "status": "blocked"},
        {"id": "H7", "title": "Feedback Loop", "detail": "execution disabled", "status": "blocked"},
    ]

    hardware_stages = [
        {
            "id": "H0–H1", "title": "认证连接与 Unitree LowState 只读入口", "status": "partial", "label": "连接完成",
            "done": [
                "通过开发机进入 unitree@unitree-g1-nx，机器人由操作员保持 Damping。",
                f"固定官方 unitree_sdk2_python commit {sdk['official_sources']['python']['commit'][:10]}…。",
                "实现 subscriber-only LowState adapter，并以 AST 测试禁止 command API。",
            ],
            "meaning": [
                "已经有安全的源码基础，可在后续独立批准后读取真实反馈。",
                "SSH shell 在线不等于 DDS、LowState 或动作链已验证。",
            ],
            "missing": [
                "只读 inventory：Python、CycloneDDS、SDK、robot-facing interface。",
                "真实订阅后验证 29-DOF、索引、单位、tick、IMU 顺序和 freshness。",
            ],
            "evidence": "UNITREE_G1_SDK_READONLY.md · g1_unitree_lowstate.py",
        },
        {
            "id": "H2–H3", "title": "真实相机、EEF、Dex1 与物理标定", "status": "blocked", "label": "缺少真实数据",
            "done": [
                "仿真端已冻结三相机 key、图像格式、pelvis EEF 和 Dex1 state。",
                "文档确认左腕相机曾损坏并粘回，已标记为高风险输入。",
            ],
            "meaning": [
                "真机 observation 必须与训练/仿真 contract 一致，不能复制或替换相机。",
                "有画面不代表外参、时间同步或 EEF site 正确。",
            ],
            "missing": [
                "三相机内外参、时间同步、左腕相机完整性。",
                "pelvis/桌面/方块坐标、EEF offset、Dex1 零点与方向的真实测量。",
            ],
            "evidence": "g1_policy_contract.yaml hardware blockers · operator document",
        },
        {
            "id": "H4", "title": "L40S output-only policy 服务", "status": "blocked", "label": "GPU 阻塞",
            "done": [
                "建立 Mac 127.0.0.1:8000→L40S 127.0.0.1:8000 的 loopback tunnel。",
                "建立 ControlMaster、keeper、连接 runbook 和 GPU 只读监控。",
            ],
            "meaning": [
                "网络路径已准备，但 tunnel listener 不等于 policy server 可用。",
                "不会终止共享账号下的其他 GPU workload。",
            ],
            "missing": [
                f"当前没有完全未分配 GPU；最佳候选仍只有约 {gpu['decision']['candidate_free_memory_mib']/1024:.1f} GiB 空闲且有常驻任务。",
                "空卡上 strict restore、peak memory 和 metadata/hash Gate。",
            ],
            "evidence": "results/l40s_gpu_availability_20260819.json",
        },
        {
            "id": "H5", "title": "Zero-motion Shadow / HIL", "status": "todo", "label": "首个允许测试",
            "done": [
                "定义 current-pose hold：真实 LowState→FK→当前 EEF→IK→safety，不 publish。",
                "结构化记录要求 publisher_created=false、robot_command_sent=false。",
            ],
            "meaning": [
                "这是把仿真安全链接到真实数据的最低风险测试。",
                "机器人不应因软件产生动作，模式保持 Damping。",
            ],
            "missing": [
                "阶段 A inventory 与 subscriber-only capture 尚未运行。",
                "真实 FK/contract parity、stale、断线和 no-command watchdog 尚未验证。",
            ],
            "evidence": "results/g1_first_motion_readiness_decision_20260818.json",
        },
        {
            "id": "H6–H7", "title": "Hardware command、watchdog 与真实反馈闭环", "status": "blocked", "label": "禁止动作",
            "done": [
                "已明确禁止直接运行官方 low-level/arm7 运动示例。",
                "已注册 hardware_execution_allowed=false 和 fail-closed runtime gates。",
            ],
            "meaning": [
                "当前没有可审计的 Unitree command adapter。",
                "从 Damping 切换模式本身也是硬件动作，不因 SSH 成功而获准。",
            ],
            "missing": [
                "官方关节/速度/力矩/电流/温度 limits 与正确控制接口。",
                "本地 watchdog、E-stop、通信丢失、balance/stance 和 command feedback。",
                "Shadow/HIL 通过后，先做非 VLA、非 Adaptive 的确定性单臂 free-space 小动作。",
            ],
            "evidence": "g1_policy_contract.yaml · Unitree SDK read-only audit",
        },
    ]

    css = """
    :root{--bg:#060913;--panel:#0e1628dc;--line:#ffffff14;--text:#eef5ff;--muted:#8ea0ba;--cyan:#35d7e8;--blue:#7185ff;--green:#41d99c;--amber:#ffc45c;--red:#ff6f89;--violet:#b594ff}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--text);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:radial-gradient(circle at 8% 2%,#153c63 0,transparent 26%),radial-gradient(circle at 92% 4%,#322264 0,transparent 28%),linear-gradient(180deg,#080d18,var(--bg) 44%);line-height:1.55}.shell{width:min(1440px,calc(100% - 32px));margin:auto;padding:28px 0 70px}.glass{background:var(--panel);border:1px solid var(--line);box-shadow:0 26px 90px #0007;backdrop-filter:blur(18px);border-radius:24px}.top{display:flex;align-items:center;justify-content:space-between;margin-bottom:34px}.brand{display:flex;align-items:center;gap:12px;font-weight:850}.logo{width:44px;height:44px;display:grid;place-items:center;border-radius:14px;background:linear-gradient(135deg,var(--cyan),var(--blue));color:#06111b}.stamp{color:var(--muted);font-size:12px}.hero{display:grid;grid-template-columns:1.5fr .5fr;gap:18px}.hero-main{padding:38px}.eyebrow{color:var(--cyan);font-size:12px;font-weight:900;letter-spacing:.17em}.hero h1{font-size:clamp(40px,6vw,78px);line-height:.98;letter-spacing:-.055em;margin:14px 0 20px;max-width:980px}.hero p{font-size:17px;color:#b9c7da;max-width:900px}.verdict{padding:28px;display:flex;flex-direction:column;justify-content:center}.verdict strong{font-size:25px;color:var(--red)}.verdict small{color:var(--muted);margin-top:9px}.metrics{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:18px 0}.metric{padding:18px}.metric label{display:block;color:var(--muted);font-size:11px;letter-spacing:.07em;text-transform:uppercase}.metric b{display:block;font-size:25px;margin-top:9px}.metric small{color:var(--muted)}nav{position:sticky;top:10px;z-index:5;display:flex;gap:7px;width:max-content;max-width:100%;overflow:auto;margin:24px 0;padding:8px;background:#0a1020e8;border:1px solid var(--line);border-radius:16px;backdrop-filter:blur(18px)}nav a{color:var(--muted);text-decoration:none;padding:9px 13px;border-radius:10px;font-weight:750;white-space:nowrap}nav a:hover{background:#ffffff0b;color:var(--text)}section{margin-top:26px}.section{padding:27px}.section-head{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:20px}.section-head h2{margin:0;font-size:27px}.section-head p{margin:6px 0 0;color:var(--muted)}.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:12px}.legend span:before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}.legend .pass:before{background:var(--green)}.legend .partial:before{background:var(--amber)}.legend .blocked:before{background:var(--red)}.legend .todo:before{background:var(--violet)}.flow{display:flex;gap:25px;overflow:auto;padding:8px 4px 20px}.flow-node{position:relative;min-width:170px;padding:18px;border:1px solid var(--line);border-radius:17px;background:#ffffff05}.flow-node:not(:last-child):after{content:"→";position:absolute;right:-21px;top:40%;color:#66809d}.flow-node span{font-size:10px;font-weight:900;letter-spacing:.12em}.flow-node b,.flow-node small{display:block}.flow-node b{margin:7px 0}.flow-node small{color:var(--muted)}.flow-node.pass{border-color:#41d99c55}.flow-node.pass span{color:var(--green)}.flow-node.partial{border-color:#ffc45c66}.flow-node.partial span{color:var(--amber)}.flow-node.blocked{border-color:#ff6f8955}.flow-node.blocked span{color:var(--red)}.flow-node.todo{border-color:#b594ff44}.flow-node.todo span{color:var(--violet)}.stage-card{padding:24px;margin-top:16px;border-left:3px solid var(--line)}.stage-card.pass{border-left-color:var(--green)}.stage-card.partial{border-left-color:var(--amber)}.stage-card.blocked{border-left-color:var(--red)}.stage-card.todo{border-left-color:var(--violet)}.stage-top{display:flex;justify-content:space-between;gap:12px}.stage-id{font-size:11px;font-weight:900;letter-spacing:.15em;color:var(--cyan)}.stage-card h3{font-size:22px;margin:10px 0 18px}.stage-columns{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}.stage-columns>div{border:1px solid var(--line);border-radius:15px;background:#ffffff04;padding:16px}.stage-columns h4{margin:0 0 10px;color:var(--cyan)}ul{margin:0;padding-left:19px;color:var(--muted);font-size:13px}li+li{margin-top:7px}.evidence{font-size:12px;color:#7488a5;margin:14px 0 0}.pill{display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:99px;font-size:11px;font-weight:800}.pill i{width:6px;height:6px;border-radius:50%}.pill.pass{color:var(--green);background:#41d99c14}.pill.pass i{background:var(--green)}.pill.partial{color:var(--amber);background:#ffc45c14}.pill.partial i{background:var(--amber)}.pill.blocked{color:var(--red);background:#ff6f8914}.pill.blocked i{background:var(--red)}.pill.todo{color:var(--violet);background:#b594ff14}.pill.todo i{background:var(--violet)}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}.callout{padding:22px}.callout h3{margin-top:0}.callout.allowed{border-color:#41d99c44}.callout.forbidden{border-color:#ff6f8955}.callout p{color:var(--muted)}.roadmap{display:grid;grid-template-columns:repeat(4,1fr);gap:13px}.roadmap article{padding:18px;border:1px solid var(--line);border-radius:16px;background:#ffffff04}.roadmap em{font-style:normal;color:var(--cyan);font-size:11px;font-weight:900;letter-spacing:.1em}.roadmap h3{margin:8px 0}.roadmap p{font-size:13px;color:var(--muted)}.flags{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.flag{display:flex;justify-content:space-between;padding:12px 14px;border:1px solid var(--line);border-radius:12px;color:var(--muted)}.flag b{color:var(--red);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.sources{columns:2}.sources a{color:var(--cyan);text-decoration:none;display:block;margin:8px 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}footer{text-align:center;color:#60718b;padding:30px;font-size:12px}@media(max-width:1100px){.metrics{grid-template-columns:repeat(3,1fr)}.stage-columns,.roadmap{grid-template-columns:1fr 1fr}}@media(max-width:760px){.shell{width:min(100% - 18px,1440px)}.hero,.two-col{grid-template-columns:1fr}.hero-main{padding:25px}.metrics,.stage-columns,.roadmap,.flags{grid-template-columns:1fr}.section{padding:19px}.sources{columns:1}}
    """

    sources = [
        "results/lgg100_author32_revalidation_summary.json",
        "results/lgg100_author32_t0_phase_speed_sweep_validation.json",
        "results/lgg100_author32_closed_loop_t0_offset008_baseline_realtime.json",
        "results/lgg100_author32_online_ik_convergence_trace.json",
        "results/lgg100_author32_online_ik_parameter_sweep.json",
        "results/unitree_g1_sdk2_readonly_audit_20260818.json",
        "results/l40s_gpu_availability_20260819.json",
        "g1_policy_contract.yaml",
        "UNITREE_G1_SDK_READONLY.md",
    ]

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G1 VLA · Simulation → Real Robot Readiness</title><style>{css}</style></head>
<body><main class="shell">
<div class="top"><div class="brand"><span class="logo">G1</span><span>VLA Safety Lab</span></div><div class="stamp">Evidence-bound · generated {html.escape(generated)}</div></div>
<section class="hero"><div class="glass hero-main"><div class="eyebrow">LGG100 · UNITREE G1 EDU · FAIL-CLOSED</div><h1>Simulation 到真机闭环：我们在哪里，还缺什么。</h1><p>当前离线模型输出与动作语义已有证据，Adaptive 展示了远处加速和近处减速；但 IK 全路径、Adaptive-OFF 闭环、100 ms 实时性、真实传感器与硬件 command 安全链尚未完成。</p></div><aside class="glass verdict"><span class="eyebrow">CURRENT VERDICT</span><strong>不允许真机动作</strong><small>只允许 read-only inventory、subscriber-only LowState 和 zero-motion Shadow。</small></aside></section>
<div class="metrics">
<div class="glass metric"><label>Neural output</label><b>{inference['finite_shape_passes']}/{inference['samples']}</b><small>finite [32,16]</small></div>
<div class="glass metric"><label>Semantic</label><b>{html.escape(semantic['best_hypothesis'])}</b><small>{semantic['usable_samples']} usable samples</small></div>
<div class="glass metric"><label>Far duration</label><b>−{abs(scenarios['far']['duration_change_percent']):.1f}%</b><small>offline chunk</small></div>
<div class="glass metric"><label>Closed-loop</label><b>{closed['completed_cycles']}</b><small>completed cycles</small></div>
<div class="glass metric"><label>Commit age</label><b>{commit_age:.2f} ms</b><small>limit {max_age:.0f} ms</small></div>
<div class="glass metric"><label>Tests</label><b>{tests['passed']}/{tests['total']}</b><small>local passed</small></div>
</div>
<nav><a href="#simulation">Simulation</a><a href="#hardware">真机闭环</a><a href="#allowed">当前权限</a><a href="#roadmap">最短路径</a><a href="#flags">Release Gates</a><a href="#sources">证据</a></nav>
<section id="simulation" class="glass section"><div class="section-head"><div><h2>Simulation Gate 图</h2><p>绿色代表证据通过；红色不是“没做”，而是现有证据明确不满足 Gate。</p></div><div class="legend"><span class="pass">通过</span><span class="partial">部分</span><span class="blocked">阻塞</span><span class="todo">未开始</span></div></div><div class="flow">{''.join(flow_node(node) for node in sim_flow)}</div></section>
<section>{''.join(stage_card(stage) for stage in sim_stages)}</section>
<section id="hardware" class="glass section"><div class="section-head"><div><h2>真实机器人闭环图</h2><p>目标是 Real sensors → Policy → IK/Safety → Command → Robot → Fresh feedback；当前只有认证 shell 和本地只读 adapter。</p></div></div><div class="flow">{''.join(flow_node(node) for node in real_flow)}</div></section>
<section>{''.join(stage_card(stage) for stage in hardware_stages)}</section>
<section id="allowed" class="two-col"><article class="glass callout allowed"><h3>现在可以做</h3>{list_html(["阶段 A：只读检查 Python、SDK、CycloneDDS、网卡和服务。","独立批准后运行 subscriber-only LowState 固定样本采集。","真实状态的 current-pose FK/IK zero-motion Shadow。","CPU-only IK 32-step sequential swept-path 回归。","只读复查 GPU；不干扰共享 workload。"])}<p>这些步骤都不能创建 robot command，也不能改变 Damping。</p></article><article class="glass callout forbidden"><h3>现在禁止做</h3>{list_html(["运行官方 g1_low_level_example.py 或 g1_arm7_sdk_dds_example.py。","创建 ChannelPublisher、LowCmd、arm_sdk、SportClient 或 Dex1 command。","切换 Ready/Walk/Control 或释放 motion service。","把仿真动作、LGG100 action 或 Adaptive timing 发送给机器人。","因 GPU 不足而停止或挤占其他任务。"])}<p>任何网络连接、SSH 登录或 tunnel listener 都不构成运动授权。</p></article></section>
<section id="roadmap" class="glass section"><div class="section-head"><div><h2>达到真机闭环的最短安全路径</h2><p>每一级失败都停止，不自动进入下一级。</p></div></div><div class="roadmap">
<article><em>01 · SIMULATION</em><h3>解除 IK/闭环/延迟阻塞</h3><p>完整 32-step 回归 → Adaptive-OFF 稳定抓取 → commit ≤100 ms → randomized/fault tests。</p></article>
<article><em>02 · READ ONLY</em><h3>真实 observation parity</h3><p>LowState、Dex1、三相机、时间同步、EEF 与物理场景标定。</p></article>
<article><em>03 · SHADOW/HIL</em><h3>真实输入，零命令</h3><p>current-pose hold、VLA output quarantine、stale/disconnect/watchdog/E-stop。</p></article>
<article><em>04 · STAGED MOTION</em><h3>确定性小动作后才到 VLA</h3><p>单臂 free-space → 双臂 → 桌面 → 轻物体 → Adaptive-OFF → paired Adaptive-ON。</p></article>
</div></section>
<section id="flags" class="glass section"><div class="section-head"><div><h2>当前 Release Gates</h2><p>这些值保持 false 是当前正确的安全结果。</p></div></div><div class="flags">{''.join(f'<div class="flag"><span>{name}</span><b>false</b></div>' for name in ['policy_task_quality_passed','physical_scene_calibration_verified','real_time_watchdog_validated','g1_contract_verified','g1_sim_eligible','g1_execution_enabled','hardware_execution_performed'])}</div></section>
<section id="sources" class="glass section"><div class="section-head"><div><h2>证据来源</h2><p>页面结论绑定到以下代码与 JSON；终端印象不能替代这些产物。</p></div></div><div class="sources">{''.join(f'<a href="{html.escape(source)}">{html.escape(source)}</a>' for source in sources)}</div></section>
<footer>G1 VLA Simulation → Deployment Summary · no hardware action performed</footer>
</main></body></html>"""


def main() -> None:
    OUTPUT.write_text(build())
    print(OUTPUT)


if __name__ == "__main__":
    main()
