#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the single-table Simulation → G1 deployment summary."""

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


def bullet_cell(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


def status_pill(status: str, label: str) -> str:
    return f'<span class="pill {html.escape(status)}"><i></i>{html.escape(label)}</span>'


def table_row(row: dict[str, Any]) -> str:
    return f"""
    <tr class="{html.escape(row['status'])}">
      <td class="stage"><span class="domain">{html.escape(row['domain'])}</span><b>{html.escape(row['id'])}</b><strong>{html.escape(row['title'])}</strong></td>
      <td class="state">{status_pill(row['status'], row['label'])}</td>
      <td>{bullet_cell(row['done'])}</td>
      <td>{bullet_cell(row['result'])}</td>
      <td>{bullet_cell(row['meaning'])}</td>
      <td>{bullet_cell(row['missing'])}</td>
      <td>{bullet_cell(row['next'])}</td>
      <td class="evidence">{bullet_cell(row['evidence'])}</td>
    </tr>
    """.strip()


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
            "domain": "REAL ROBOT", "id": "H0", "title": "网络与认证 Shell", "status": "partial", "label": "仅连接",
            "done": [
                "机器人由操作员保持 Damping、安全绳/支撑和 E-stop 就绪。",
                "Mac→开发机 192.168.1.13→机器人 192.168.123.164 登录成功。",
                "建立独立 L40S ControlMaster、keeper 和 loopback tunnel。",
            ],
            "result": ["Prompt: unitree@unitree-g1-nx。", "SSH shell 可用；没有执行机器人命令。", "开发机链路曾观察到较大 RTT jitter。"],
            "meaning": ["已具备只读 inventory 入口。", "SSH 在线不代表实时链路或动作权限。"],
            "missing": ["机器人内部实时 supervisor 架构。", "稳定有线延迟、断线与恢复测试。"],
            "next": ["只执行阶段 A inventory；保持 Damping。"],
            "evidence": ["results/g1_robot_ssh_connection_status_20260818.json", "CONNECTION_RUNBOOK.md"],
        },
        {
            "domain": "REAL ROBOT", "id": "H1", "title": "Unitree SDK / LowState", "status": "partial", "label": "本地完成",
            "done": [
                f"固定官方 unitree_sdk2_python {sdk['official_sources']['python']['commit'][:10]}…。",
                "确认 G1 使用 unitree_hg、rt/lowstate，双臂索引 15–28。",
                "实现 subscriber-only adapter 和 AST 禁写测试。",
            ],
            "result": ["本地 extraction/fail-closed 测试通过。", "SDK 未上传、未安装、未在机器人初始化 DDS。", "官方运动示例被明确禁止。"],
            "meaning": ["有可审计的真实反馈读取代码。", "还没有任何真实 LowState 样本。"],
            "missing": ["Python/CycloneDDS/SDK/网卡 inventory。", "29-DOF variant、单位、IMU 顺序、tick 和 freshness 真机验证。"],
            "next": ["独立确认后执行只读 inventory，再决定是否运行固定样本 subscriber。"],
            "evidence": ["UNITREE_G1_SDK_READONLY.md", "g1_unitree_lowstate.py", "results/unitree_g1_sdk2_readonly_audit_20260818.json"],
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
                "远端 port 8000 未监听，LGG100 server 未运行。",
                f"最佳候选 GPU 仍只有约 {gpu['decision']['candidate_free_memory_mib']/1024:.1f} GiB 空闲且有其他 workload。",
            ],
            "meaning": ["Tunnel listener 只证明网络路径，不证明 policy 可用。"],
            "missing": ["完全空闲或管理员明确分配的 GPU。", "Strict restore peak memory 与 server metadata/hash Gate。"],
            "next": ["等待现有任务正常结束；不停止或挤占其他任务。"],
            "evidence": ["results/l40s_gpu_availability_20260819.json"],
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

    css = """
    :root{--bg:#070b14;--panel:#0e1727;--line:#ffffff17;--text:#eef5ff;--muted:#9aabc1;--cyan:#37d9e8;--green:#45dca1;--amber:#ffc75d;--red:#ff738d;--violet:#b69aff}*{box-sizing:border-box}html{color-scheme:dark}body{margin:0;background:radial-gradient(circle at 8% 0,#173d61 0,transparent 27%),radial-gradient(circle at 92% 0,#332268 0,transparent 25%),var(--bg);color:var(--text);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.5}.shell{width:min(1880px,calc(100% - 28px));margin:auto;padding:26px 0 60px}.hero{display:flex;justify-content:space-between;align-items:end;gap:24px;padding:28px;margin-bottom:16px;border:1px solid var(--line);border-radius:22px;background:#0e1727dd;box-shadow:0 28px 90px #0007;backdrop-filter:blur(18px)}.eyebrow{color:var(--cyan);font-size:11px;font-weight:900;letter-spacing:.16em}.hero h1{font-size:clamp(32px,4vw,58px);line-height:1;margin:10px 0 12px;letter-spacing:-.045em}.hero p{margin:0;color:var(--muted);max-width:1050px}.verdict{text-align:right;min-width:240px}.verdict b{display:block;color:var(--red);font-size:20px}.verdict small{color:var(--muted)}.legend{display:flex;gap:13px;flex-wrap:wrap;padding:12px 18px;color:var(--muted);font-size:12px}.legend span:before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}.legend .pass:before{background:var(--green)}.legend .partial:before{background:var(--amber)}.legend .blocked:before{background:var(--red)}.legend .todo:before{background:var(--violet)}.table-wrap{overflow:auto;max-height:calc(100vh - 210px);border:1px solid var(--line);border-radius:20px;background:#0b1220e8;box-shadow:0 28px 90px #0008}table{width:100%;min-width:1900px;border-collapse:separate;border-spacing:0;font-size:13px}caption{text-align:left;padding:15px 18px;color:var(--muted);border-bottom:1px solid var(--line)}thead{position:sticky;top:0;z-index:8;background:#131e31}th{text-align:left;padding:14px 15px;color:#b8c7db;font-size:11px;letter-spacing:.08em;text-transform:uppercase;border-bottom:1px solid #ffffff24}th:nth-child(1){width:200px}th:nth-child(2){width:100px}th:nth-child(3),th:nth-child(4),th:nth-child(5),th:nth-child(6),th:nth-child(7){width:270px}th:nth-child(8){width:230px}td{padding:16px 15px;vertical-align:top;border-bottom:1px solid #ffffff0d;border-right:1px solid #ffffff09;background:#0d1625aa}tr:hover td{background:#142138}tr.pass td:first-child{box-shadow:inset 4px 0 var(--green)}tr.partial td:first-child{box-shadow:inset 4px 0 var(--amber)}tr.blocked td:first-child{box-shadow:inset 4px 0 var(--red)}tr.todo td:first-child{box-shadow:inset 4px 0 var(--violet)}.stage{position:sticky;left:0;z-index:3;background:#101b2d!important}.stage .domain{display:block;color:var(--cyan);font-size:9px;font-weight:900;letter-spacing:.14em}.stage b{display:block;margin:6px 0;color:#7891af}.stage strong{display:block;font-size:16px}.state{position:sticky;left:200px;z-index:3;background:#101b2d!important}.pill{display:inline-flex;align-items:center;gap:6px;padding:6px 9px;border-radius:99px;font-size:11px;font-weight:850;white-space:nowrap}.pill i{width:6px;height:6px;border-radius:50%}.pill.pass{color:var(--green);background:#45dca116}.pill.pass i{background:var(--green)}.pill.partial{color:var(--amber);background:#ffc75d16}.pill.partial i{background:var(--amber)}.pill.blocked{color:var(--red);background:#ff738d16}.pill.blocked i{background:var(--red)}.pill.todo{color:var(--violet);background:#b69aff16}.pill.todo i{background:var(--violet)}ul{margin:0;padding-left:17px;color:var(--muted)}li+li{margin-top:8px}.evidence li{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:#7890ae;overflow-wrap:anywhere}footer{text-align:center;color:#64758c;padding:22px;font-size:11px}@media(max-width:760px){.shell{width:calc(100% - 12px)}.hero{align-items:start;flex-direction:column;padding:20px}.verdict{text-align:left}.table-wrap{max-height:calc(100vh - 260px)}}
    """

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G1 VLA · Simulation 与真机闭环总表</title><style>{css}</style></head>
<body><main class="shell">
<header class="hero"><div><div class="eyebrow">LGG100 · UNITREE G1 EDU · EVIDENCE-BOUND SUMMARY</div><h1>Simulation 与真机闭环总表</h1><p>所有工作、结果、含义、缺口、下一步和证据集中在下方唯一表格。横向滚动查看全部八列，前两列固定。</p></div><div class="verdict"><b>当前不允许真机动作</b><small>只允许 read-only / subscriber-only / zero-motion</small></div></header>
<div class="legend"><span class="pass">通过</span><span class="partial">部分完成</span><span class="blocked">阻塞</span><span class="todo">未开始</span></div>
<div class="table-wrap"><table><caption>Generated {html.escape(generated)} · Neural {inference['finite_shape_passes']}/{inference['samples']} · Closed-loop {closed['completed_cycles']} cycles · Commit {commit_age:.2f}/{maximum_age:.0f} ms · Tests {tests['passed']}/{tests['total']}</caption>
<thead><tr><th>阶段</th><th>状态</th><th>我们做了什么</th><th>当前结果 / 数据</th><th>这意味着什么</th><th>还差什么</th><th>下一步</th><th>证据</th></tr></thead>
<tbody>{''.join(table_row(row) for row in rows)}</tbody></table></div>
<footer>G1 VLA single-table summary · g1_execution_enabled=false · no hardware action performed</footer>
</main></body></html>"""


def main() -> None:
    OUTPUT.write_text(build())
    print(OUTPUT)


if __name__ == "__main__":
    main()
