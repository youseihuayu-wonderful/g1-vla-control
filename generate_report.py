#!/usr/bin/env python3
"""Generate the G1-EDU-only simulation and deployment report."""

from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path

from g1_policy_contract import CONTRACT_ID, CONTRACT_SHA256, CONTRACT_VERSION

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
HISTORY_PATH = RESULTS / "validation_history.json"
REPORT_PATH = ROOT / "validation_report.html"

CHECKS = [
    ("Robot", "Unitree G1 Education 29-DoF model", "pass", "MuJoCo Menagerie G1; policy scope excludes legs and torso"),
    ("Hands", "Official dual Dex1-1 geometry and mapping", "pass", "0 rad closed / 5.5 rad open; left/right mount tests pass"),
    ("Contract", "Frozen 16-D pelvis-frame EEF schema", "pass", f"{CONTRACT_ID} · {CONTRACT_VERSION}"),
    ("Contract", "MuJoCo state/action boundary round trip", "pass", "pelvis frame, metres, xyzw, 50 mm wrist +X site"),
    ("Observation", "Three policy RGB streams", "pass", "640×480 RGB → aspect-preserving resize with zero padding → 224×224; shared preprocessing"),
    ("Control", "G1 dual-arm IK and ordered 14-joint target", "pass", "same ordered joint list is frozen for the hardware adapter"),
    ("Safety", "EEF and joint command v/a/jerk filters", "pass", "both are in the production-contract simulation path"),
    ("Safety", "Phase-aware reachability/collision preflight", "pass", "exact-horizon contract fixture 50/50 accepted in grasp phase"),
    ("Dynamics", "G1 contract fixture baseline", "pass", "finite, command limits pass, pelvis stable and endpoint below 35 mm"),
    ("Adaptive", "G1 far→near local behavior", "pass", "17 cm lift: far median/max 1.30×/1.60×; near 0.50×; path unchanged"),
    ("Adaptive", "Real LGG100 task-speed benefit", "blocked", "real LGG100 calls=0 and deterministic adaptive run remains slower overall"),
    ("Dynamics", "Actual controller jerk", "partial", "MuJoCo position-actuator transient remains too large for a hardware-safety claim"),
    ("Robustness", "Mass/friction/object-position randomization", "pass", "20/20 existing scene-stability trials pass"),
    ("CI", "Ubuntu 22.04 headless G1 + Adaptive pipeline", "pass", "GitHub Actions Run 31637736729: 24/24 + 50-step dynamics + far/near coverage + cameras + workflow report"),
    ("Vision", "Simulation-to-hardware camera calibration", "blocked", "G1 EDU camera intrinsics/extrinsics, mounts and time sync not measured"),
    ("Hardware", "Official joint/torque/current limits", "blocked", "must come from the exact G1 EDU controller and firmware"),
    ("Hardware", "Unitree feedback/command adapter", "todo", "low-level state, hold, watchdog and communication-loss behavior not implemented"),
    ("Policy", "Production VLA selection", "partial", "LGG100/stack-cube-eef-24k selected at pinned revision cced7a…"),
    ("Policy", "LGG100 G1-contract eligibility", "blocked", "real-weight loader exists, but unpublished semantics mean g1_contract_verified=false"),
    ("Task", "Closed-loop neural stack-cube trials", "todo", "requires a G1-contract-verified policy; fixture motion is not task evidence"),
    ("Robot", "Shadow/HIL and staged G1 EDU run", "todo", "hardware gates must pass before any action execution"),
]


def load_json(name: str) -> dict:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def status_label(status: str) -> str:
    return {"pass": "已验证", "partial": "部分完成", "blocked": "阻塞", "todo": "待完成"}[status]


def render_checks() -> str:
    return "\n".join(
        f"<tr><td><b>{escape(area)}</b></td><td>{escape(item)}</td>"
        f"<td><span class='pill {status}'><i></i>{status_label(status)}</span></td>"
        f"<td class='muted'>{escape(note)}</td></tr>"
        for area, item, status, note in CHECKS
    )


def render_history(history: list[dict]) -> str:
    relevant = history[-8:]
    return "\n".join(
        "<div class='event'><time>{}</time><b>{}</b><p>{}</p></div>".format(
            escape(event.get("time", "")),
            escape(event.get("title", "Validation")),
            escape(event.get("detail", "")),
        )
        for event in reversed(relevant)
    )


def fnum(value, digits=3, scale=1.0, suffix="") -> str:
    if value is None:
        return "—"
    return f"{float(value) * scale:.{digits}f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    history = json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else []
    if args.record:
        history.append({
            "time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
            "title": "G1 EDU validation run",
            "detail": args.record,
        })
    HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n")

    tests = load_json("test_summary.json") or {"passed": 0, "total": 0}
    baseline = load_json("baseline.json")
    adaptive = load_json("adaptive.json")
    adaptive_phase = load_json("g1_adaptive_phase_validation.json")
    phase_coverage = adaptive_phase.get("coverage", {})
    phase_comparison = adaptive_phase.get("comparison", {})
    base_dyn = baseline.get("dynamics", {})
    adapt_dyn = adaptive.get("dynamics", {})
    preflight = baseline.get("preflight", {})
    passed = sum(x[2] == "pass" for x in CHECKS)
    partial = sum(x[2] == "partial" for x in CHECKS)
    progress = round(100 * (passed + 0.5 * partial) / len(CHECKS))
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    base_duration = base_dyn.get("simulated_duration_s")
    adapt_duration = adapt_dyn.get("simulated_duration_s")
    duration_change = None
    if base_duration and adapt_duration:
        duration_change = 100 * (adapt_duration - base_duration) / base_duration

    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 EDU 仿真与部署</title>
<style>
:root{{--bg:#070b12;--panel:#101824dc;--line:#ffffff18;--text:#eef6ff;--muted:#91a0b5;--cyan:#23d5e8;--blue:#6675ff;--green:#39d99a;--amber:#ffc14d;--red:#ff718b}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 8% 0,#123655 0,transparent 30%),radial-gradient(circle at 92% 8%,#2a205c 0,transparent 30%),var(--bg);color:var(--text);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}.shell{{width:min(1220px,calc(100% - 30px));margin:auto;padding:26px 0 60px}}.top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:34px}}.brand{{font-weight:800;display:flex;align-items:center;gap:12px}}.mark{{display:grid;place-items:center;width:42px;height:42px;border-radius:13px;background:linear-gradient(135deg,var(--cyan),var(--blue));color:#06101c}}a{{color:var(--cyan)}}.hero{{display:grid;grid-template-columns:1.4fr .6fr;gap:20px;align-items:stretch}}.hero h1{{font-size:clamp(42px,6vw,72px);line-height:.98;letter-spacing:-.055em;margin:12px 0 18px}}.hero p{{color:#b7c5d7;font-size:17px;line-height:1.7;max-width:800px}}.eyebrow{{color:var(--cyan);font-size:12px;font-weight:800;letter-spacing:.18em}}.glass{{background:var(--panel);border:1px solid var(--line);border-radius:22px;box-shadow:0 25px 70px #0006;backdrop-filter:blur(16px)}}.score{{display:grid;place-items:center;text-align:center;padding:25px}}.score strong{{font-size:52px}}.score span{{color:var(--muted)}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}}.metric{{padding:20px}}.metric label{{display:block;color:var(--muted);font-size:12px;margin-bottom:12px}}.metric strong{{font-size:27px}}.metric small{{color:var(--muted);margin-left:5px}}nav{{position:sticky;top:10px;z-index:10;display:flex;gap:7px;width:max-content;max-width:100%;overflow:auto;padding:8px;margin:0 0 25px;background:#0b111dec;border:1px solid var(--line);border-radius:16px;backdrop-filter:blur(18px)}}button{{border:0;border-radius:11px;padding:10px 15px;background:transparent;color:var(--muted);font-weight:750;cursor:pointer;white-space:nowrap}}button.active{{background:linear-gradient(135deg,var(--cyan),#79ecf5);color:#06111d}}.panel{{display:none}}.panel.active{{display:block}}.section{{padding:25px;margin-top:24px}}.head{{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:20px}}h2{{margin:0;font-size:23px}}h3{{margin:0 0 12px}}.muted,.head p{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{padding:18px;border:1px solid var(--line);border-radius:16px;background:#ffffff05}}.card em{{font-size:11px;font-style:normal;letter-spacing:.1em;color:var(--cyan)}}.card p{{color:var(--muted);line-height:1.55;font-size:13px;margin:8px 0 0}}.flow{{display:flex;align-items:center;gap:9px;overflow:auto;padding:5px 0}}.node{{padding:13px 15px;border:1px solid var(--line);border-radius:13px;background:#ffffff06;white-space:nowrap}}.arrow{{color:var(--cyan)}}table{{width:100%;border-collapse:collapse;font-size:14px}}th{{text-align:left;padding:11px;color:#738399;border-bottom:1px solid var(--line);font-size:11px;text-transform:uppercase;letter-spacing:.08em}}td{{padding:14px 11px;border-bottom:1px solid #ffffff0c;vertical-align:top}}code{{color:#a8eefa}}.pill{{display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:99px;font-size:11px;font-weight:750}}.pill i{{width:6px;height:6px;border-radius:50%}}.pass{{color:var(--green);background:#39d99a14}}.pass i{{background:var(--green)}}.partial{{color:var(--amber);background:#ffc14d14}}.partial i{{background:var(--amber)}}.blocked{{color:var(--red);background:#ff718b14}}.blocked i{{background:var(--red)}}.todo{{color:#aeb7ff;background:#6675ff18}}.todo i{{background:#aeb7ff}}.notice{{padding:16px;border:1px solid #ffc14d44;background:#ffc14d0c;color:#ddcc9f;border-radius:15px;line-height:1.6}}.rows .row{{display:flex;justify-content:space-between;gap:15px;padding:9px 0;border-bottom:1px dashed #ffffff16;color:var(--muted)}}.row b{{color:var(--text)}}.camera-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.camera{{overflow:hidden;border:1px solid var(--line);border-radius:15px;background:#050912}}.camera img{{width:100%;display:block;aspect-ratio:4/3;object-fit:cover}}.camera span{{display:block;padding:11px;color:var(--muted);font-size:12px}}.event{{border-left:2px solid var(--cyan);padding:0 0 20px 16px}}.event time{{font-size:11px;color:var(--muted);display:block}}.event b{{display:block;margin:5px 0}}.event p{{margin:0;color:var(--muted);line-height:1.5}}footer{{text-align:center;color:#68778c;margin-top:25px;font-size:12px}}@media(max-width:850px){{.hero,.grid,.camera-grid{{grid-template-columns:1fr}}.metrics{{grid-template-columns:1fr 1fr}}.section{{padding:18px;overflow:auto}}}}@media(max-width:520px){{.metrics{{grid-template-columns:1fr}}}}
</style></head><body><main class="shell">
<div class="top"><div class="brand"><div class="mark">G1</div>G1 EDU · 双 Dex1</div><a href="https://github.com/danielchen26/g1-vla-control">GitHub ↗</a></div>
<section class="hero"><div><div class="eyebrow">ONE ROBOT · ONE CONTRACT · SIM TO REAL</div><h1>只验证以后能部署到 G1 EDU 的链路。</h1><p>唯一目标是固定站立的 Unitree G1 Education + 双 Dex1 上肢操作。MuJoCo 与未来真机必须共享相机输入、16-D pelvis-frame EEF contract、安全拒绝、IK、14 关节顺序和 Dex1 方向；不兼容的机器人 action、过渡模型和 transport mock 已移出主流程。</p></div><div class="glass score"><div><strong>{progress}%</strong><br><span>G1 专用 Gate 完成度</span></div></div></section>
<section class="metrics"><div class="glass metric"><label>自动测试</label><strong>{tests.get('passed',0)}/{tests.get('total',0)}</strong><small>通过</small></div><div class="glass metric"><label>目标 VLA</label><strong>LGG100</strong><small>已选定</small></div><div class="glass metric"><label>G1 Contract Preflight</label><strong>{preflight.get('accepted',0)}/{preflight.get('checked',0)}</strong><small>通过</small></div><div class="glass metric"><label>真实 LGG100 调用</label><strong>0</strong><small>等待 GPU</small></div></section>
<nav><button data-tab="overview" class="active">G1 摘要</button><button data-tab="contract">唯一 Contract</button><button data-tab="simulation">G1 Simulation</button><button data-tab="real-vla">真正 VLA + Module</button><button data-tab="deployment">真机部署步骤</button><button data-tab="evidence">证据</button></nav>

<div id="overview" class="panel active">
<section class="glass section"><div class="head"><div><h2>唯一系统边界</h2><p>模拟器不是另一套机器人；它只是 G1 EDU I/O 的可替换实现。</p></div><span class="pill partial"><i></i>LGG100 Selected · GPU Pending</span></div><div class="flow"><div class="node">3× RGB + 16-D State</div><div class="arrow">→</div><div class="node">LGG100 stack-cube-eef-24k</div><div class="arrow">→</div><div class="node">16-D EEF Chunk</div><div class="arrow">→</div><div class="node">Preflight + EEF Filter</div><div class="arrow">→</div><div class="node">G1 双臂 IK</div><div class="arrow">→</div><div class="node">14 Joint Filter + Dex1</div><div class="arrow">→</div><div class="node">MuJoCo / G1 EDU</div></div></section>
<section class="glass section"><div class="head"><div><h2>保留什么</h2><p>每个模块都必须能原样复用于真机，或明确是 MuJoCo/硬件边界实现。</p></div></div><div class="grid"><div class="card"><em>SHARED</em><h3>Policy Contract</h3><p>图像 keys、RGB 预处理、state/action 逐维顺序、pelvis frame、xyzw、30 Hz、horizon 和 Dex1 范围。</p></div><div class="card"><em>SHARED</em><h3>Safety + IK</h3><p>chunk validator、stale 拒绝、碰撞 preflight、EEF/Joint limits、双臂 IK、watchdog 状态机。</p></div><div class="card"><em>BOUNDARY</em><h3>只替换 I/O</h3><p>现在由 MuJoCo 读相机/状态和接收 joint target；以后只替换为 Unitree SDK 与真实相机。</p></div></div></section>
<section class="glass section"><div class="head"><div><h2>当前 Gate</h2><p>测试通过不等于 G1 真机安全。</p></div></div><table><thead><tr><th>区域</th><th>检查</th><th>状态</th><th>证据边界</th></tr></thead><tbody>{render_checks()}</tbody></table></section>
</div>

<div id="contract" class="panel">
<section class="glass section"><div class="head"><div><h2>{CONTRACT_ID}</h2><p>版本 {CONTRACT_VERSION} · SHA-256 <code>{CONTRACT_SHA256[:16]}…</code></p></div><span class="pill pass"><i></i>项目定义已冻结</span></div><div class="notice"><b>规则：</b>训练、MuJoCo、policy server 和 G1 EDU adapter 必须引用同一个 contract ID 与 SHA。仅仅输出 16 个数不算兼容；frame、逐维语义、单位、EEF site、时序和 normalization 全部一致才允许进入仿真闭环。</div></section>
<section class="glass section"><div class="head"><div><h2>Observation</h2><p>未来真机必须产生完全相同的 policy 输入。</p></div></div><table><thead><tr><th>字段</th><th>定义</th><th>真机要求</th></tr></thead><tbody><tr><td><code>observation/cam_left_high</code></td><td>640×480 RGB → center crop → bilinear 224×224</td><td>标定高位相机，不允许替换/复制腕相机</td></tr><tr><td><code>observation/cam_left_wrist</code></td><td>同一共享预处理 → uint8 RGB [224,224,3]</td><td>左腕真实安装位姿、时间戳、丢帧检测</td></tr><tr><td><code>observation/cam_right_wrist</code></td><td>同一共享预处理 → uint8 RGB [224,224,3]</td><td>右腕独立图像，不做镜像</td></tr><tr><td><code>observation/state[16]</code></td><td>左 7 + 右 7 EEF pose + 双 Dex1</td><td>pelvis frame；m；xyzw；rad</td></tr><tr><td><code>prompt</code></td><td>已训练 canonical instruction</td><td>未知任务 fail-closed</td></tr></tbody></table></section>
<section class="glass section"><div class="head"><div><h2>Action → G1 Joint Command</h2><p>VLA 不控制腿、躯干或平衡。</p></div></div><table><thead><tr><th>维度</th><th>语义</th><th>单位/Frame</th></tr></thead><tbody><tr><td>0:3</td><td>左 EEF XYZ absolute target</td><td>m · pelvis</td></tr><tr><td>3:7</td><td>左 EEF quaternion</td><td>unit xyzw · pelvis</td></tr><tr><td>7:10</td><td>右 EEF XYZ absolute target</td><td>m · pelvis</td></tr><tr><td>10:14</td><td>右 EEF quaternion</td><td>unit xyzw · pelvis</td></tr><tr><td>14:16</td><td>左右 Dex1 motor target</td><td>rad · 0 closed / 5.5 open</td></tr><tr><td>IK output</td><td>左 7 + 右 7 absolute joint target</td><td>rad · contract 固定顺序</td></tr></tbody></table></section>
</div>

<div id="simulation" class="panel">
<section class="glass section"><div class="head"><div><h2>G1 EDU Production-Path Regression</h2><p>输入是 contract fixture，不伪装成 LGG100；输出链与未来 G1 相同。</p></div><span class="pill {'pass' if baseline.get('success') else 'blocked'}"><i></i>{'通过' if baseline.get('success') else '失败'}</span></div><div class="grid"><div class="card rows"><h3>8 cm Precision Fixture</h3><div class="row"><span>Preflight</span><b>{preflight.get('accepted',0)}/{preflight.get('checked',0)}</b></div><div class="row"><span>Adaptive scale</span><b>{fnum(adaptive.get('retiming',{}).get('scale_min'),2)}–{fnum(adaptive.get('retiming',{}).get('scale_max'),2)}×</b></div><div class="row"><span>Far ≥16 cm covered</span><b>NO</b></div><div class="row"><span>vs baseline</span><b>{fnum(duration_change,1,suffix='%')}</b></div></div><div class="card rows"><h3>17 cm Far→Near Coverage</h3><div class="row"><span>Preflight</span><b>{adaptive_phase.get('preflight',{}).get('accepted',0)}/{adaptive_phase.get('preflight',{}).get('checked',0)}</b></div><div class="row"><span>Far scale median/max</span><b>{fnum(phase_coverage.get('far_scale_median'),2)} / {fnum(phase_coverage.get('far_scale_max'),2)}×</b></div><div class="row"><span>Near scale</span><b>{fnum(phase_coverage.get('near_scale_median'),2)}×</b></div><div class="row"><span>该快时快 / 该慢时慢</span><b>{'PASS' if phase_comparison.get('local_phase_behavior_passed') else 'FAIL'}</b></div></div><div class="card rows"><h3>Evidence Boundary</h3><div class="row"><span>Adaptive overall faster</span><b>{'YES' if phase_comparison.get('adaptive_faster_overall') else 'NO'}</b></div><div class="row"><span>Real LGG100 chunk</span><b>NO</b></div><div class="row"><span>Task benefit proven</span><b>NO</b></div><div class="row"><span>Production enabled</span><b>NO</b></div></div></div><div class="notice" style="margin-top:18px"><b>准确结论：</b>G1 mechanics coverage 已观察到远距离中位/最大 <code>{fnum(phase_coverage.get('far_scale_median'),2)}× / {fnum(phase_coverage.get('far_scale_max'),2)}×</code>，近目标 <code>{fnum(phase_coverage.get('near_scale_median'),2)}×</code>，所以确定性 G1 路径做到了局部“该快时快、该慢时慢”。但 Adaptive 总用时仍比 baseline 多 <code>{fnum(phase_comparison.get('duration_delta_s'),3,suffix=' s')}</code>，且真实 LGG100 调用为 0；现在不能声称 LGG100 任务更快。</div></section>
<section class="glass section"><div class="head"><div><h2>只允许这一条 Simulation Gate</h2><p>任何 policy 先证明 G1 contract，再获得 MuJoCo action 权限。</p></div></div><table><thead><tr><th>Gate</th><th>输入</th><th>验收</th><th>权限</th></tr></thead><tbody><tr><td><b>S0 Contract</b></td><td>YAML + validators</td><td>ID/SHA、图像、state/action、frame、joint order 全一致</td><td>测试 fixture</td></tr><tr><td><b>S1 Observation</b></td><td>MuJoCo 3 RGB + state</td><td>与未来 G1 adapter 完全同 schema</td><td>Policy output-only</td></tr><tr><td><b>S2 Neural Offline</b></td><td>G1-verified checkpoint</td><td>finite [T,16]、norm、timing、fingerprint</td><td>仍不执行</td></tr><tr><td><b>S3 Preflight</b></td><td>同一保存 chunk</td><td>所有 target 的 IK、limit、collision 全过</td><td>MuJoCo 低速</td></tr><tr><td><b>S4 Closed Loop</b></td><td>每个 replan 的真实 observation</td><td>任务成功、无 stale、无 forbidden contact</td><td>仅 MuJoCo</td></tr><tr><td><b>S5 Randomized</b></td><td>相机/物体/摩擦/网络变化</td><td>预注册成功率与安全阈值</td><td>申请 Shadow</td></tr></tbody></table></section>
<section class="glass section notice"><b>LGG100 是已选生产 VLA，不是被替换的模型。</b>当前阶段先严格恢复真实权重并做 output-only。由于发布仓库缺作者 transform/golden sample，server 暂时必须保留 <code>g1_contract_verified=false</code>、<code>g1_action_compatible=false</code>；完成语义验证前不允许把 chunk 送入 G1 dynamics。</section>
</div>

<div id="real-vla" class="panel">
<section class="glass section"><div class="head"><div><h2>真正 LGG100 + 我们的 Adaptive Module</h2><p>完整命令和验收细节见 <code>REAL_LGG100_ADAPTIVE_WORKFLOW.md</code>；以下顺序不能跳过。</p></div><span class="pill blocked"><i></i>当前 R0 · 等待 GPU Host</span></div><div class="notice"><b>目标：</b>先证明真正 LGG100 能在冻结的 G1 contract 下稳定输出，再把同一个真实 chunk 分成 Baseline/Adaptive，最后做 multi-chunk 任务级配对。当前真实神经调用数为 <b>0</b>，所以 deterministic far/near 通过不能写成 LGG100 module 已通过。</div></section>
<section class="glass section"><div class="head"><div><h2>Phase A · 取得真实神经输出</h2><p>只允许 output-only，不执行 MuJoCo dynamics。</p></div></div><table><thead><tr><th>Gate</th><th>要做什么</th><th>入口 / 产物</th><th>通过与停止</th></tr></thead><tbody><tr><td><b>R0 · GPU</b></td><td>提供 Ubuntu NVIDIA SSH alias；检查 OS、GPU、VRAM、磁盘和 JAX backend</td><td><code>nvidia-smi</code>、JAX GPU；计划产物 <code>lgg100_gpu_preflight.json</code></td><td>CPU fallback、OOM、未授权 host 或必须公开 8000：停止</td></tr><tr><td><b>R1 · Pin</b></td><td>固定 OpenPI <code>15a9616…</code>、HF <code>cced7a…</code>、G1 contract ID/SHA</td><td>17-file checkpoint、metadata/norm SHA</td><td>revision/hash/文件树漂移：停止</td></tr><tr><td><b>R2 · Restore</b></td><td><code>remove_extra_params=False</code> 严格恢复真实 Orbax 权重</td><td><code>lgg100_candidate_server.py</code> + server log</td><td>missing/extra leaf、shape mismatch、OOM：停止；成功仍不代表 G1 compatible</td></tr><tr><td><b>R3 · Tunnel</b></td><td>只通过 SSH tunnel 连接本地 <code>127.0.0.1:8000</code></td><td>受控 WebSocket</td><td>端口暴露公网：停止</td></tr><tr><td><b>R4 · Output-only</b></td><td>先 3-call probe，再 30-call；三路 G1 observation → 真正 LGG100</td><td><code>lgg100_sim_smoke.py</code> → real JSON + fingerprinted NPZ</td><td>必须 30/30 finite <code>[32,16]</code>；shape/quat/timeout/metadata 错误：停止</td></tr></tbody></table></section>
<section class="glass section"><div class="head"><div><h2>Phase B · 证明输出语义就是 G1 16-D</h2><p>这是当前最关键的 Gate；16-D shape 本身不够。</p></div><span class="pill todo"><i></i>需实现 Semantic Validator</span></div><div class="grid"><div class="card"><em>R5 · DATA</em><h3>至少 30 个多 episode 时间点</h3><p>同步三路 RGB、raw 14-joint + 2 Dex1、candidate FK 16-D、future 50 frames 和 timestamp。</p></div><div class="card"><em>R5 · HYPOTHESES</em><h3>逐一排除错误解释</h3><p>pelvis/world、EEF site、xyzw/wxyz、absolute/delta、左右顺序、Dex1 方向、q01/q99、horizon 和 lag。</p></div><div class="card"><em>R5 · ATTESTATION</em><h3>Hash-bound 人工审阅</h3><p>绑定 HF/OpenPI/contract/norm/transform/sample manifest；只有通过才可产生 <code>g1_contract_verified=true</code>。</p></div></div><p class="notice" style="margin-top:18px"><b>停止条件：</b>多个语义假设无法区分、物理轨迹不合理、大量 IK/碰撞失败、需要补零/换左右手/逐帧手改，或没有 semantic attestation。禁止直接编辑 smoke JSON 解锁。</p></section>
<section class="glass section"><div class="head"><div><h2>Phase C · 将 Module 接到真实 LGG100 Chunk</h2><p>先补全 module context，再 preflight，再做同 chunk A/B。</p></div></div><table><thead><tr><th>Gate</th><th>必须完成</th><th>产物</th><th>通过标准</th></tr></thead><tbody><tr><td><b>R6 · Context</b></td><td>增加 task phase、object/robot clearance、tracking error、Dex1/contact、chunk age/stale、IK margin 和 stability</td><td><code>adaptive_speed_context.py</code> + validation JSON</td><td>靠近障碍/抓取/放置/stale 时禁止加速；只有稳定 free-space 可 &gt;1×</td></tr><tr><td><b>R7 · Offline Preflight</b></td><td>fingerprint → shape/finite/quat/range → 50/50 IK/joint/collision/command envelope</td><td><code>lgg100_real_chunk_preflight.json</code></td><td>任一 target 拒绝则不执行 dynamics</td></tr><tr><td><b>R8 · Same-chunk A/B</b></td><td>LGG100 只推理一次；两侧同 chunk SHA、同 action samples、同初态/IK/controller，只改 timestamps</td><td><code>lgg100_adaptive_ab.json</code></td><td>far &gt;1×、near/approach &lt;1×；limits 100%；jerk/error/contact 不恶化</td></tr></tbody></table><p class="muted">如果真实 chunk 没覆盖 far 或 near，结论只能是 coverage 不足，不能判 module 成功。Single-chunk 也不能证明叠积木成功率。</p></section>
<section class="glass section"><div class="head"><div><h2>Phase D · 真正 LGG100 闭环与任务级 A/B</h2><p>Baseline 自己先会完成任务，才有资格测试 Adaptive。</p></div></div><table><thead><tr><th>Gate</th><th>实验设计</th><th>数量 / 指标</th><th>停止条件</th></tr></thead><tbody><tr><td><b>R9 · Baseline Closed Loop</b></td><td>Adaptive OFF；observation → LGG100 → preflight → conservative stride → replan</td><td>multi-chunk timestamp/hash/replay；stale/timeout 必须 hold</td><td>Baseline 不会叠积木或不可 replay：先修 policy，不测加速</td></tr><tr><td><b>R10 · Development A/B</b></td><td>相同初态、场景、prompt、网络 profile 和 sampling seed 的配对闭环</td><td>至少 30 pairs</td><td>成功率/碰撞/掉落/jerk 恶化：module 关闭</td></tr><tr><td><b>R10 · Formal A/B</b></td><td>方块/质量/摩擦/视觉/网络随机化</td><td>建议至少 100 pairs；成功率、P50/P95/P99 时间和安全指标</td><td>未达到预注册阈值：<code>production_adaptive_enabled=false</code></td></tr><tr><td><b>R11 · Fault Injection</b></td><td>latency、jitter、掉线、stale、tracking error、IK near-limit、接触/滑落</td><td>所有异常优先 slow/hold/reject</td><td>异常时仍加速：停止</td></tr><tr><td><b>R12 · Release</b></td><td>汇总 neural/semantic/baseline/adaptive/safety gates</td><td>MuJoCo production verdict</td><td>仍不自动授权 G1 hardware</td></tr></tbody></table></section>
<section class="glass section"><div class="head"><div><h2>Module 的任务级初始验收阈值</h2><p>项目预注册初值，不是 Unitree 官方硬件限制。</p></div></div><div class="grid"><div class="card rows"><h3>Task</h3><div class="row"><span>Success non-inferiority</span><b>≥ Baseline − 5pp</b></div><div class="row"><span>Median completion time</span><b>至少改善 10%</b></div><div class="row"><span>P95 completion time</span><b>恶化不超过 5%</b></div></div><div class="card rows"><h3>Safety</h3><div class="row"><span>Forbidden collision/drop</span><b>不得增加</b></div><div class="row"><span>P95 actual jerk</span><b>恶化不超过 5%</b></div><div class="row"><span>Command hard limits</span><b>100%</b></div></div><div class="card rows"><h3>Fail Closed</h3><div class="row"><span>Timeout / stale</span><b>全部 hold</b></div><div class="row"><span>未知 phase / 低 clearance</span><b>禁止加速</b></div><div class="row"><span>任一 Gate 失败</span><b>Module OFF</b></div></div></div></section>
<section class="glass section notice"><b>下一次实际执行顺序：</b>R0 GPU → R1 pin → R2 strict restore → R3 tunnel → R4 3-call/30-call → R5 semantic attestation → R6 context → R7 preflight → R8 same-chunk A/B → R9 Baseline closed loop → R10 30/100 paired trials → R11 faults → R12 release。当前用户只需提供 Ubuntu NVIDIA 的 SSH alias 或 <code>GPU_USER@GPU_HOST</code>；不要发送密码。</section>
</div>

<div id="deployment" class="panel">
<section class="glass section"><div class="head"><div><h2>MuJoCo → G1 EDU Step-by-step</h2><p>只列最终机器人会使用的交付物。</p></div><span class="pill blocked"><i></i>当前 D0 完成 · D1 待办</span></div><table><thead><tr><th>Step</th><th>具体工作</th><th>产物</th><th>停止条件</th></tr></thead><tbody><tr><td><b>D0 · Freeze</b></td><td>冻结三相机、16-D state/action、EEF site、30 Hz、horizon、14 joint order</td><td><code>g1_policy_contract.yaml</code> + validators</td><td>任何模块 contract SHA 不同</td></tr><tr><td><b>D1 · Hardware parity</b></td><td>确定 G1 EDU firmware/SDK/control mode；测相机内外参、时间同步、EEF site；取得官方 limits</td><td><code>g1_hardware_profile.yaml</code> + calibration</td><td>型号、关节顺序、控制模式或 limits 不明确</td></tr><tr><td><b>D2 · Dataset</b></td><td>100 episodes 转换为冻结的 pelvis-frame 16-D contract；episode-level split；canonical prompt</td><td>OpenPI/LeRobot dataset + manifest + norm stats</td><td>逐帧不可追溯、frame leakage、round-trip 失败</td></tr><tr><td><b>D3 · LGG100 Restore</b></td><td>固定 HF revision <code>cced7a…</code>，在 Ubuntu NVIDIA 上 strict Orbax restore；恢复/验证作者 transform 与 horizon</td><td>真实 server metadata + model/config audit + golden sample</td><td>参数树不完全匹配，或 action 语义仍无法验证</td></tr><tr><td><b>D4 · Offline Gate</b></td><td>真实 policy 只生成保存 chunk，不执行；验证 schema、时序、range、IK、碰撞</td><td><code>lgg100_vla_smoke_real.json</code> + fingerprinted NPZ</td><td><code>g1_contract_verified</code> 不是 true</td></tr><tr><td><b>D5 · MuJoCo</b></td><td>原速/低速 multi-chunk 闭环、随机化、网络 jitter、stale/断线</td><td>任务成功率 + failure replay</td><td>安全/任务阈值任一不通过</td></tr><tr><td><b>D6 · Shadow/HIL</b></td><td>G1 真实相机和 state 进入 policy；机器人保持 hold，只记录建议 action</td><td>真机 parity/shadow logs</td><td>时序、frame、相机域、EEF/joint 不一致</td></tr><tr><td><b>D7 · Staged action</b></td><td>硬 E-stop + 支撑：单臂 free-space → 双臂 → 桌面 → 轻物体 → 已训练任务</td><td>operator checklist + run logs</td><td>watchdog、hold、通信中断、官方 limits 未独立通过</td></tr></tbody></table></section>
<section class="glass section"><div class="head"><div><h2>真机运行拓扑</h2><p>Policy 可远程运行，但停止能力必须在 G1 本地。</p></div></div><div class="grid"><div class="card"><em>GPU</em><h3>LGG100 Policy Server</h3><p>固定 revision 的真实 Orbax 权重；只接收冻结 observation，返回带 contract metadata/timestamp 的 16-D chunk；端口只走受控局域网或 SSH tunnel。</p></div><div class="card"><em>ON G1</em><h3>Realtime Supervisor</h3><p>相机/状态同步、stale 拒绝、preflight、安全 limits、IK、watchdog、hold/E-stop；网络断开不依赖 GPU 才能停。</p></div><div class="card"><em>UNITREE</em><h3>Official Controller</h3><p>腿和平衡继续由官方控制器负责。上肢 joint target 必须经过准确 firmware/control-mode 的官方限制和反馈验证。</p></div></div></section>
<section class="glass section notice"><b>现在只推进 LGG100：</b>先取得 Ubuntu NVIDIA host，严格加载 revision <code>cced7a…</code>并生成真实 output-only chunk；并行建立 exact G1 EDU hardware profile。LGG100 语义验证、真实 MuJoCo closed loop 和硬件安全 profile 缺一项，都不允许真机 action。</section>
</div>

<div id="evidence" class="panel">
<section class="glass section"><div class="head"><div><h2>三路 G1 Policy Observation</h2><p>MuJoCo 当前渲染；真机必须用同 keys/shape/color order 替换。</p></div></div><div class="camera-grid"><div class="camera"><img src="results/camera_observations/cam_left_high.png"><span>cam_left_high</span></div><div class="camera"><img src="results/camera_observations/cam_left_wrist.png"><span>cam_left_wrist</span></div><div class="camera"><img src="results/camera_observations/cam_right_wrist.png"><span>cam_right_wrist</span></div></div></section>
<section class="glass section"><div class="head"><div><h2>关键文件</h2><p>主链不再依赖其他机器人 action schema。</p></div></div><table><thead><tr><th>文件</th><th>职责</th><th>真机复用</th></tr></thead><tbody><tr><td><code>g1_policy_contract.yaml</code></td><td>唯一版本化语义</td><td>直接复用</td></tr><tr><td><code>g1_policy_contract.py</code></td><td>observation/action/metadata fail-closed validator</td><td>直接复用</td></tr><tr><td><code>g1_mujoco_bridge.py</code></td><td>MuJoCo state/image 与 pelvis/world 边界</td><td>由 G1 hardware bridge 替换</td></tr><tr><td><code>safety_governor.py</code></td><td>preflight、EEF 和 joint command filters</td><td>参数换成官方硬件 limits 后复用</td></tr><tr><td><code>g1_dual_arm_ik.py</code></td><td>固定 14-joint 双臂 IK</td><td>校准模型/反馈后复用</td></tr><tr><td><code>run_simulation.py</code></td><td>完整 production-path contract regression</td><td>进入真机前持续回归</td></tr><tr><td><code>g1_adaptive_phase_validation.py</code></td><td>17 cm G1 far→near 加速/减速覆盖</td><td>真实 LGG100 chunk 到位后复用同一判定</td></tr></tbody></table></section>
<section class="glass section"><div class="head"><div><h2>验证历史</h2><p>这里只记录 G1 仿真、contract、安全和部署主线；完整改动仍由 Git 审计。</p></div></div>{render_history(history)}</section>
</div>
<footer>Generated {escape(generated)} · contract {CONTRACT_ID} · hardware execution disabled</footer>
</main><script>
const buttons=[...document.querySelectorAll('button[data-tab]')],panels=[...document.querySelectorAll('.panel')];
function show(id){{if(!document.getElementById(id))id='overview';buttons.forEach(b=>b.classList.toggle('active',b.dataset.tab===id));panels.forEach(p=>p.classList.toggle('active',p.id===id));history.replaceState(null,'','#'+id)}}
buttons.forEach(b=>b.onclick=()=>show(b.dataset.tab));show(location.hash.slice(1)||'overview');
</script></body></html>'''
    REPORT_PATH.write_text(html)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
