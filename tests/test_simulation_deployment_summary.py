import html as html_lib
import json
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_simulation_deployment_summary import (
    PRE_REAL_IMMEDIATE,
    PRE_REAL_NOT_REPLACEABLE,
    PRE_REAL_SIMULATION_PRIORITIES,
    PRE_REAL_WAIT_L40S,
    build,
    build_public,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "simulation_deployment_summary.html"


class SimulationDeploymentSummaryTests(unittest.TestCase):
    def test_dashboard_contains_current_evidence_metrics(self):
        rendered = build()
        tests = json.loads((ROOT / "results" / "test_summary.json").read_text())
        for expected in (
            "50/50",
            "absolute_xyzw_lr",
            "119.94 ms",
            "Gate ≤ 100 ms",
            f"{tests['passed']}/{tests['total']}",
            "0 cycles",
            "15 组候选",
        ):
            self.assertIn(expected, rendered)

    def test_render_has_professional_tabbed_information_architecture(self):
        rendered = build()
        self.assertEqual(rendered.count("<table"), 3)
        self.assertEqual(rendered.count("</table>"), 3)
        for tab, panel in (
            ("tab-current", "panel-current"),
            ("tab-next", "panel-next"),
            ("tab-speed", "panel-speed"),
            ("tab-simulation", "panel-simulation"),
            ("tab-hardware", "panel-hardware"),
            ("tab-updates", "panel-updates"),
        ):
            self.assertIn(f'id="{tab}" role="tab"', rendered)
            self.assertIn(f'id="{panel}" role="tabpanel"', rendered)
        for heading in (
            "当前状态",
            "下一步计划",
            "速度模块验证明细",
            "Simulation 详细进展",
            "之后的真机阶段",
            "开发更新",
            "已完成工作",
            "客观结果 / 数据",
            "证据含义",
            "未完成项",
            "后续动作",
        ):
            self.assertIn(heading, rendered)
        self.assertIn("ArrowRight", rendered)
        self.assertIn("aria-selected", rendered)

    def test_desktop_tabs_are_a_left_vertical_column(self):
        rendered = build()
        self.assertIn('<div class="workspace">', rendered)
        self.assertIn('aria-orientation="vertical"', rendered)
        self.assertIn("grid-template-columns:220px minmax(0,1fr)", rendered)
        self.assertIn("flex-direction:column", rendered)
        self.assertIn("ArrowDown", rendered)
        self.assertIn("syncTabOrientation", rendered)

    def test_terminology_has_links_and_detailed_hover_tooltips(self):
        rendered = build()
        term_links = re.findall(r'<a class="term"[^>]+data-tip="([^"]+)"[^>]*>([^<]+)</a>', rendered)
        self.assertGreater(len(term_links), 40)
        linked_terms = {term for _, term in term_links}
        for expected in ("IK", "FK", "EEF", "LowState", "DDS", "watchdog", "Adaptive-OFF"):
            self.assertIn(expected, linked_terms)
        self.assertTrue(all(len(tip) >= 20 for tip, _ in term_links))
        self.assertIn('id="term-tooltip"', rendered)
        self.assertIn("mouseenter", rendered)
        self.assertIn("focus", rendered)

    def test_gen15_row_uses_the_correct_generalist_ai_model(self):
        report = build()
        plain = html_lib.unescape(re.sub(r"<[^>]+>", "", report))
        self.assertIn("Generalist AI GEN-1.5 启发", plain)
        self.assertIn("3–12 秒单次 demonstration", plain)
        self.assertIn("one-shot 平均成功率 59%±10%", plain)
        self.assertIn("目标模型准确识别为 Generalist AI GEN-1.5", plain)

    def test_speed_module_table_separates_verified_and_missing_evidence(self):
        report = build()
        plain = html_lib.unescape(re.sub(r"<[^>]+>", "", report))
        for expected in (
            "模块边界与不变量",
            "Fail-closed Hold 条件",
            "阶段速度策略",
            "MuJoCo 状态测量与 Context 性能",
            "Deterministic far-to-near Coverage",
            "真实 LGG100 单 Chunk Phase Sweep",
            "Adaptive-OFF 闭环基线",
            "任务级 Adaptive-ON 资格",
            "Near behavior0.50×",
            "Far duration−6.90%",
            "Mixed transition未通过",
            "Task-level speedup未证明",
            "controlled_phase_speed_behavior_passed=false",
        ):
            self.assertIn(expected, plain)
        speed_panel = report.split('id="panel-speed"', 1)[1].split('id="panel-simulation"', 1)[0]
        self.assertIn('class="speed-evidence-list"', speed_panel)
        self.assertEqual(speed_panel.count('data-speed-row="'), 10)
        self.assertNotIn("<table", speed_panel)
        self.assertNotIn("table-shell", speed_panel)
        self.assertIn("grid-template-columns:repeat(3,minmax(0,1fr))", report)
        next_panel = report.split('id="panel-next"', 1)[1].split('id="panel-speed"', 1)[0]
        self.assertNotIn("GEN-1.5 式数据基础设施", next_panel)
        self.assertNotIn("Physical-prompt recorder/replay 格式", next_panel)

    def test_language_is_objective_and_names_shihua_yu(self):
        report = build()
        plain = html_lib.unescape(re.sub(r"<[^>]+>", "", report))
        self.assertIn('<meta name="author" content="Shihua Yu">', report)
        self.assertGreaterEqual(plain.count("Shihua Yu"), 3)
        self.assertNotIn("我们", plain)
        self.assertNotIn("我下一步", plain)
        self.assertIn("缺少证据的项目不推定为通过", plain)

    def test_pre_real_simulation_plan_is_partitioned_into_tabs(self):
        report = build()
        plain = html_lib.unescape(re.sub(r"<[^>]+>", "", report))
        for expected in (
            "真机前执行计划",
            "当前可执行 · 不依赖空闲 GPU",
            "必须等 L40S 空闲",
            "Simulation 无法替代的部分",
            "客观执行顺序",
            "S4 完整 32-step sequential IK + swept-path 回归是当前最高优先级；该工作不涉及任何真机命令。",
            *PRE_REAL_IMMEDIATE,
            *PRE_REAL_WAIT_L40S,
            *PRE_REAL_NOT_REPLACEABLE,
        ):
            self.assertIn(expected, plain)
        for priority, stage, work, gate in PRE_REAL_SIMULATION_PRIORITIES:
            self.assertIn(priority, plain)
            self.assertIn(stage, plain)
            self.assertIn(work, plain)
            self.assertIn(gate, plain)

    def test_public_report_redacts_private_topology_and_rewrites_links(self):
        report = build_public()
        for forbidden in (
            "192.168.", "10.145.", "10.188.", "unitree@", "yixiao@", "user1@",
            "/Users/", "/home/", "unitree-g1-nx", "nnmc65", "shihua-vla",
        ):
            self.assertNotIn(forbidden, report)
        self.assertIn("PUBLIC SANITIZED", report)
        self.assertIn("private LAN identities redacted", report)
        relative_links = re.findall(r'href="(?!https://|http://|#)([^"]+)"', report)
        self.assertEqual(relative_links, [])

    def test_current_robot_and_gpu_status_uses_fresh_login_inventory(self):
        report = build()
        plain = html_lib.unescape(re.sub(r"<[^>]+>", "", report))
        self.assertIn("L40S 认证 shell 已在 Herdr w1:p3 建立", plain)
        self.assertIn("实时 free range=11.0–11.0 GiB", plain)
        self.assertIn("fully_idle_gpu_count=0", plain)
        self.assertIn("瞬时 utilization=0% 不等于空闲", plain)
        self.assertIn("独立 Q0.5 当前 RUNNING", plain)
        self.assertIn("目标真实 inference 4 小时", plain)
        self.assertIn("不是 dummy occupancy", plain)
        self.assertIn("authenticated nested shell 均已建立", plain)
        self.assertIn("隔离 subscriber-only Python environment", plain)
        self.assertIn("真实 LowState 固定读取：100/100 samples", plain)
        self.assertIn("duplicate ticks=4", plain)
        self.assertIn("robot_connection_available=true", plain)

    def test_checked_report_keeps_hardware_gates_closed(self):
        report = REPORT.read_text()
        self.assertIn("当前不允许真机动作", report)
        self.assertIn("g1_execution_enabled=false", report)
        self.assertIn("hardware_execution_performed=false", report)
        self.assertIn("明确禁止官方 low-level/arm7 运动示例", report)


if __name__ == "__main__":
    unittest.main()
