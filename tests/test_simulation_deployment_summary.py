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
    def test_single_table_contains_current_evidence_metrics(self):
        rendered = build()
        tests = json.loads((ROOT / "results" / "test_summary.json").read_text())
        for expected in (
            "50/50",
            "absolute_xyzw_lr",
            "119.94/100 ms",
            f"{tests['passed']}/{tests['total']}",
            "Closed-loop 0 cycles",
            "15 组候选",
        ):
            self.assertIn(expected, rendered)

    def test_render_is_exactly_one_detailed_table(self):
        rendered = build()
        self.assertEqual(rendered.count("<table>"), 1)
        self.assertEqual(rendered.count("</table>"), 1)
        for heading in (
            "阶段",
            "状态",
            "我们做了什么",
            "当前结果 / 数据",
            "这意味着什么",
            "还差什么",
            "下一步",
            "证据",
        ):
            self.assertIn(heading, rendered)
        self.assertNotIn("flow-node", rendered)
        self.assertNotIn("stage-card", rendered)

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
        self.assertNotIn("GR00T N1.5", plain)

    def test_pre_real_simulation_plan_is_published_verbatim(self):
        report = build()
        plain = html_lib.unescape(re.sub(r"<[^>]+>", "", report))
        for expected in (
            "可以。HTML 里在真机之前，仍有大量工作可以完全在 Simulation/Replay 中完成。优先级如下。",
            "当前可以立即做、不依赖空闲 GPU",
            "必须等 L40S 空闲",
            "Simulation 无法替代的部分",
            "因此最合理的执行顺序是：",
            "我下一步可以直接从 S4 完整 32-step sequential IK + swept-path 回归 开始，不涉及任何真机命令。",
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
        for forbidden in ("192.168.", "10.145.", "10.188.", "unitree@", "yixiao@", "/Users/"):
            self.assertNotIn(forbidden, report)
        self.assertIn("PUBLIC SANITIZED", report)
        self.assertIn("身份已脱敏", report)
        relative_links = re.findall(r'href="(?!https://|http://|#)([^"]+)"', report)
        self.assertEqual(relative_links, [])

    def test_checked_report_keeps_hardware_gates_closed(self):
        report = REPORT.read_text()
        self.assertIn("当前不允许真机动作", report)
        self.assertIn("g1_execution_enabled=false", report)
        self.assertIn("hardware_execution_performed=false", report)
        self.assertIn("明确禁止官方 low-level/arm7 运动示例", report)


if __name__ == "__main__":
    unittest.main()
