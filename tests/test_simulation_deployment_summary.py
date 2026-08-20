import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_simulation_deployment_summary import build


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

    def test_checked_report_keeps_hardware_gates_closed(self):
        report = REPORT.read_text()
        self.assertIn("当前不允许真机动作", report)
        self.assertIn("g1_execution_enabled=false", report)
        self.assertIn("hardware_execution_performed=false", report)
        self.assertIn("明确禁止官方 low-level/arm7 运动示例", report)


if __name__ == "__main__":
    unittest.main()
