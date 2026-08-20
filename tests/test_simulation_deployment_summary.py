import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_simulation_deployment_summary import build


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "simulation_deployment_summary.html"


class SimulationDeploymentSummaryTests(unittest.TestCase):
    def test_render_contains_current_evidence_metrics(self):
        rendered = build()
        tests = json.loads((ROOT / "results" / "test_summary.json").read_text())
        for expected in (
            "50/50",
            "absolute_xyzw_lr",
            "119.94 ms",
            f"{tests['passed']}/{tests['total']}",
            "0 cycles",
            "15 组候选",
        ):
            self.assertIn(expected, rendered)

    def test_render_explains_each_required_dimension(self):
        rendered = build()
        for expected in (
            "Simulation Gate 图",
            "真实机器人闭环图",
            "做了什么",
            "这意味着什么",
            "还差什么",
            "现在可以做",
            "现在禁止做",
            "达到真机闭环的最短安全路径",
        ):
            self.assertIn(expected, rendered)

    def test_checked_report_keeps_hardware_gates_closed(self):
        html = REPORT.read_text()
        self.assertIn("不允许真机动作", html)
        self.assertIn("g1_execution_enabled</span><b>false", html)
        self.assertIn("hardware_execution_performed</span><b>false", html)
        self.assertNotIn("ChannelPublisher、LowCmd、arm_sdk", html.split("现在可以做", 1)[1].split("现在禁止做", 1)[0])


if __name__ == "__main__":
    unittest.main()
