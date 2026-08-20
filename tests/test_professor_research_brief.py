import html as html_lib
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_professor_research_brief import SOURCE, build
from generate_simulation_deployment_summary import build as build_dashboard


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "vercel_public" / "research-brief" / "index.html"


class ProfessorResearchBriefTests(unittest.TestCase):
    def test_source_has_professor_ready_research_structure(self):
        source = SOURCE.read_text()
        for heading in (
            "## 1. 研究目标与核心问题",
            "## 2. 核心研究问题",
            "## 3. 核心学术贡献",
            "## 4. 研究假设与主要评价指标",
            "## 5. 研究系统架构",
            "## 6. 当前已验证结果",
            "## 7. Adaptive Speed Module：已做到什么程度",
            "## 8. 当前主要阻塞项",
            "## 9. 下一阶段实验计划",
            "## 10. 真机阶段与安全边界",
            "## 12. 当前结论",
        ):
            self.assertIn(heading, source)
        self.assertIn("项目负责人：Shihua Yu", source)

    def test_render_is_print_ready_and_evidence_bound(self):
        report = build()
        plain = html_lib.unescape(re.sub(r"<[^>]+>", "", report))
        self.assertIn('<meta name="author" content="Shihua Yu">', report)
        self.assertEqual(report.count('<a href="#section-'), 12)
        self.assertEqual(report.count("<table>"), 3)
        self.assertIn("Print / PDF", plain)
        self.assertIn("@media print", report)
        for evidence in (
            "50/50 为有限 [32,16]",
            "P50 约 80.52 ms",
            "duration 改善 6.90%",
            "completed_cycles=0",
            "119.94 ms",
            "尚未证明",
            "g1_execution_enabled=false",
        ):
            self.assertIn(evidence, plain)

    def test_public_brief_contains_no_private_lab_identity(self):
        report = build()
        for forbidden in (
            "192.168.", "10.145.", "10.188.", "unitree@", "yixiao@", "user1@",
            "/Users/", "/home/", "unitree-g1-nx", "nnmc65", "shihua-vla",
        ):
            self.assertNotIn(forbidden, report)
        self.assertTrue(PUBLIC.exists())
        self.assertEqual(PUBLIC.read_text(), report)

    def test_dashboard_links_to_professor_brief(self):
        dashboard = build_dashboard()
        self.assertIn("教授版研究概要 ↗", dashboard)
        self.assertIn("/research-brief/", dashboard)


if __name__ == "__main__":
    unittest.main()
