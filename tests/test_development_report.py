import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_report import render_updates


ROOT = Path(__file__).resolve().parents[1]
UPDATES = ROOT / "results" / "development_updates.json"
REPORT = ROOT / "validation_report.html"
REQUIRED_FIELDS = {
    "time", "title", "status", "summary", "done", "conclusions",
    "fixes", "remaining", "next", "evidence",
}
LIST_FIELDS = {
    "done", "conclusions", "fixes", "remaining", "next", "evidence",
}


class DevelopmentReportTests(unittest.TestCase):
    def setUp(self):
        self.updates = json.loads(UPDATES.read_text())

    def test_every_update_has_complete_audit_sections(self):
        self.assertTrue(self.updates)
        for update in self.updates:
            self.assertFalse(REQUIRED_FIELDS - set(update))
            self.assertIn(update["status"], {"pass", "partial", "blocked", "todo"})
            for field in LIST_FIELDS:
                self.assertIsInstance(update[field], list)
                self.assertTrue(update[field], f"{update['title']} has empty {field}")

    def test_update_renderer_includes_required_chinese_headings(self):
        rendered = render_updates(self.updates)
        for heading in ("本次完成", "结论", "修复", "还差什么", "下一步", "证据"):
            self.assertIn(heading, rendered)

    def test_checked_in_html_contains_latest_update_and_closed_hardware_gate(self):
        html = REPORT.read_text()
        self.assertIn(self.updates[-1]["title"], html)
        self.assertIn("g1_execution_enabled=false", html)
        self.assertIn("真机执行关闭", html)


if __name__ == "__main__":
    unittest.main()
