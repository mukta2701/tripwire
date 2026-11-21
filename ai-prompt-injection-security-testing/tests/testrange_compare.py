"""Range compare tests: Range folders load and render with the Range note."""

import json
import tempfile
import unittest
from pathlib import Path

from tester import compare
from tester.test_runner import run_tests
from agent_range import suite

RANGE_NOTE = "For Range columns, FAIL means a forbidden tool call"


def write_summary(root, name, summary):
    folder = Path(root) / name
    folder.mkdir()
    (folder / "results.json").write_text(json.dumps(summary), encoding="utf-8")
    (folder / "results.manifest.json").write_text(
        json.dumps(summary["manifest"]), encoding="utf-8")
    return folder


class RangeCompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_two_range_runs_render_one_row_per_case_with_note(self):
        safe = write_summary(self.tmp.name, "safe",
                             suite.run_range_tests("safe", trials=2))
        unsafe = write_summary(self.tmp.name, "unsafe",
                               suite.run_range_tests("unsafe", trials=2))
        self.assertEqual(compare.main([str(safe), str(unsafe),
                                       "--output", str(Path(self.tmp.name) / "c.html")]), 0)
        html = compare.render_compare(
            [compare.load_run(safe), compare.load_run(unsafe)], now="T0")
        self.assertEqual(html.count('<tr class="case">'), 7)
        self.assertIn(RANGE_NOTE, html)
        self.assertIn("send_message", html)

    def test_mixed_range_and_default_runs_load(self):
        range_folder = write_summary(self.tmp.name, "r",
                                     suite.run_range_tests("unsafe", trials=1))
        default_folder = write_summary(self.tmp.name, "d", run_tests(mode="unsafe", trials=1))
        runs = [compare.load_run(range_folder), compare.load_run(default_folder)]
        self.assertTrue(runs[0]["is_range"])
        self.assertFalse(runs[1]["is_range"])
        self.assertIn(RANGE_NOTE, compare.render_compare(runs, now="T0"))

    def test_runs_without_suite_have_no_range_note(self):
        one = write_summary(self.tmp.name, "a", run_tests(mode="safe", trials=1))
        two = write_summary(self.tmp.name, "b", run_tests(mode="unsafe", trials=1))
        html = compare.render_compare(
            [compare.load_run(one), compare.load_run(two)], now="T0")
        self.assertNotIn(RANGE_NOTE, html)


if __name__ == "__main__":
    unittest.main()
