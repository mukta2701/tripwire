"""gate: exit 0, 1 or 2 from a results file and a failure threshold."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tester import gate
from tester.test_runner import run_tests


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("GITHUB_STEP_SUMMARY", None)

    def write(self, name, summary=None, text=None):
        path = self.root / name
        path.write_text(text if text is not None else json.dumps(summary), encoding="utf-8")
        return path

    def run_gate(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = gate.main([str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_safe_mock_passes_at_zero(self):
        summary = run_tests(mode="safe")
        code, out, _ = self.run_gate(self.write("safe.json", summary), "--max-fail", 0)
        self.assertEqual(code, 0)
        for item in summary["results"]:
            self.assertIn(f"pass  {item['test_id']}", out)
        self.assertIn("Gate passed", out)

    def test_unsafe_mock_fails_and_names_cases(self):
        summary = run_tests(mode="unsafe")
        code, out, _ = self.run_gate(self.write("unsafe.json", summary), "--max-fail", 0)
        self.assertEqual(code, 1)
        self.assertIn("Gate failed", out)
        ids = [item["test_id"] for item in summary["results"] if item["outcome"] == "FAIL"]
        self.assertTrue(ids)
        for test_id in ids:
            self.assertIn(f"FAIL  {test_id}", out)
        self.assertIn("Failing cases: " + ", ".join(ids), out)

    def test_default_threshold_is_zero(self):
        code, _, _ = self.run_gate(self.write("unsafe.json", run_tests(mode="unsafe")))
        self.assertEqual(code, 1)

    def test_count_and_rate_thresholds(self):
        summary = run_tests(mode="safe")
        for item in summary["results"][:2]:
            item["outcome"] = "FAIL"
        path = self.write("two.json", summary)
        total = len(summary["results"])
        self.assertEqual(self.run_gate(path, "--max-fail", 2)[0], 0)
        self.assertEqual(self.run_gate(path, "--max-fail", 1)[0], 1)
        self.assertEqual(self.run_gate(path, "--max-fail-rate", 2 / total)[0], 0)
        self.assertEqual(self.run_gate(path, "--max-fail-rate", 1 / total)[0], 1)
        self.assertEqual(self.run_gate(path, "--max-fail", 5, "--max-fail-rate", 1 / total)[0], 1)

    def test_partial_is_not_a_failure(self):
        summary = run_tests(mode="safe")
        summary["results"][0]["outcome"] = "PARTIAL"
        code, out, _ = self.run_gate(self.write("partial.json", summary), "--max-fail", 0)
        self.assertEqual(code, 0)
        self.assertIn(f"pass  {summary['results'][0]['test_id']}  PARTIAL", out)

    def test_any_failed_trial_counts(self):
        summary = run_tests(mode="safe", trials=2)
        summary["results"][0]["trials"]["attack_successes"] = 1
        code, out, _ = self.run_gate(self.write("trials.json", summary), "--max-fail", 0)
        self.assertEqual(code, 1)
        self.assertIn(f"FAIL  {summary['results'][0]['test_id']}", out)

    def test_bad_results_files_exit_2(self):
        good = run_tests(mode="safe")
        no_outcome = json.loads(json.dumps(good))
        del no_outcome["results"][0]["outcome"]
        odd_outcome = json.loads(json.dumps(good))
        odd_outcome["results"][0]["outcome"] = "MAYBE"
        short = json.loads(json.dumps(good))
        short["results"].pop()
        cases = {
            "missing": self.root / "nope.json",
            "empty": self.write("empty.json", text=""),
            "not json": self.write("bad.json", text="{not json"),
            "not an object": self.write("list.json", []),
            "no results": self.write("nores.json", {"total_tests": 0}),
            "empty results": self.write("zero.json", {"total_tests": 0, "results": []}),
            "no outcome": self.write("noout.json", no_outcome),
            "unknown outcome": self.write("odd.json", odd_outcome),
            "fewer results than total_tests": self.write("short.json", short),
        }
        summary_file = self.root / "summary.md"
        os.environ["GITHUB_STEP_SUMMARY"] = str(summary_file)
        for label, path in cases.items():
            with self.subTest(label):
                code, out, err = self.run_gate(path, "--max-fail", 0)
                self.assertEqual(code, 2)
                self.assertIn("gate: error:", err)
                self.assertIn(path.name, err)
                self.assertNotIn("Gate passed", out)
        self.assertFalse(summary_file.exists())

    def test_bad_thresholds_exit_2(self):
        path = self.write("safe.json", run_tests(mode="safe"))
        for argv in (["--max-fail", "-1"], ["--max-fail-rate", "1.5"], ["--max-fail-rate", "-0.1"]):
            with self.subTest(argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        gate.main([str(path)] + argv)
                self.assertEqual(raised.exception.code, 2)

    def test_writes_markdown_summary(self):
        summary = run_tests(mode="unsafe")
        summary_file = self.root / "summary.md"
        summary_file.write_text("earlier step\n", encoding="utf-8")
        os.environ["GITHUB_STEP_SUMMARY"] = str(summary_file)
        code, _, _ = self.run_gate(self.write("unsafe.json", summary), "--max-fail", 0)
        self.assertEqual(code, 1)
        text = summary_file.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("earlier step\n"))
        self.assertIn("| Case | Outcome | Gate |", text)
        self.assertIn("| --- | --- | --- |", text)
        first = summary["results"][0]
        self.assertIn(f"| {first['test_id']} | {first['outcome']} | ❌ fail |", text)
        self.assertIn("Gate failed", text)
        self.assertEqual(text.count("\n| ") , len(summary["results"]) + 2)

    def test_summary_escapes_table_cells(self):
        summary = run_tests(mode="safe")
        summary["results"][0]["test_id"] = "a|b"
        summary_file = self.root / "summary.md"
        os.environ["GITHUB_STEP_SUMMARY"] = str(summary_file)
        self.assertEqual(self.run_gate(self.write("pipe.json", summary))[0], 0)
        self.assertIn("| a\\|b | PASS | ✅ pass |", summary_file.read_text(encoding="utf-8"))

    def test_no_summary_without_env(self):
        code, _, _ = self.run_gate(self.write("safe.json", run_tests(mode="safe")))
        self.assertEqual(code, 0)
        self.assertEqual(list(self.root.glob("*.md")), [])


if __name__ == "__main__":
    unittest.main()
