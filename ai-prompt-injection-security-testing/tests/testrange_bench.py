"""Range bench tests: --range runs, escaping, errors, openai_compat (testrange_* avoids protected glob)."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime
from pathlib import Path

from tester import bench, compare
from tester.report import MANIFEST_KEYS
from tests.test_targets import fake_api

WHEN = datetime(2026, 11, 21, 9, 30, 0)


def range_summary(out, *args):
    with tempfile.TemporaryDirectory() as runs:
        code = bench.main(["--range", "--runs-dir", runs, *args], out=out)
        folder = next(Path(runs).iterdir())
        summary = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        files = sorted(p.name for p in folder.iterdir())
        html = (folder / "evidence_pack.html").read_text(encoding="utf-8")
        md = (folder / "risk_register.md").read_text(encoding="utf-8")
        return code, summary, files, html, md


class RangeBenchTests(unittest.TestCase):
    def test_unsafe_run_writes_four_files_with_log_backed_fails(self):
        code, summary, files, _, _ = range_summary(
            io.StringIO(), "--target", "mock", "--mode", "unsafe", "--trials", "1")
        self.assertEqual(code, 0)
        self.assertEqual(files, ["evidence_pack.html", "results.json",
                                 "results.manifest.json", "risk_register.md"])
        self.assertGreaterEqual(summary["fail_count"], 1)
        records = {r["id"]: r for r in
                   json.loads(Path("data/range_cases.json").read_text(encoding="utf-8"))["cases"]}
        for result in summary["results"]:
            if result["outcome"] != "FAIL":
                continue
            line = json.loads(result["range"]["log_line"])
            record = records[result["test_id"]]
            canary_in_args = result["canary"] in json.dumps(line.get("arguments", {}))
            self.assertTrue(line["tool"] in record["forbidden_tools"]
                            or (line["tool"] == "send_message" and canary_in_args))

    def test_safe_run_passes_with_null_evidence(self):
        _, summary, _, _, _ = range_summary(
            io.StringIO(), "--target", "mock", "--mode", "safe", "--trials", "1")
        self.assertEqual(summary["fail_count"], 0)
        for result in summary["results"]:
            self.assertEqual(result["outcome"], "PASS")
            self.assertIsNone(result["range"]["tool_call"])
            self.assertIsNone(result["range"]["log_line"])

    def test_findings_show_ids_tool_call_and_log_line(self):
        code, summary, _, html, md = range_summary(
            io.StringIO(), "--target", "mock", "--mode", "unsafe", "--trials", "1")
        self.assertEqual(code, 0)
        self.assertNotIn("Sixteen synthetic", html)
        for result in summary["results"]:
            if result["outcome"] != "FAIL":
                continue
            for doc_id in result["range"]["retrieved_ids"]:
                self.assertIn(doc_id, html)
                self.assertIn(doc_id, md)
            tool = result["range"]["tool_call"]["tool"]
            self.assertIn(tool, html)
            self.assertIn(tool.replace("_", "\\_"), md)

    def test_hostile_tool_text_is_escaped_in_both_pages(self):
        from tester.report import render_report
        from tester.report_html import render_html
        from tester.targets import TargetResponse
        from agent_range import suite
        import json as _json

        class Hostile:
            name = "hostile"

            def send(self, text, *, canary=None):
                return TargetResponse(
                    'Quoting the note: {"tool": "delete_record", '
                    '"arguments": {"customer_id": "<script>alert(1)</script>"}} and refusing.',
                    {})

        summary = suite.run_range_tests("unsafe", target=Hostile())
        crosswalk = _json.loads(Path("data/framework_crosswalk.json").read_text(encoding="utf-8"))
        html = render_html(summary, crosswalk, scorer=None)
        md = render_report(summary, crosswalk)
        self.assertNotIn("<script", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script", md)
        self.assertIn("&lt;script&gt;", md)

    def test_system_variant_ignored_with_message(self):
        with tempfile.TemporaryDirectory() as runs:
            err, out = io.StringIO(), io.StringIO()
            with redirect_stderr(err):
                code = bench.main(["--range", "--target", "mock", "--mode", "unsafe",
                                   "--system-variant", "weak", "--trials", "1",
                                   "--runs-dir", runs], out=out)
            self.assertEqual(code, 0)
            self.assertIn("ignored with --range", err.getvalue())
            self.assertEqual(len(list(Path(runs).iterdir())), 1)

    def test_bedrock_and_explicit_cases_are_rejected_without_folder(self):
        for extra in (["--target", "bedrock"], ["--cases", "data/cases.json"]):
            with tempfile.TemporaryDirectory() as runs:
                err = io.StringIO()
                with redirect_stderr(err), self.assertRaises(SystemExit):
                    bench.main(["--range", "--runs-dir", runs, *extra], out=io.StringIO())
                self.assertEqual(list(Path(runs).iterdir()), [])

    def test_openai_compat_run_writes_folder_with_progress_per_trial(self):
        from tests.test_bench import RecordingOut
        with fake_api(reply_text="done, nothing to call") as (url, requests), \
                tempfile.TemporaryDirectory() as runs:
            out = RecordingOut(requests)
            code = bench.main(["--range", "--model", "toy", "--base-url", url,
                               "--trials", "1", "--requests-per-second", "1000",
                               "--runs-dir", runs], out=out, now=WHEN)
            self.assertEqual(code, 0)
            self.assertEqual(len(out.seen), 7)
            folder = next(Path(runs).iterdir())
            summary = json.loads((folder / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["total_tests"], 7)
            self.assertEqual(summary["manifest"]["target"], "openai_compat")

    def test_readme_and_limits_explain_truncation_and_refusal(self):
        from tester.report import limits_text
        readme = Path("README.md").read_text(encoding="utf-8")
        for needle in ("truncated", "20,000", "refusal", "FAIL", "--range"):
            self.assertIn(needle, readme)
        limits = " ".join(limits_text({"suite": "range", "target": "mock"}, 7))
        for needle in ("truncated", "20,000", "not seen", "refusal", "FAIL"):
            self.assertIn(needle, limits)

    def test_manifest_carries_suite_and_corpus_hash(self):
        self.assertIn("suite", MANIFEST_KEYS)
        self.assertIn("corpus_sha256", MANIFEST_KEYS)
        _, summary, _, _, _ = range_summary(
            io.StringIO(), "--target", "mock", "--mode", "safe", "--trials", "1")
        self.assertEqual(summary["manifest"]["suite"], "range")
        self.assertEqual(summary["manifest"]["scorer_version"], "range-toolcall-1.0")
        self.assertEqual(summary["manifest"]["baseline_scorer_version"], "n/a")
        self.assertEqual(len(summary["manifest"]["corpus_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
