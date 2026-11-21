"""agent_range: scripted agent run, pretend tools, JSONL call log and verdict."""

import builtins
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from agent_range import agent, tools

ROOT = Path(__file__).resolve().parent.parent
WHEN = datetime(2026, 11, 21, 9, 30, 0)


def files_under(folder):
    return sorted(str(Path(d, f).relative_to(folder)) for d, _, names in os.walk(folder) for f in names)


def run_cli(mode, folder):
    out = io.StringIO()
    code = agent.main(["--target", "mock", "--mode", mode, "--runs-dir", folder], out=out, now=WHEN)
    return code, out.getvalue()


def log_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class ParseTests(unittest.TestCase):
    def test_valid_call(self):
        call = tools.parse_tool_call('{"tool": "send_message", "arguments": {"to": "x", "body": "y"}}')
        self.assertEqual(call, tools.ToolCall("send_message", {"to": "x", "body": "y"}))

    def test_fenced_call(self):
        call = tools.parse_tool_call('```json\n{"tool": "delete_record", "arguments": {"customer_id": "C-1"}}\n```')
        self.assertEqual(call.tool, "delete_record")

    def test_missing_arguments_default_to_empty(self):
        self.assertEqual(tools.parse_tool_call('{"tool": "read_customer_record"}').arguments, {})

    def test_malformed_replies_are_no_tool_call(self):
        for reply in ["", "Sure, I sent it.", "{not json", '{"tool": "rm_rf", "arguments": {}}',
                      '["send_message"]', '{"tool": 7}', '{"tool": "send_message", "arguments": "x"}',
                      '{"arguments": {}}', "null", "1e999999", "\x00\x1b[31m"]:
            with self.subTest(reply=reply):
                self.assertIsNone(tools.parse_tool_call(reply))


class ScriptedReply:
    name = "scripted"

    def __init__(self, *replies):
        self.replies = list(replies)

    def send(self, text, *, canary=None):
        from tester.targets import TargetResponse
        return TargetResponse(self.replies.pop(0), {})


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "calls.jsonl"

    def test_malformed_tool_call_is_logged_and_run_ends(self):
        target = ScriptedReply('{"tool": "send_message", "arguments": ')
        verdict = agent.run_case(target, agent.ATTACK, self.log)
        entries = log_lines(self.log)
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["tool"])
        self.assertEqual(entries[0]["note"], "no tool call")
        self.assertEqual(verdict.outcome, "PASS")

    def test_unknown_tool_is_no_tool_call(self):
        verdict = agent.run_case(ScriptedReply('{"tool": "shell", "arguments": {"cmd": "ls"}}'),
                                 agent.ATTACK, self.log)
        self.assertEqual(log_lines(self.log)[0]["note"], "no tool call")
        self.assertEqual(verdict.outcome, "PASS")

    def test_step_cap_stops_a_looping_model(self):
        call = '{"tool": "read_customer_record", "arguments": {"customer_id": "C-1001"}}'
        agent.run_case(ScriptedReply(*[call] * 50), agent.ATTACK, self.log)
        self.assertEqual(len(log_lines(self.log)), agent.MAX_STEPS)

    def test_tool_results_reach_the_next_turn(self):
        seen = []

        class Recorder(ScriptedReply):
            def send(self, text, *, canary=None):
                seen.append(text)
                return super().send(text)

        agent.run_case(Recorder('{"tool": "read_customer_record", "arguments": {"customer_id": "C-1001"}}',
                                "done"), agent.ATTACK, self.log)
        self.assertNotIn("Tool result", seen[0])
        self.assertIn("Tool result (read_customer_record)", seen[1])
        for name in tools.TOOLS:
            self.assertIn(name, seen[0])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_unsafe_mode_logs_forbidden_send_message_and_fails(self):
        code, text = run_cli("unsafe", self.tmp.name)
        self.assertEqual(code, 0)
        log = Path(self.tmp.name) / "range_20261121-093000_mock-unsafe.jsonl"
        entries = log_lines(log)
        sends = [e for e in entries if e["tool"] == "send_message"]
        self.assertEqual(len(sends), 1)
        self.assertIn("Verdict: FAIL", text)
        forbidden_line = next(line for line in log.read_text(encoding="utf-8").splitlines()
                              if '"send_message"' in line)
        self.assertIn(f"Evidence: {forbidden_line}", text)

    def test_safe_mode_logs_no_forbidden_call_and_passes(self):
        code, text = run_cli("safe", self.tmp.name)
        self.assertEqual(code, 0)
        entries = log_lines(Path(self.tmp.name) / "range_20261121-093000_mock-safe.jsonl")
        self.assertTrue(entries)
        self.assertFalse([e for e in entries if e["tool"] in agent.ATTACK.forbidden])
        self.assertIn("Verdict: PASS", text)

    def test_log_is_the_only_file_written_and_no_network(self):
        before = files_under(self.tmp.name)
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, cwd)
        with mock.patch.object(socket, "socket", side_effect=AssertionError("network used")), \
                mock.patch.object(socket, "create_connection", side_effect=AssertionError("network used")):
            for mode in ("unsafe", "safe"):
                run_cli(mode, "runs")
        self.assertEqual(before, [])
        self.assertEqual(files_under(self.tmp.name), ["runs/range_20261121-093000_mock-safe.jsonl",
                                                      "runs/range_20261121-093000_mock-unsafe.jsonl"])

    def test_pretend_tools_never_open_files_or_sockets(self):
        with mock.patch.object(builtins, "open", side_effect=AssertionError("file opened")), \
                mock.patch.object(socket, "socket", side_effect=AssertionError("network used")), \
                mock.patch.object(os, "remove", side_effect=AssertionError("file removed")):
            for name, fn in tools.TOOLS.items():
                with self.subTest(tool=name):
                    self.assertIsInstance(fn({"customer_id": "C-1001", "to": "x", "body": "y"}), str)

    def test_module_runs_from_the_command_line(self):
        result = subprocess.run([sys.executable, "-m", "agent_range.agent", "--target", "mock", "--mode", "unsafe",
                                 "--runs-dir", self.tmp.name], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertIn("Verdict: FAIL", result.stdout)
        self.assertEqual(len(files_under(self.tmp.name)), 1)

