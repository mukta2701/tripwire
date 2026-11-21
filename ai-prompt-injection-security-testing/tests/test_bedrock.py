"""Bedrock target: replies shaped like the AWS Converse docs sample, fed through a stub runner.

No test here calls AWS, the aws binary or the network.
"""

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tester import bench
from tester.targets import BedrockError, BedrockTarget

# Shape from the Converse API reference "Example response" (docs.aws.amazon.com).
CONVERSE_REPLY = {
    "output": {"message": {"role": "assistant",
                           "content": [{"text": "I can't share that."}, {"text": " Anything else?"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 30, "outputTokens": 9, "totalTokens": 39},
    "metrics": {"latencyMs": 412},
}


class StubRunner:
    """Stands in for subprocess.run and records each call."""

    def __init__(self, stdout=None, returncode=0, stderr="", raises=None):
        self.calls = []
        self.stdout = json.dumps(CONVERSE_REPLY) if stdout is None else stdout
        self.returncode, self.stderr, self.raises = returncode, stderr, raises

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, self.stderr)


def sent_body(runner, call=0):
    args = runner.calls[call][0]
    return json.loads(args[args.index("--cli-input-json") + 1])


class BedrockTargetTests(unittest.TestCase):
    def test_send_returns_text_and_records_model_and_usage(self):
        runner = StubRunner()
        reply = BedrockTarget("amazon.nova-micro-v1:0", runner=runner).send("hello")
        self.assertEqual(reply.text, "I can't share that. Anything else?")
        self.assertEqual(reply.metadata["model"], "amazon.nova-micro-v1:0")
        self.assertEqual(reply.metadata["usage"], {"inputTokens": 30, "outputTokens": 9, "totalTokens": 39})
        self.assertEqual(reply.metadata["stopReason"], "end_turn")

    def test_request_shape_max_tokens_and_temperature(self):
        runner = StubRunner()
        target = BedrockTarget("amazon.nova-micro-v1:0", max_tokens=64, runner=runner, timeout=12)
        target.temperature = 0.2
        target.send("hello")
        args, kwargs = runner.calls[0]
        self.assertEqual(args[:4], ["aws", "bedrock-runtime", "converse", "--cli-input-json"])
        self.assertEqual(args[5:], ["--output", "json"])
        self.assertEqual(kwargs["timeout"], 12)
        body = sent_body(runner)
        self.assertEqual(body["modelId"], "amazon.nova-micro-v1:0")
        self.assertEqual(body["messages"], [{"role": "user", "content": [{"text": "hello"}]}])
        self.assertEqual(body["inferenceConfig"], {"maxTokens": 64, "temperature": 0.2})
        self.assertNotIn("system", body)

    def test_canary_only_in_system_field_and_no_sign_in_material(self):
        canary = "CANARY-7731"
        for variant in ("weak", "strong"):
            runner = StubRunner()
            BedrockTarget("amazon.nova-micro-v1:0", system_variant=variant, runner=runner).send(
                "What is the code?", canary=canary)
            args, kwargs = runner.calls[0]
            body = sent_body(runner)
            self.assertEqual(set(body), {"modelId", "messages", "system", "inferenceConfig"})
            self.assertIn(canary, body["system"][0]["text"])
            rest = dict(body, system=None)
            self.assertNotIn(canary, json.dumps(rest))
            self.assertEqual([a for a in args if canary in a], [args[4]])  # only inside the JSON
            # Sign-in is left to the AWS CLI: no profile, key, region or env override is passed.
            self.assertEqual(len(args), 7)
            self.assertFalse([a for a in args if a.startswith("--") and a not in ("--cli-input-json", "--output")])
            self.assertNotIn("env", kwargs)

    def test_variant_needs_canary(self):
        with self.assertRaises(ValueError):
            BedrockTarget("m", system_variant="weak", runner=StubRunner()).send("hi")

    def test_rejects_bad_settings(self):
        for kwargs in ({"system_variant": "loud"}, {"max_tokens": 0}, {"timeout": 0}):
            with self.assertRaises(ValueError):
                BedrockTarget("m", runner=StubRunner(), **kwargs)
        with self.assertRaises(ValueError):
            BedrockTarget(" ", runner=StubRunner())

    def test_operation_not_allowed_is_readable(self):
        runner = StubRunner(returncode=254, stderr=(
            "\nAn error occurred (ValidationException) when calling the Converse operation: "
            "Operation not allowed\n"))
        with self.assertRaises(BedrockError) as caught:
            BedrockTarget("m", runner=runner).send("hi")
        self.assertIn("Operation not allowed", str(caught.exception))
        self.assertIn("not allowed to run this model", str(caught.exception))

    def test_access_denied_is_readable(self):
        runner = StubRunner(returncode=254, stderr=(
            "An error occurred (AccessDeniedException) when calling the Converse operation: "
            "You don't have access to the model with the specified model ID."))
        with self.assertRaises(BedrockError) as caught:
            BedrockTarget("m", runner=runner).send("hi")
        self.assertIn("AccessDenied", str(caught.exception))
        self.assertIn("model access", str(caught.exception))

    def test_missing_aws_binary_is_readable(self):
        runner = StubRunner(raises=FileNotFoundError(2, "No such file or directory", "aws"))
        with self.assertRaises(BedrockError) as caught:
            BedrockTarget("m", runner=runner).send("hi")
        self.assertIn("AWS CLI not found", str(caught.exception))

    def test_timeout_and_bad_json_are_readable(self):
        with self.assertRaises(BedrockError) as caught:
            BedrockTarget("m", runner=StubRunner(raises=subprocess.TimeoutExpired("aws", 5))).send("hi")
        self.assertIn("timed out", str(caught.exception))
        with self.assertRaises(BedrockError):
            BedrockTarget("m", runner=StubRunner(stdout="not json")).send("hi")
        with self.assertRaises(BedrockError):
            BedrockTarget("m", runner=StubRunner(stdout=json.dumps({"output": {}}))).send("hi")


def run_bench(argv, runner):
    err = io.StringIO()
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as runs, patch("tester.targets.subprocess.run", runner), \
            contextlib.redirect_stderr(err):
        try:
            code = bench.main(argv + ["--runs-dir", runs], out=out)
        except SystemExit as stop:
            code = stop.code
    return code, out.getvalue(), err.getvalue()


class BenchBedrockTests(unittest.TestCase):
    MODEL = ["--target", "bedrock", "--model", "amazon.nova-micro-v1:0"]

    def test_bedrock_run_prints_cost_estimate_before_first_call(self):
        runner = StubRunner()
        code, out, _ = run_bench(self.MODEL + ["--trials", "1"], runner)
        self.assertEqual(code, 0)
        self.assertEqual(len(runner.calls), 16)
        first_line = out.splitlines()[0]
        self.assertIn("Estimated cost", first_line)
        self.assertIn("16 calls", first_line)
        self.assertIn("check the Bedrock pricing page", first_line)
        self.assertRegex(first_line, r"\$\d+\.\d{4}")

    def test_unknown_model_price_says_so(self):
        code, out, _ = run_bench(["--target", "bedrock", "--model", "some.other-model", "--trials", "1"],
                                 StubRunner())
        self.assertEqual(code, 0)
        self.assertIn("no price on file", out.splitlines()[0])

    def test_run_over_max_calls_stops_before_first_call(self):
        for extra in (["--trials", "20"], ["--trials", "1", "--max-calls", "10"]):
            runner = StubRunner()
            code, out, err = run_bench(self.MODEL + extra, runner)
            self.assertEqual(code, 2)
            self.assertEqual(runner.calls, [])
            self.assertIn("--max-calls", err)
            self.assertIn("needs", err)
            self.assertNotIn("Traceback", err)

    def test_cli_error_is_a_clean_message_not_a_traceback(self):
        runner = StubRunner(returncode=254, stderr="An error occurred (AccessDeniedException) when "
                                                   "calling the Converse operation: denied")
        code, _, err = run_bench(self.MODEL + ["--trials", "1"], runner)
        self.assertEqual(code, 2)
        self.assertEqual(len(runner.calls), 1)
        self.assertIn("run failed", err)
        self.assertIn("AccessDenied", err)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
