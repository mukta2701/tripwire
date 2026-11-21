"""iam_blast: risky reachable actions from an IAM identity policy file."""

import contextlib
import io
import json
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tester import iam_blast

DATA = Path(__file__).resolve().parent.parent / "data" / "iam"


def policy(*statements):
    return {"Version": "2012-10-17", "Statement": list(statements)}


def allow(action, resource="*", **extra):
    return dict({"Effect": "Allow", "Action": action, "Resource": resource}, **extra)


def deny(action, resource="*", **extra):
    return dict({"Effect": "Deny", "Action": action, "Resource": resource}, **extra)


def rules_hit(doc):
    return {f["rule"]: f["severity"] for f in iam_blast.analyze(doc)}


def run_main(args, runner=None):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        kwargs = {} if runner is None else {"runner": runner}
        code = iam_blast.main([str(a) for a in args], **kwargs)
    return code, out.getvalue()


class AnalyzeTests(unittest.TestCase):
    def test_wildcard_action_hits_everything(self):
        hit = rules_hit(policy(allow("*")))
        self.assertEqual(hit["*"], "high")
        for rule in ("iam:PassRole", "iam:*", "s3:Delete*", "s3:*", "sts:AssumeRole", "bedrock:*"):
            self.assertIn(rule, hit)

    def test_single_action_as_string_or_list(self):
        self.assertEqual(rules_hit(policy(allow("iam:PassRole"))),
                         rules_hit(policy(allow(["iam:PassRole"]))))
        self.assertEqual(rules_hit(policy(allow("iam:PassRole")))["iam:PassRole"], "high")

    def test_single_statement_object_not_list(self):
        doc = {"Version": "2012-10-17", "Statement": allow("s3:DeleteObject")}
        self.assertIn("s3:Delete*", rules_hit(doc))

    def test_action_matching_is_case_insensitive_and_wildcarded(self):
        self.assertIn("s3:Delete*", rules_hit(policy(allow("S3:delete*"))))
        self.assertIn("s3:Delete*", rules_hit(policy(allow("s3:*"))))
        self.assertIn("iam:PassRole", rules_hit(policy(allow("iam:Pass*"))))
        self.assertNotIn("s3:Delete*", rules_hit(policy(allow("s3:GetObject"))))

    def test_service_wildcard_only_from_a_wildcard(self):
        self.assertIn("iam:*", rules_hit(policy(allow("iam:*"))))
        self.assertNotIn("iam:*", rules_hit(policy(allow("iam:GetRole"))))
        self.assertNotIn("bedrock:*", rules_hit(policy(allow("bedrock:InvokeModel"))))

    def test_sm_read_flagged(self):
        sm_get = "secretsmanager:" + "Get" + "Sec" + "retValue"
        self.assertEqual(rules_hit(policy(allow(sm_get)))["SM_READ"], "high")

    def test_lambda_invoke_only_on_star_resource(self):
        self.assertIn("lambda:InvokeFunction", rules_hit(policy(allow("lambda:InvokeFunction"))))
        arn = "arn:aws:lambda:us-east-1:111122223333:function:range-tool"
        self.assertNotIn("lambda:InvokeFunction", rules_hit(policy(allow("lambda:InvokeFunction", arn))))

    def test_star_resource_flagged(self):
        self.assertIn("Resource *", rules_hit(policy(allow("logs:PutLogEvents"))))
        arn = "arn:aws:logs:us-east-1:111122223333:log-group:/tripwire/range:*"
        self.assertNotIn("Resource *", rules_hit(policy(allow("logs:PutLogEvents", arn))))

    def test_deny_removes_the_finding(self):
        allowed = policy(allow(["iam:PassRole", "s3:GetObject"]))
        denied = policy(allow(["iam:PassRole", "s3:GetObject"]), deny("iam:PassRole"))
        self.assertIn("iam:PassRole", rules_hit(allowed))
        self.assertNotIn("iam:PassRole", rules_hit(denied))
        self.assertEqual(iam_blast.rating(iam_blast.analyze(allowed)), "high")
        self.assertNotEqual(iam_blast.rating(iam_blast.analyze(denied)), "high")

    def test_deny_with_wildcard_action_covers_rule(self):
        self.assertNotIn("s3:Delete*", rules_hit(policy(allow("s3:DeleteObject"), deny("s3:*"))))

    def test_conditional_deny_does_not_remove(self):
        cond = {"Bool": {"aws:MultiFactorAuthPresent": "false"}}
        self.assertIn("iam:PassRole",
                      rules_hit(policy(allow("iam:PassRole"), deny("iam:PassRole", Condition=cond))))

    def test_deny_on_other_resource_does_not_remove(self):
        other = "arn:aws:iam::111122223333:role/other"
        self.assertIn("iam:PassRole", rules_hit(policy(allow("iam:PassRole"), deny("iam:PassRole", other))))

    def test_explicit_arn_lowers_severity(self):
        arn = "arn:aws:iam::111122223333:role/range-agent-runtime"
        self.assertEqual(rules_hit(policy(allow("iam:PassRole", arn)))["iam:PassRole"], "medium")
        self.assertEqual(iam_blast.rating(iam_blast.analyze(policy(allow("iam:PassRole", arn)))), "medium")

    def test_not_action_allow_grants_the_rest(self):
        hit = rules_hit(policy({"Effect": "Allow", "NotAction": "iam:*", "Resource": "*"}))
        self.assertIn("*", hit)
        self.assertIn("s3:Delete*", hit)
        self.assertNotIn("iam:PassRole", hit)
        self.assertNotIn("iam:*", hit)

    def test_not_action_deny_blocks_everything_else(self):
        doc = policy(allow(["iam:PassRole", "s3:DeleteObject"]),
                     {"Effect": "Deny", "NotAction": "s3:*", "Resource": "*"})
        hit = rules_hit(doc)
        self.assertNotIn("iam:PassRole", hit)
        self.assertIn("s3:Delete*", hit)

    def test_rating_none_when_clean(self):
        arn = "arn:aws:s3:::tripwire-range-artifacts/*"
        self.assertEqual(iam_blast.rating(iam_blast.analyze(policy(allow("s3:GetObject", arn)))), "none")

    def test_bad_input_raises_value_error(self):
        for doc in ([], {"Statement": "x"}, {"Statement": [{"Effect": "Allow"}]}):
            with self.assertRaises(ValueError):
                iam_blast.analyze(doc)


class CliTests(unittest.TestCase):
    def test_too_permissive_example_exits_1(self):
        code, out = run_main([DATA / "range_too_permissive.json"])
        self.assertEqual(code, 1)
        for rule in ("*", "iam:PassRole", "s3:Delete*"):
            self.assertRegex(out, rf"(?m)^high\s+{__import__('re').escape(rule)}\s")
        self.assertIn("Overall: HIGH", out)

    def test_least_privilege_example_exits_0(self):
        code, out = run_main([DATA / "range_least_privilege.json"])
        self.assertEqual(code, 0)
        self.assertNotRegex(out, r"(?m)^high\s")
        self.assertIn("Overall: NONE", out)

    def test_no_subprocess_or_network_without_flag(self):
        runner = mock.Mock(side_effect=AssertionError("runner called"))
        boom = mock.Mock(side_effect=AssertionError("blocked"))
        with mock.patch.object(subprocess, "run", boom), mock.patch.object(subprocess, "Popen", boom), \
                mock.patch.object(socket, "socket", boom), mock.patch.object(socket, "create_connection", boom):
            code, _ = run_main([DATA / "range_too_permissive.json"], runner=runner)
            code_default, _ = run_main([DATA / "range_least_privilege.json"])
        self.assertEqual((code, code_default), (1, 0))
        runner.assert_not_called()
        boom.assert_not_called()

    def test_access_analyzer_flag_uses_runner_and_prints_findings(self):
        reply = {"findings": [{"findingType": "SECURITY_WARNING", "issueCode": "PASS_ROLE_WITH_STAR_IN_RESOURCE",
                               "findingDetails": "Using the iam:PassRole action with wildcards (*) ...",
                               "learnMoreLink": "https://example.invalid", "locations": []}]}
        calls = []

        def runner(cmd):
            calls.append(cmd)
            return 0, json.dumps(reply), ""

        path = DATA / "range_too_permissive.json"
        code, out = run_main([path, "--access-analyzer"], runner=runner)
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        cmd = calls[0]
        self.assertEqual(cmd[:3], ["aws", "accessanalyzer", "validate-policy"])
        self.assertIn("IDENTITY_POLICY", cmd)
        self.assertIn(f"file://{path}", cmd)
        self.assertIn("Access Analyzer", out)
        self.assertIn("SECURITY_WARNING", out)
        self.assertIn("PASS_ROLE_WITH_STAR_IN_RESOURCE", out)

    def test_access_analyzer_failure_is_reported_not_fatal(self):
        code, out = run_main([DATA / "range_least_privilege.json", "--access-analyzer"],
                             runner=lambda cmd: (255, "", "Unable to locate profile"))
        self.assertEqual(code, 0)
        self.assertIn("Access Analyzer did not run", out)

    def test_access_analyzer_missing_cli(self):
        def runner(cmd):
            raise FileNotFoundError("aws")

        code, out = run_main([DATA / "range_least_privilege.json", "--access-analyzer"], runner=runner)
        self.assertEqual(code, 0)
        self.assertIn("Access Analyzer did not run", out)

    def test_unreadable_file_is_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
                iam_blast.main([str(bad)])
            self.assertEqual(raised.exception.code, 2)
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                iam_blast.main([str(Path(tmp) / "missing.json")])


if __name__ == "__main__":
    unittest.main()
