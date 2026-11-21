"""Blast radius from an IAM identity policy file: which risky actions could a hijacked agent reach?

Usage: python3 -m tester.iam_blast data/iam/range_too_permissive.json [--access-analyzer]

Reads a local policy JSON file only. Exit code 1 when any high finding is reachable, 0 otherwise.
Conditions on Allow statements are ignored (counted as reachable); a Deny only removes a finding
when it has no Condition and covers the Allow statement's resources.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SEVERITIES = ("high", "medium", "low")
SM = "secrets" + "manager"

# (pattern shown, probe actions, severity on Resource "*", why, only when Resource is "*")
RULES = [
    ("*", ["*"], "high", "every action in every service", False),
    ("iam:*", ["iam:*"], "high", "full IAM control: create users, attach admin policies", False),
    ("iam:PassRole", ["iam:PassRole"], "high", "hand a stronger role to a service it starts", False),
    ("sts:AssumeRole", ["sts:AssumeRole"], "high", "switch into another role", False),
    ("SM_READ", [f"{SM}:GetSecretValue", f"{SM}:BatchGetSecretValue"], "high",
     "read stored Secrets Manager values", False),
    ("s3:*", ["s3:*"], "high", "every S3 action, including bucket policy changes", False),
    ("s3:Delete*", ["s3:Delete*", "s3:DeleteObject", "s3:DeleteObjectVersion", "s3:DeleteBucket"], "high",
     "delete buckets and objects", False),
    ("bedrock:*", ["bedrock:*"], "medium", "every Bedrock action, including guardrail and model changes", False),
    ("lambda:InvokeFunction", ["lambda:InvokeFunction"], "medium", "run any Lambda function", True),
]
STAR_RESOURCE = ("Resource *", "medium", "the statement's actions apply to every resource")

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f‪-‮⁦-⁩]")


def clean(value, limit=160) -> str:
    text = _CONTROL.sub(" ", str(value))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _glob(pattern: str, value: str) -> bool:
    """IAM wildcard match: * and ?, case-insensitive. Probes with a literal * match only wider wildcards."""
    regex = re.escape(pattern.lower()).replace(r"\*", ".*").replace(r"\?", ".")
    return re.fullmatch(regex, value.lower()) is not None


def _as_list(value, what: str, where: str) -> list:
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not items or not all(isinstance(v, str) for v in items):
        raise ValueError(f"{where}: {what} must be a string or a list of strings")
    return items


def _statements(doc) -> list:
    if not isinstance(doc, dict) or "Statement" not in doc:
        raise ValueError("policy must be a JSON object with a Statement")
    raw = doc["Statement"]
    raw = [raw] if isinstance(raw, dict) else raw
    if not isinstance(raw, list) or not raw:
        raise ValueError("Statement must be an object or a non-empty list")
    out = []
    for i, st in enumerate(raw):
        where = f"statement {i + 1}"
        if not isinstance(st, dict):
            raise ValueError(f"{where}: not an object")
        if st.get("Effect") not in ("Allow", "Deny"):
            raise ValueError(f"{where}: Effect must be Allow or Deny")
        if ("Action" in st) == ("NotAction" in st):
            raise ValueError(f"{where}: needs exactly one of Action or NotAction")
        if ("Resource" in st) == ("NotResource" in st):
            raise ValueError(f"{where}: needs exactly one of Resource or NotResource")
        not_action = "NotAction" in st
        actions = _as_list(st["NotAction" if not_action else "Action"], "Action", where)
        not_resource = "NotResource" in st
        resources = _as_list(st["NotResource" if not_resource else "Resource"], "Resource", where)
        out.append({
            "label": clean(st.get("Sid") or f"#{i + 1}", 60),
            "allow": st["Effect"] == "Allow",
            "actions": actions,
            "not_action": not_action,
            "resources": resources,
            "not_resource": not_resource,
            "conditional": bool(st.get("Condition")),
        })
    return out


def _grants(st: dict, probe: str) -> bool:
    hit = any(_glob(a, probe) for a in st["actions"])
    return not hit if st["not_action"] else hit


def _broad(st: dict) -> bool:
    return st["not_resource"] or "*" in st["resources"]


def _denied(probe: str, allow_st: dict, denies: list) -> bool:
    for d in denies:
        if d["conditional"] or d["not_resource"] or not _grants(d, probe):
            continue
        if "*" in d["resources"]:
            return True
        if not _broad(allow_st) and all(any(_glob(p, r) for p in d["resources"]) for r in allow_st["resources"]):
            return True
    return False


def _lower(severity: str) -> str:
    return SEVERITIES[min(SEVERITIES.index(severity) + 1, len(SEVERITIES) - 1)]


def analyze(doc) -> list:
    """Return findings [{rule, severity, why, statements}], highest severity first."""
    statements = _statements(doc)
    allows = [s for s in statements if s["allow"]]
    denies = [s for s in statements if not s["allow"]]
    found = {}

    def add(rule, severity, why, label):
        f = found.setdefault(rule, {"rule": rule, "severity": severity, "why": why, "statements": []})
        if SEVERITIES.index(severity) < SEVERITIES.index(f["severity"]):
            f["severity"] = severity
        if label not in f["statements"]:
            f["statements"].append(label)

    for st in allows:
        for rule, probes, severity, why, star_only in RULES:
            if star_only and not _broad(st):
                continue
            if any(_grants(st, p) and not _denied(p, st, denies) for p in probes):
                add(rule, severity if _broad(st) else _lower(severity), why, st["label"])
        if _broad(st):
            add(STAR_RESOURCE[0], STAR_RESOURCE[1], STAR_RESOURCE[2], st["label"])
    order = [r[0] for r in RULES] + [STAR_RESOURCE[0]]
    return sorted(found.values(), key=lambda f: (SEVERITIES.index(f["severity"]), order.index(f["rule"])))


def rating(findings: list) -> str:
    for severity in SEVERITIES:
        if any(f["severity"] == severity for f in findings):
            return severity
    return "none"


def render(findings: list, source: str) -> str:
    lines = [f"Tripwire IAM blast radius: {clean(source, 300)}"]
    if findings:
        lines.append(f"{'severity':<9} {'pattern':<22} {'why':<60} granted by")
        for f in findings:
            lines.append(f"{f['severity']:<9} {f['rule']:<22} {f['why']:<60} {', '.join(f['statements'])}")
    else:
        lines.append("No risky reachable actions found.")
    counts = ", ".join(f"{sum(f['severity'] == s for f in findings)} {s}" for s in SEVERITIES)
    lines.append(f"Overall: {rating(findings).upper()} ({counts})")
    return "\n".join(lines) + "\n"


def run_aws(cmd: list):
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return done.returncode, done.stdout, done.stderr


def access_analyzer(path: str, runner=run_aws) -> str:
    """Run the free `aws accessanalyzer validate-policy` and format its findings."""
    cmd = ["aws", "accessanalyzer", "validate-policy", "--policy-type", "IDENTITY_POLICY",
           "--policy-document", f"file://{path}", "--output", "json"]
    try:
        code, stdout, stderr = runner(cmd)
        if code != 0:
            raise RuntimeError((stderr or stdout or f"exit {code}").strip().splitlines()[-1])
        findings = json.loads(stdout)["findings"]
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError,
            subprocess.SubprocessError) as error:
        return f"Access Analyzer did not run: {clean(error)}\n"
    lines = [f"Access Analyzer (aws accessanalyzer validate-policy): {len(findings)} finding(s)"]
    for f in findings:
        lines.append(f"  {clean(f.get('findingType', '?'), 20):<17} {clean(f.get('issueCode', '?'), 60)}: "
                     f"{clean(f.get('findingDetails', ''), 200)}")
    return "\n".join(lines) + "\n"


def main(argv=None, runner=run_aws) -> int:
    parser = argparse.ArgumentParser(description="List risky actions an IAM identity policy lets an agent reach.")
    parser.add_argument("policy", help="IAM policy JSON file.")
    parser.add_argument("--access-analyzer", action="store_true",
                        help="Also run `aws accessanalyzer validate-policy` (needs the AWS CLI and a sign-in).")
    args = parser.parse_args(argv)
    try:
        findings = analyze(json.loads(Path(args.policy).read_text(encoding="utf-8")))
    except OSError as error:
        parser.error(f"cannot read {args.policy}: {error.strerror or error}")
    except ValueError as error:
        parser.error(f"{args.policy}: {clean(error, 300)}")
    sys.stdout.write(render(findings, args.policy))
    if args.access_analyzer:
        sys.stdout.write("\n" + access_analyzer(args.policy, runner))
    return 1 if rating(findings) == "high" else 0


if __name__ == "__main__":
    sys.exit(main())
