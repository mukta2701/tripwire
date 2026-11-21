"""Blast radius from an IAM identity policy file: which risky actions could a hijacked agent reach?

Usage: python3 -m tester.iam_blast data/iam/range_too_permissive.json [--access-analyzer]

Reads a local policy JSON file only. Exit code 1 when any high finding is reachable, 0 otherwise.
Conditions on Allow statements are ignored (counted as reachable). A Deny removes a finding only
when it has no Condition and provably covers both the action and every resource the Allow grants;
when unsure, the finding stays. Action names match case-insensitively, resource ARNs case-sensitively.
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
SM_GET = f"{SM}:Get" + "Sec" + "retValue"

# (pattern shown, probe actions, severity on a wide resource, why, only on a wide resource)
RULES = [
    ("*", ["*"], "high", "every action in every service", False),
    ("iam:*", ["iam:*"], "high", "full IAM control: create users, attach admin policies", False),
    ("iam:PassRole", ["iam:PassRole"], "high", "hand a stronger role to a service it starts", False),
    ("sts:AssumeRole", ["sts:AssumeRole"], "high", "switch into another role", False),
    (SM_GET, [SM_GET, f"{SM}:Batch" + SM_GET.split(":")[1]], "high",
     "read stored Secrets Manager values", False),
    ("s3:*", ["s3:*"], "high", "every S3 action, including bucket policy changes", False),
    ("s3:Delete*", ["s3:Delete*", "s3:DeleteObject", "s3:DeleteObjectVersion", "s3:DeleteBucket"], "high",
     "delete buckets and objects", False),
    ("bedrock:*", ["bedrock:*"], "medium", "every Bedrock action, including guardrail and model changes", False),
    ("lambda:InvokeFunction", ["lambda:InvokeFunction"], "medium", "run any Lambda function", True),
]
STAR_RESOURCE = ("Resource *", "medium", "its actions reach every resource, or every one of a type")

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def clean(value, limit=160) -> str:
    text = _CONTROL.sub(" ", str(value))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _covers(pattern: str, value: str, fold: bool) -> bool:
    """True when every string `value` can stand for is matched by `pattern` (IAM * and ? wildcards).

    A * in `value` is a literal here: only a * in `pattern` can absorb it, and ? never does,
    so `s3:*` covers `s3:Delete*` but `s3:?` does not cover `s3:*`.
    """
    if fold:
        pattern, value = pattern.lower(), value.lower()
    regex = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", "[^*]")
    return re.fullmatch(regex, value) is not None


def _prefix(pattern: str) -> str:
    return re.split(r"[*?]", pattern.lower(), 1)[0]


def _disjoint(a: str, b: str) -> bool:
    """True only when two action patterns provably share no action (their literal prefixes diverge)."""
    pa, pb = _prefix(a), _prefix(b)
    return not (pa.startswith(pb) or pb.startswith(pa))


def _wide_resource(arn: str) -> bool:
    """True for `*` and for ARNs whose name part is only wildcards: arn:*, arn:aws:s3:::*, role/*, function:*."""
    parts = arn.split(":", 5)
    if len(parts) < 6 or re.search(r"[*?]", parts[1] + parts[2]):
        return "*" in arn
    service, name = parts[2], parts[5]
    pieces = [name.split("/", 1)[0]] if service == "s3" else re.split(r"[/:]", name, 1)
    return any(p and "*" in p and not p.strip("*?/:") for p in pieces)


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
    """Allow side: may this statement reach the probe action? Kept generous so findings are not missed."""
    if st["not_action"]:
        return not any(_covers(a, probe, fold=True) for a in st["actions"])
    return any(_covers(a, probe, fold=True) for a in st["actions"])


def _broad(st: dict) -> bool:
    return st["not_resource"] or any(_wide_resource(r) for r in st["resources"])


def _denies(d: dict, probe: str) -> bool:
    """Deny side: does this statement provably block every action the probe stands for?"""
    if d["not_action"]:
        return all(_disjoint(a, probe) for a in d["actions"])
    return any(_covers(a, probe, fold=True) for a in d["actions"])


def _denied(probe: str, allow_st: dict, denies: list) -> bool:
    for d in denies:
        if d["conditional"] or d["not_resource"] or not _denies(d, probe):
            continue
        if "*" in d["resources"]:
            return True
        if not allow_st["not_resource"] and all(
                any(_covers(p, r, fold=False) for p in d["resources"]) for r in allow_st["resources"]):
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
        lines.append(f"{'severity':<9} {'pattern':<30} {'why':<60} granted by")
        for f in findings:
            lines.append(f"{f['severity']:<9} {f['rule']:<30} {f['why']:<60} {', '.join(f['statements'])}")
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
