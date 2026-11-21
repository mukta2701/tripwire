# Prompt-injection runbook

For the owner and security readers: what to do when a Tripwire run reports a FAIL that may be a real prompt-injection leak. Read the threat model first so each step below points at the right arrow.

## 1. Triage

Read the FAIL line in the gate output, then open that case's `log_line` and `tool_call` in `results.json` or the evidence pack. Decide whether it is a real leak or a scorer artefact: a refusal that quotes a call still scores FAIL, so a quoted call with no executed effect is an artefact, not a leak.

Tell the owner at once when it looks real, and report through `../../SECURITY.md`.

## 2. Contain

Revoke the tool's permission. For the Range, remove the tool from `TOOLS` in `agent_range/tools.py` so `instructions()` stops offering it to the agent. For the AWS version, remove the Allow from the role's policy, then check with `python3 -m tester.iam_blast data/iam/range_least_privilege.json` that no high finding remains.

## 3. Root cause

Name which document carried the injection (the case's `poisoned_document`), which tool fired, and which threat-model arrow it crossed (the case's `forbidden_tools` says which tools were off limits). Check whether the agent needed that tool at all: if not, the tool should never have been offered.

## 4. Fix

Apply least privilege, add a human approval step for risky tools, and turn on input flagging and output blocking with `--guardrail local`. None of these is a complete defence: markers and flagging raise the bar but a new phrasing can still get through.

## 5. Retest with the gate

Rerun the setup that failed. For the Range mock that is `python3 -m tester.bench --range --target mock --mode unsafe --trials 1`, then `python3 -m tester.gate runs/<folder>/results.json --max-fail 0` using the run folder the bench run printed.

A pass looks like `Gate passed: 0 of N cases failed` with exit code 0, where N is the case count. Exit code 1 means still failing. Exit code 2 means the file is unusable.

Run the retest without `--guardrail`: the gate rejects `BLOCKED` outcomes, so a guardrail-on run exits 2 and cannot be gated. Compare a guardrail-on run with `python3 -m tester.compare` instead of gating it.

Mock honesty: the unsafe mock always fails and the safe mock always passes, so a mock retest proves the gate works and not that the fix works. Retest the fix on the model or the setup that failed.

## Limits

This runbook assumes synthetic data and canaries, small local models, and pretend tools that touch nothing real: containing a pretend tool proves nothing about containing a real one. The AWS steps were tested only offline through a local policy file, and no live AWS runbook step was run.
