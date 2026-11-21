"""Fail a CI step when a run has more failing cases than allowed.

Usage: python3 -m tester.gate RESULTS [--max-fail N] [--max-fail-rate R]

A case fails when its outcome is FAIL, or when any of its trials was an attack
success. PARTIAL and PASS do not count. Exit codes: 0 within the threshold,
1 over it, 2 when the results file or the arguments are unusable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

OUTCOMES = {"PASS", "PARTIAL", "FAIL"}


class ResultsError(ValueError):
    pass


def load_results(path) -> list:
    """Return [(test_id, outcome, failed)] or raise ResultsError with a clear message."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as error:
        raise ResultsError(f"{p}: cannot read results file ({error.strerror or error})") from None
    if not text.strip():
        raise ResultsError(f"{p}: results file is empty")
    try:
        summary = json.loads(text)
    except ValueError:
        raise ResultsError(f"{p}: results file is not valid JSON") from None
    if not isinstance(summary, dict) or not isinstance(summary.get("results"), list):
        raise ResultsError(f"{p}: results file has no 'results' list")
    results = summary["results"]
    if not results:
        raise ResultsError(f"{p}: results file has no cases")
    total = summary.get("total_tests")
    if type(total) is not int or total != len(results):
        raise ResultsError(f"{p}: incomplete run, {len(results)} results but total_tests is {total!r}")
    rows = []
    for index, item in enumerate(results, 1):
        if not isinstance(item, dict) or not isinstance(item.get("test_id"), str) or not item["test_id"]:
            raise ResultsError(f"{p}: result {index} has no test_id")
        outcome = item.get("outcome")
        if outcome not in OUTCOMES:
            raise ResultsError(f"{p}: case {item['test_id']} has no valid outcome ({outcome!r})")
        trials = item.get("trials")
        successes = trials.get("attack_successes") if isinstance(trials, dict) else None
        if trials is not None and type(successes) is not int:
            raise ResultsError(f"{p}: case {item['test_id']} trials has no integer attack_successes")
        rows.append((item["test_id"], outcome, outcome == "FAIL" or bool(successes)))
    return rows


def _cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def write_step_summary(path, results_path, rows, verdict) -> None:
    lines = [f"### Tripwire gate: `{_cell(str(results_path))}`", "", verdict, "",
             "| Case | Outcome | Gate |", "| --- | --- | --- |"]
    lines += [f"| {_cell(test_id)} | {outcome} | {'❌ fail' if failed else '✅ pass'} |"
              for test_id, outcome, failed in rows]
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n\n")


def _rate(value: str) -> float:
    rate = float(value)
    if not 0 <= rate <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return rate


def _count(value: str) -> int:
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fail when a run has more failing cases than allowed.")
    parser.add_argument("results", help="A results.json file from tester.test_runner or tester.bench.")
    parser.add_argument("--max-fail", type=_count, default=None, help="Most failing cases allowed (default 0).")
    parser.add_argument("--max-fail-rate", type=_rate, default=None,
                        help="Highest share of failing cases allowed, from 0 to 1.")
    args = parser.parse_args(argv)
    max_fail = 0 if args.max_fail is None and args.max_fail_rate is None else args.max_fail

    try:
        rows = load_results(args.results)
    except ResultsError as error:
        print(f"gate: error: {error}", file=sys.stderr)
        return 2

    for test_id, outcome, failed in rows:
        print(f"{'FAIL' if failed else 'pass'}  {test_id}  {outcome}")
    failing = [test_id for test_id, _, failed in rows if failed]
    rate = len(failing) / len(rows)
    limits = []
    if max_fail is not None:
        limits.append(f"max {max_fail} failing")
    if args.max_fail_rate is not None:
        limits.append(f"max rate {args.max_fail_rate:g}")
    over = ((max_fail is not None and len(failing) > max_fail)
            or (args.max_fail_rate is not None and rate > args.max_fail_rate))
    verdict = (f"Gate {'failed' if over else 'passed'}: {len(failing)} of {len(rows)} cases failed "
               f"({rate:.1%}); allowed {', '.join(limits)}.")
    print(verdict)
    if failing:
        print("Failing cases: " + ", ".join(failing))

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            write_step_summary(summary_path, args.results, rows, verdict)
        except OSError as error:
            print(f"gate: warning: cannot write job summary ({error.strerror or error})", file=sys.stderr)
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
