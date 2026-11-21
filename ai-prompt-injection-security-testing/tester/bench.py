"""One command: run the cases, show live progress, save everything in a run folder."""

from __future__ import annotations

import argparse
import json
import re
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tester.report import DEFAULT_CROSSWALK, render_report  # noqa: E402
from tester.report_html import render_html, scorer_panel_data  # noqa: E402
from tester.scoring_criteria import DEFAULT_CASES_PATH, load_case_data  # noqa: E402
from tester.targets import (BEDROCK_PRICES_ESTIMATE, SYSTEM_VARIANTS, BedrockTarget,  # noqa: E402
                            MockTarget, OpenAICompatTarget)
from tester.test_runner import run_tests  # noqa: E402
from agent_range.suite import run_range_tests  # noqa: E402

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def safe_text(value, limit: int) -> str:
    """One line, no ANSI or control characters: model replies are untrusted."""
    text = _CONTROL.sub("", _ANSI.sub("", " ".join(str(value).split())))
    return text if len(text) <= limit else text[:limit] + "..."


def folder_name(when: datetime, label: str) -> str:
    return f"{when:%Y-%m-%d_%H%M}_{re.sub(r'[^A-Za-z0-9._-]+', '-', label).strip('-') or 'run'}"


def cost_line(model: str, cases, trials: int, system_variant: str, max_tokens: int) -> str:
    """Worst-case spend: every call uses its full output cap. Input is guessed at 4 chars per unit."""
    template = SYSTEM_VARIANTS[system_variant] or ""
    chars = sum(len(c.prompt) + len(template.format(canary=c.canary)) for c in cases) * trials
    calls = len(cases) * trials
    usage_in, usage_out = chars // 4 + 1, calls * max_tokens
    head = f"Estimated cost (estimate, check the Bedrock pricing page): {calls} calls, "
    if model not in BEDROCK_PRICES_ESTIMATE:
        return head + f"no price on file for {safe_text(model, 80)}\n"
    price_in, price_out = BEDROCK_PRICES_ESTIMATE[model]
    dollars = (usage_in * price_in + usage_out * price_out) / 1_000_000
    return head + (f"up to ${dollars:.4f} (~{usage_in} in + up to {usage_out} out at "
                   f"${price_in}/${price_out} per million)\n")


class Progress:
    """Prints one block per finished request; the tally uses scorer outcomes, not a verdict."""

    def __init__(self, out, guardrail="none"):
        self.out = out
        self.guardrail = guardrail
        self.tally = {"PASS": 0, "FAIL": 0, "PARTIAL": 0, "BLOCKED": 0}

    def __call__(self, e: dict) -> None:
        self.tally[e["outcome"]] = self.tally.get(e["outcome"], 0) + 1
        t = self.tally
        width = len(str(e["total"]))
        lines = [
            f"[{e['index']:>{width}}/{e['total']}] {safe_text(e['test_id'], 20)} {safe_text(e['family'], 40)}"
            f"  trial {e['trial']}/{e['trials']}  {e['seconds']:.1f}s",
            f"  sent:  \"{safe_text(e['prompt'], 80)}\"",
            f"  reply: \"{safe_text(e['reply'], 100)}\"",
            f"  so far (scorer labels, not a verdict): {t['PASS']} PASS (refused/ok) · "
            f"{t['FAIL']} FAIL (went along) · {t['PARTIAL']} PARTIAL (unclear)",
        ]
        if self.guardrail == "local":
            lines[-1] += (f" · {t.get('BLOCKED', 0)} BLOCKED "
                          "(blocked by guardrail)")
            for block in (e.get("blocks") or []):
                lines.append(
                    f"  blocked by guardrail: {safe_text(block.get('check', ''), 40)} "
                    f"matched \"{safe_text(block.get('matched', ''), 100)}\"")
        self.out.write("\n".join(lines) + "\n")
        self.out.flush()


def main(argv=None, out=None, open_url=webbrowser.open, now=None) -> int:
    out = out if out is not None else sys.stdout
    parser = argparse.ArgumentParser(description="Run the cases with live progress and save a run folder.")
    parser.add_argument("model_name", nargs="?", default="", metavar="MODEL", help="Model name (same as --model).")
    parser.add_argument("--model", default="", help="Model name for openai_compat or bedrock.")
    parser.add_argument("--target", choices=["mock", "openai_compat", "bedrock"], default=None)
    parser.add_argument("--mode", choices=["safe", "unsafe"], default="safe", help="Mock chatbot behaviour.")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--system-variant", choices=list(SYSTEM_VARIANTS), default="none",
                        help="openai_compat and bedrock: put the case canary in a system message (weak or strong).")
    parser.add_argument("--max-calls", type=int, default=300,
                        help="bedrock only: refuse a run that needs more calls (cases x trials).")
    parser.add_argument("--max-tokens", type=int, default=256, help="bedrock only: output cap per reply.")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--requests-per-second", type=float, default=1)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--range", action="store_true",
                        help="Run the Range RAG cases (retrieval plus tool-call verdict) instead.")
    parser.add_argument("--guardrail", choices=["none", "local"], default="none",
                        help="Range only: block risky replies and calls with the local guardrail.")
    parser.add_argument("--i-have-authorization", action="store_true", help="Allow non-loopback targets.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--open", action="store_true", help="Open the HTML report when done.")
    args = parser.parse_args(argv)
    raw = argv if argv is not None else sys.argv[1:]

    model = args.model or args.model_name
    target_name = args.target or ("openai_compat" if model else "mock")
    if args.guardrail == "local" and not args.range:
        parser.error("--guardrail local needs --range: the guardrail only covers Range runs")
    if args.range:
        if target_name == "bedrock":
            parser.error("--range cannot be used with --target bedrock: Range runs need the mock or an OpenAI-compatible model")
        if any(a == "--cases" or a.startswith("--cases=") for a in raw):
            parser.error("--range cannot be combined with --cases: it already selects data/range_cases.json")
        if args.system_variant != "none":
            print("--system-variant is ignored with --range", file=sys.stderr)
        try:
            if target_name == "mock":
                range_target = None
            else:
                range_target = OpenAICompatTarget(
                    args.base_url, model, i_have_authorization=args.i_have_authorization,
                    timeout=args.timeout, requests_per_second=args.requests_per_second)
            summary = run_range_tests(mode=args.mode, target=range_target, trials=args.trials,
                                      temperature=args.temperature, progress=Progress(out, guardrail=args.guardrail),
                                      guardrail=args.guardrail)
            crosswalk = json.loads(Path(DEFAULT_CROSSWALK).read_text(encoding="utf-8"))
        except Exception as error:  # surface any run failure as a clean message, no partial folder
            parser.error(f"run failed: {safe_text(error, 300)}")
        when = now or datetime.now()
        label = f"mock-{args.mode}" if target_name == "mock" else model
        run_dir = args.runs_dir / folder_name(when, label)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (run_dir / "results.manifest.json").write_text(json.dumps(summary["manifest"], indent=2) + "\n", encoding="utf-8")
        (run_dir / "risk_register.md").write_text(render_report(summary, crosswalk), encoding="utf-8")
        html_path = run_dir / "evidence_pack.html"
        html_path.write_text(render_html(summary, crosswalk, scorer=scorer_panel_data()), encoding="utf-8")
        out.write(f"Run folder: {run_dir}\n")
        out.flush()
        if args.open:
            open_url(html_path.resolve().as_uri())
        return 0
    if target_name == "mock" and args.system_variant != "none":
        print("--system-variant is ignored for the mock target", file=sys.stderr)
    try:
        if target_name == "bedrock":
            cases, _ = load_case_data(args.cases)
            calls = len(cases) * args.trials
            if calls > args.max_calls:
                parser.error(f"this run needs {calls} calls ({len(cases)} cases x {args.trials} trials), "
                             f"over --max-calls {args.max_calls}; nothing was sent")
            target = BedrockTarget(model, system_variant=args.system_variant,
                                   max_tokens=args.max_tokens, timeout=args.timeout)
            out.write(cost_line(model, cases, args.trials, args.system_variant, args.max_tokens))
            out.flush()
        elif target_name == "mock":
            target = MockTarget(args.mode)
        else:
            target = OpenAICompatTarget(
                args.base_url, model, i_have_authorization=args.i_have_authorization,
                timeout=args.timeout, requests_per_second=args.requests_per_second,
                system_variant=args.system_variant)
        summary = run_tests(mode=args.mode, cases_path=args.cases, target=target, trials=args.trials,
                            temperature=args.temperature, progress=Progress(out))
        crosswalk = json.loads(Path(DEFAULT_CROSSWALK).read_text(encoding="utf-8"))
    except Exception as error:  # surface any run failure as a clean message, no partial folder
        parser.error(f"run failed: {safe_text(error, 300)}")

    when = now or datetime.now()
    label = f"mock-{args.mode}" if target_name == "mock" else model
    if target_name != "mock" and args.system_variant != "none":
        label += f"_{args.system_variant}"
    run_dir = args.runs_dir / folder_name(when, label)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "results.manifest.json").write_text(json.dumps(summary["manifest"], indent=2) + "\n", encoding="utf-8")
    (run_dir / "risk_register.md").write_text(render_report(summary, crosswalk), encoding="utf-8")
    html_path = run_dir / "evidence_pack.html"
    html_path.write_text(render_html(summary, crosswalk, scorer=scorer_panel_data()), encoding="utf-8")

    out.write(f"Run folder: {run_dir}\n")
    out.flush()
    if args.open:
        open_url(html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
