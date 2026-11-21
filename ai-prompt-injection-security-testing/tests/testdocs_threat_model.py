"""Doc-link test: the threat model and runbook stay consistent with the repo.

Reads the two Markdown files and the repo, nothing else: no Range imports,
no network. Named testdocs_* so it matches discover's test*.py pattern but
not the protected tests/test_*.py glob (same trick as testrange_*).
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
THREAT_MODEL = DOCS / "threat_model.md"
RUNBOOK = DOCS / "runbook_prompt_injection.md"

REQUIRED_ROWS = (
    "Injected document",
    "Tool misuse",
    "Canary leak through a reply",
    "Over-broad IAM role",
    "Missing logging",
)

COMMAND_RE = re.compile(r"python3 -m ([\w.]+)([^`\n]*)")
FLAG_RE = re.compile(r"--[A-Za-z][\w-]*")
DATA_PATH_RE = re.compile(r"`(data/[^\s`]+?)`")
MERMAID_FENCE_RE = re.compile(r"```mermaid")
LIMITS_HEADING_RE = re.compile(r"^#+\s+Limits", re.MULTILINE)


def load_valid_ids():
    """Every case id from every data/*.json file that has a cases list."""
    valid = set()
    for path in sorted(ROOT.glob("data/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cases = payload.get("cases")
        if isinstance(cases, list):
            for case in cases:
                if isinstance(case, dict) and isinstance(case.get("id"), str):
                    valid.add(case["id"])
    return valid


def id_pattern_for(valid_ids):
    prefixes = sorted({vid.split("-", 1)[0] for vid in valid_ids if "-" in vid})
    return re.compile(r"\b(?:%s)-\d+\b" % "|".join(prefixes))


def find_candidate_ids(text, pattern):
    return pattern.findall(text)


def invalid_ids(text, valid_ids):
    """Candidate ids in text that are not real case ids."""
    pattern = id_pattern_for(valid_ids)
    return [found for found in find_candidate_ids(text, pattern) if found not in valid_ids]


def resolve_module(module):
    return ROOT / (module.replace(".", "/") + ".py")


def command_problems(text):
    """Every python3 -m module must resolve, every --flag must exist in it."""
    problems = []
    for match in COMMAND_RE.finditer(text):
        module, rest = match.group(1), match.group(2)
        target = resolve_module(module)
        if not target.is_file():
            problems.append("module %r has no file %s" % (module, target))
            continue
        source = target.read_text(encoding="utf-8")
        for flag in FLAG_RE.findall(rest):
            if flag not in source:
                problems.append("flag %r is not in %s" % (flag, target))
    return problems


def command_count(text):
    return len(COMMAND_RE.findall(text))


def missing_data_paths(text):
    problems = []
    for path in DATA_PATH_RE.findall(text):
        if not (ROOT / path.rstrip(".,;:")).exists():
            problems.append("data path %r does not exist" % path)
    return problems


def table_rows(text):
    return [line for line in text.splitlines() if line.lstrip().startswith("|")]


def row_for(rows, name):
    wanted = name.lower()
    for row in rows:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        if cells and cells[0].lower() == wanted:
            return row
    return None


def limits_section(text):
    match = LIMITS_HEADING_RE.search(text)
    return text[match.start():] if match else ""


def section_after(text, heading):
    """The section starting at a heading line, up to the next ## heading."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(heading))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end])


class ThreatModelDocsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid_ids = load_valid_ids()
        cls.threat = THREAT_MODEL.read_text(encoding="utf-8")
        cls.runbook = RUNBOOK.read_text(encoding="utf-8")
        cls.rows = table_rows(cls.threat)

    def test_case_ids_are_real(self):
        found = find_candidate_ids(self.threat, id_pattern_for(self.valid_ids))
        self.assertGreater(len(found), 0, "threat model names no case ids")
        self.assertEqual(invalid_ids(self.threat, self.valid_ids), [])

    def test_commands_resolve(self):
        self.assertGreater(command_count(self.threat), 0, "threat model has no commands")
        self.assertEqual(command_problems(self.threat), [])

    def test_data_paths_exist(self):
        self.assertEqual(missing_data_paths(self.threat), [])

    def test_required_rows(self):
        for name in REQUIRED_ROWS:
            row = row_for(self.rows, name)
            self.assertIsNotNone(row, "missing threat table row %r" % name)
            has_id = bool(set(find_candidate_ids(row, id_pattern_for(self.valid_ids))))
            has_command = command_count(row) > 0
            self.assertTrue(
                has_id or has_command or "no Tripwire test" in row,
                "row %r needs a case id, a command, or 'no Tripwire test'" % name,
            )

    def test_runbook_names_gate(self):
        self.assertIn("tester.gate", self.runbook)
        self.assertIn("python3 -m tester.gate", self.runbook)
        self.assertEqual(command_problems(self.runbook), [])

    def test_contain_step_checks_edited_policy(self):
        contain = section_after(self.runbook, "## 2. Contain")
        self.assertIn("tester.iam_blast", contain)
        self.assertNotIn("range_least_privilege", contain)

    def test_limits_sections(self):
        for text, name in ((self.threat, "threat_model.md"), (self.runbook, "runbook_prompt_injection.md")):
            section = limits_section(text).lower()
            self.assertTrue(LIMITS_HEADING_RE.search(text), "%s needs a Limits heading" % name)
            self.assertIn("synthetic", section, "%s Limits must say synthetic" % name)
            self.assertIn("pretend", section, "%s Limits must say pretend" % name)

    def test_single_mermaid_block(self):
        self.assertEqual(len(MERMAID_FENCE_RE.findall(self.threat)), 1)

    def test_id_check_bites_on_made_up_id(self):
        made_up = "a fabricated id such as RNG-99 is not a real case"
        self.assertEqual(invalid_ids(made_up, self.valid_ids), ["RNG-99"])


if __name__ == "__main__":
    unittest.main()
