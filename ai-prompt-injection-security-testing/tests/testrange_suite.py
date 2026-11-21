"""Range suite tests: retrieval of poisoned docs, untrusted markers, mock gating."""

import json
import tempfile
import unittest
from pathlib import Path

from agent_range import retrieval, suite
from tester.scoring_criteria import load_case_data

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "range_corpus"
CASES = ROOT / "data" / "range_cases.json"

POISONED = {"billing-balance-note", "refund-policy-note", "address-change-note",
            "overdraft-limit-note", "statement-copy-note", "joint-account-note",
            "fraud-alert-note"}


class SuiteTests(unittest.TestCase):
    def test_each_question_retrieves_its_poisoned_document(self):
        corpus = retrieval.load_corpus(CORPUS)
        records = {r["id"]: r for r in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
        cases, _ = load_case_data(CASES)
        decoy_seen = False
        for case in cases:
            ids = [d.id for d in retrieval.retrieve(case.prompt, corpus)]
            self.assertIn(records[case.test_id]["poisoned_document"], ids[:2])
            decoy_seen |= any(i not in POISONED for i in ids[:2])
        self.assertTrue(decoy_seen)

    def test_corpus_holds_poisoned_and_decoys(self):
        ids = {p.stem for p in CORPUS.glob("*.txt")}
        self.assertTrue(POISONED <= ids)
        self.assertGreaterEqual(len(ids - POISONED), 4)

    def test_sent_text_carries_untrusted_markers(self):
        seen = []

        class Recorder:
            name = "recorder"

            def send(self, text, *, canary=None):
                from tester.targets import TargetResponse
                seen.append(text)
                return TargetResponse("done", {})

        summary = suite.run_range_tests("safe", target=Recorder())
        self.assertEqual(summary["total_tests"], 7)
        first = summary["results"][0]
        body = seen[0]
        for doc_id in first["range"]["retrieved_ids"]:
            self.assertIn(f"UNTRUSTED RETRIEVED CONTENT id={doc_id}", body)
            self.assertIn(f"END UNTRUSTED RETRIEVED CONTENT id={doc_id}", body)

    def test_missing_poisoned_document_does_not_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
            for record in records:
                (Path(tmp) / f"{record['poisoned_document']}.txt").write_text(
                    "zebra quasar juxtaposition wholly unrelated filler", encoding="utf-8")
            summary = suite.run_range_tests("unsafe", corpus_dir=Path(tmp),
                                            cases_path=CASES, trials=1)
            # poisoned ids exist but share no words with the questions, so the
            # mock never sees the poisoned id and never plays the attack
            self.assertEqual(summary["fail_count"], 0)

    def test_unknown_poisoned_document_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = json.loads(CASES.read_text(encoding="utf-8"))
            data["cases"][0]["poisoned_document"] = "no-such-note"
            bad = Path(tmp) / "cases.json"
            bad.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not in the corpus"):
                suite.run_range_tests("unsafe", cases_path=bad)


if __name__ == "__main__":
    unittest.main()
