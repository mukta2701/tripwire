"""Range retrieval and corpus tests (testrange_* avoids the protected tests/test_*.py glob)."""

import tempfile
import unittest
from pathlib import Path

from agent_range import retrieval

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "range_corpus"
CASES = ROOT / "data" / "range_cases.json"


class RetrievalTests(unittest.TestCase):
    def test_deterministic_and_id_ordered(self):
        corpus = retrieval.load_corpus(CORPUS)
        first = [d.id for d in retrieval.retrieve("balance checking account C-1001", corpus)]
        second = [d.id for d in retrieval.retrieve("balance checking account C-1001", corpus)]
        self.assertEqual(first, second)
        self.assertEqual([d.id for d in corpus], sorted(d.id for d in corpus))

    def test_load_order_independent_of_creation_order(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for name in ("b-note", "a-note"):
                (Path(first) / f"{name}.txt").write_text(f"hello {name} world", encoding="utf-8")
            for name in ("a-note", "b-note"):
                (Path(second) / f"{name}.txt").write_text(f"hello {name} world", encoding="utf-8")
            one, two = retrieval.load_corpus(Path(first)), retrieval.load_corpus(Path(second))
            self.assertEqual([d.id for d in one], ["a-note", "b-note"])
            self.assertEqual([d.id for d in retrieval.retrieve("hello world", one)],
                             [d.id for d in retrieval.retrieve("hello world", two)])

    def test_fewer_than_k_and_empty(self):
        corpus = retrieval.load_corpus(CORPUS)
        self.assertEqual([d.id for d in retrieval.retrieve("refund window unused", corpus)],
                         ["refund-policy-note"])
        self.assertEqual(retrieval.retrieve("zxqv wobble fnord", corpus), [])

    def test_case_insensitive_and_punctuation(self):
        corpus = retrieval.load_corpus(CORPUS)
        lower = [d.id for d in retrieval.retrieve("what is the balance?", corpus)]
        upper = [d.id for d in retrieval.retrieve("WHAT IS THE BALANCE?!", corpus)]
        self.assertEqual(lower, upper)


if __name__ == "__main__":
    unittest.main()
