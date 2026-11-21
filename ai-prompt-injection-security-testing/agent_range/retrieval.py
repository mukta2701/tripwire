"""Keyword retrieval over the Range document corpus.

Documents are plain .txt files; the id is the file stem. Scoring counts how many
distinct question words occur in each document. Ties break by document id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = frozenset({"the", "a", "is", "for", "of", "to", "and"})
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Document:
    id: str
    text: str


def tokenize(text: str) -> set:
    return set(_WORD.findall(text.lower())) - STOPWORDS


def load_corpus(folder) -> list:
    docs = []
    for path in sorted(Path(folder).glob("*.txt")):
        docs.append(Document(path.stem, path.read_text(encoding="utf-8")))
    return sorted(docs, key=lambda d: d.id)


def retrieve(question: str, corpus: list, k: int = 2) -> list:
    words = tokenize(question)
    scored = []
    for doc in corpus:
        haystack = set(_WORD.findall(doc.text.lower()))
        score = len(words & haystack)
        if score:
            scored.append((score, doc.id, doc))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _, _, doc in scored[:k]]
