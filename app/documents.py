"""Tiny in-memory document store with keyword-overlap retrieval (RAG source)."""
import json
import os
import re

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class DocumentStore:
    def __init__(self, docs: list[dict]):
        self.docs = docs

    @classmethod
    def from_file(cls, path: str | None = None) -> "DocumentStore":
        path = path or os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs.json")
        with open(path) as f:
            return cls(json.load(f))

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = _tokens(query)
        scored = []
        for d in self.docs:
            overlap = len(q & _tokens(d["title"] + " " + d["text"]))
            if overlap:
                scored.append((overlap, d))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:k]]
