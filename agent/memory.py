"""Long-term memory.

Two layers:
  * VectorMemory  - facts about the user and past exchanges, stored in ChromaDB and
                    retrieved by meaning (embedding similarity), not keywords.
  * SummaryStore  - a rolling plain-language summary of the whole relationship so far,
                    refreshed every few turns and kept in a small JSON file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable
import uuid

import chromadb

EmbedFn = Callable[[list[str]], list[list[float]]]

FACT = "fact"
EXCHANGE = "exchange"


@dataclass
class Memory:
    id: str
    text: str
    kind: str
    created: str
    distance: float | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class VectorMemory:
    def __init__(self, path: Path | str, embed: EmbedFn, collection: str = "memories"):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})
        self._embed = embed

    # ---------- write ----------
    def add(self, text: str, kind: str) -> str:
        mem_id = uuid.uuid4().hex[:12]
        self._col.add(
            ids=[mem_id],
            documents=[text],
            embeddings=self._embed([text]),
            metadatas=[{"kind": kind, "created": _now()}],
        )
        return mem_id

    def add_fact(self, text: str, duplicate_distance: float = 0.12) -> str | None:
        """Store a fact unless we already know something almost identical."""
        nearest = self.search(text, kind=FACT, k=1)
        if nearest and nearest[0].distance is not None and nearest[0].distance <= duplicate_distance:
            return None
        return self.add(text, FACT)

    def forget(self, mem_id: str) -> None:
        self._col.delete(ids=[mem_id])

    def clear(self) -> None:
        ids = self._col.get()["ids"]
        if ids:
            self._col.delete(ids=ids)

    # ---------- read ----------
    def search(self, query: str, kind: str | None = None, k: int = 5, max_distance: float | None = None) -> list[Memory]:
        total = self.count(kind)
        if total == 0:
            return []
        res = self._col.query(
            query_embeddings=self._embed([query]),
            n_results=min(k, total),
            where={"kind": kind} if kind else None,
        )
        out = []
        for mem_id, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if max_distance is not None and dist > max_distance:
                continue
            out.append(Memory(mem_id, doc, meta["kind"], meta["created"], dist))
        return out

    def all(self, kind: str | None = None) -> list[Memory]:
        res = self._col.get(where={"kind": kind} if kind else None)
        items = [Memory(i, d, m["kind"], m["created"]) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
        return sorted(items, key=lambda m: m.created, reverse=True)

    def count(self, kind: str | None = None) -> int:
        if kind is None:
            return self._col.count()
        return len(self._col.get(where={"kind": kind})["ids"])


class SummaryStore:
    """A rolling summary of everything discussed, plus a turn counter."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = {"summary": "", "turns": 0, "updated": None}
        if self.path.exists():
            try:
                self._data.update(json.loads(self.path.read_text()))
            except json.JSONDecodeError:
                pass  # a corrupted file just starts a fresh summary

    @property
    def summary(self) -> str:
        return self._data["summary"]

    @property
    def turns(self) -> int:
        return self._data["turns"]

    @property
    def updated(self) -> str | None:
        return self._data["updated"]

    def bump(self) -> int:
        self._data["turns"] += 1
        self._save()
        return self._data["turns"]

    def set(self, summary: str) -> None:
        self._data["summary"] = summary.strip()
        self._data["updated"] = _now()
        self._save()

    def clear(self) -> None:
        self._data = {"summary": "", "turns": 0, "updated": None}
        self._save()

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2))
