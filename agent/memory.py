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
import math
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
    uses: int = 0                 # how many replies this memory has been recalled into
    last_used: str | None = None  # when it was last recalled


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _days_since(stamp: str | None, now: datetime) -> float:
    if not stamp:
        return 0.0
    try:
        then = datetime.fromisoformat(stamp)
    except ValueError:
        return 0.0
    return max(0.0, (now - then).total_seconds() / 86400)


def rank_score(distance: float, uses: int, last_used: str | None, created: str,
               half_life_days: float = 90.0, now: datetime | None = None) -> float:
    """Lower is better. Similarity decides most of the ranking; on top of that:

    * reinforcement - memories that keep being useful get a small boost (up to 0.05)
    * decay         - memories nobody has needed for a while slowly sink (up to 0.08),
                      losing half of their remaining standing every `half_life_days`
    """
    now = now or datetime.now(timezone.utc)
    boost = 0.05 * math.log1p(min(uses, 10)) / math.log1p(10)
    idle = _days_since(last_used or created, now)
    decay = 0.08 * (1 - 0.5 ** (idle / half_life_days)) if half_life_days > 0 else 0.0
    return distance - boost + decay


def _to_memory(mem_id: str, doc: str, meta: dict, dist: float | None = None) -> Memory:
    return Memory(mem_id, doc, meta["kind"], meta["created"], dist,
                  int(meta.get("uses", 0)), meta.get("last_used") or None)


class VectorMemory:
    def __init__(self, path: Path | str, embed: EmbedFn, collection: str = "memories",
                 half_life_days: float = 90.0):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})
        self._embed = embed
        self.half_life_days = half_life_days

    # ---------- write ----------
    def add(self, text: str, kind: str, created: str | None = None, uses: int = 0,
            last_used: str | None = None) -> str:
        mem_id = uuid.uuid4().hex[:12]
        self._col.add(
            ids=[mem_id],
            documents=[text],
            embeddings=self._embed([text]),
            metadatas=[{"kind": kind, "created": created or _now(), "uses": int(uses), "last_used": last_used or ""}],
        )
        return mem_id

    def touch(self, ids: list[str]) -> None:
        """Record that these memories were just used in a reply (feeds reinforcement and decay)."""
        if not ids:
            return
        res = self._col.get(ids=ids)
        now = _now()
        metas = [{**m, "uses": int(m.get("uses", 0)) + 1, "last_used": now} for m in res["metadatas"]]
        self._col.update(ids=res["ids"], metadatas=metas)

    def add_fact(self, text: str, duplicate_distance: float = 0.12) -> str | None:
        """Store a fact unless we already know something almost identical."""
        nearest = self.search(text, kind=FACT, k=1, rerank=False)
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
    def search(self, query: str, kind: str | None = None, k: int = 5, max_distance: float | None = None,
               rerank: bool = True) -> list[Memory]:
        """Closest memories by meaning. `max_distance` filters on pure similarity; the order then
        also accounts for reinforcement and decay (see `rank_score`) unless `rerank` is False."""
        total = self.count(kind)
        if total == 0:
            return []
        res = self._col.query(
            query_embeddings=self._embed([query]),
            n_results=min(k * 3 if rerank else k, total),
            where={"kind": kind} if kind else None,
        )
        out = [_to_memory(i, d, m, dist) for i, d, m, dist in
               zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
               if max_distance is None or dist <= max_distance]
        if rerank:
            now = datetime.now(timezone.utc)
            out.sort(key=lambda m: rank_score(m.distance, m.uses, m.last_used, m.created, self.half_life_days, now))
        return out[:k]

    def all(self, kind: str | None = None) -> list[Memory]:
        res = self._col.get(where={"kind": kind} if kind else None)
        items = [_to_memory(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
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
