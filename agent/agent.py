"""The agent loop: recall -> respond -> remember."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any, Iterator

from . import prompts
from .config import Settings
from .llm import LLM, Message, make_llm
from .memory import EXCHANGE, FACT, Memory, SummaryStore, VectorMemory


class Agent:
    def __init__(self, settings: Settings | None = None, llm: LLM | None = None,
                 memory: VectorMemory | None = None, summaries: SummaryStore | None = None):
        self.settings = settings or Settings()
        self.llm = llm or make_llm(self.settings)
        self.memory = memory or VectorMemory(self.settings.chroma_dir, embed=self.llm.embed,
                                             half_life_days=self.settings.decay_half_life_days)
        self.summaries = summaries or SummaryStore(self.settings.summary_file)
        self.history: list[Message] = []          # this session only
        self.last_recalled: list[Memory] = []     # what was pulled into the most recent reply
        self.last_learned: list[str] = []         # facts saved after the most recent reply
        self.last_updated: list[tuple[str, str]] = []  # (outdated fact, fact that replaced it)

    # ---------- recall ----------
    def recall(self, query: str) -> list[Memory]:
        s = self.settings
        facts = self.memory.search(query, kind=FACT, k=s.facts_k, max_distance=s.max_distance)
        exchanges = self.memory.search(query, kind=EXCHANGE, k=s.exchanges_k, max_distance=s.max_distance)
        return facts + exchanges

    def build_messages(self, user_text: str) -> list[Message]:
        recalled = self.recall(user_text)
        self.last_recalled = recalled
        self.memory.touch([m.id for m in recalled])
        facts = [m.text for m in recalled if m.kind == FACT]
        exchanges = [m.text for m in recalled if m.kind == EXCHANGE]
        context = prompts.CONTEXT.format(
            summary=self.summaries.summary or "Nothing yet. This may be your first conversation.",
            facts="\n".join(f"- {f}" for f in facts) or "- (none)",
            exchanges="\n\n".join(exchanges) or "(none)",
        )
        recent = self.history[-self.settings.recent_turns * 2:]
        return [{"role": "system", "content": prompts.SYSTEM + "\n\n" + context}, *recent,
                {"role": "user", "content": user_text}]

    # ---------- respond ----------
    def respond(self, user_text: str) -> str:
        reply = self.llm.chat(self.build_messages(user_text))
        self._remember(user_text, reply)
        return reply

    def stream(self, user_text: str) -> Iterator[str]:
        """Yield the reply piece by piece; memory is updated once the reply is complete."""
        parts: list[str] = []
        for piece in self.llm.stream(self.build_messages(user_text)):
            parts.append(piece)
            yield piece
        self._remember(user_text, "".join(parts))

    # ---------- remember ----------
    def _remember(self, user_text: str, reply: str) -> None:
        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
        self.memory.add(f"User: {_clip(user_text)}\nAssistant: {_clip(reply)}", EXCHANGE)

        self.last_learned, self.last_updated = [], []
        for fact in self.extract_facts(user_text):
            self.learn(fact)

        if self.summaries.bump() % self.settings.summarize_every == 0:
            self.refresh_summary()

    def learn(self, fact: str) -> bool:
        """Store a new fact. Skip it if we already know it; replace older facts it makes outdated."""
        nearest = self.memory.search(fact, kind=FACT, k=1, rerank=False)
        if nearest and nearest[0].distance <= self.settings.duplicate_distance:
            return False
        if self.settings.update_facts:
            related = self.memory.search(fact, kind=FACT, k=3, max_distance=self.settings.related_distance, rerank=False)
            for old in self.find_outdated(fact, related):
                self.memory.forget(old.id)
                self.last_updated.append((old.text, fact))
        self.memory.add(fact, FACT)
        self.last_learned.append(fact)
        return True

    def find_outdated(self, new_fact: str, candidates: list[Memory]) -> list[Memory]:
        if not candidates:
            return []
        existing = "\n".join(f"{i}. {m.text}" for i, m in enumerate(candidates, 1))
        raw = self.llm.chat([{"role": "user", "content": prompts.RECONCILE.format(new=new_fact, existing=existing)}])
        return [candidates[i - 1] for i in parse_number_list(raw, len(candidates))]

    def extract_facts(self, user_text: str) -> list[str]:
        raw = self.llm.chat([{"role": "user", "content": prompts.EXTRACT_FACTS.format(message=user_text)}])
        return parse_fact_list(raw)

    def refresh_summary(self) -> str:
        window = self.history[-self.settings.summarize_every * 2:]
        conversation = "\n".join(f"{m['role'].title()}: {m['content']}" for m in window)
        new = self.llm.chat([{"role": "user", "content": prompts.UPDATE_SUMMARY.format(
            summary=self.summaries.summary or "(empty)", conversation=conversation)}])
        self.summaries.set(new)
        return new

    def new_session(self) -> None:
        """Start a fresh conversation, as if the app was restarted. Long-term memory is kept."""
        self.history.clear()
        self.last_recalled = []
        self.last_learned = []
        self.last_updated = []

    def reset_memory(self) -> None:
        self.memory.clear()
        self.summaries.clear()
        self.history.clear()
        self.last_recalled, self.last_learned, self.last_updated = [], [], []

    # ---------- export / import ----------
    def export_memory(self) -> dict[str, Any]:
        """Everything the agent remembers, as plain JSON (no embeddings, so it works with any model)."""
        return {
            "format": EXPORT_FORMAT,
            "version": 1,
            "exported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "summary": self.summaries.summary,
            "turns": self.summaries.turns,
            "memories": [{"text": m.text, "kind": m.kind, "created": m.created, "uses": m.uses,
                          "last_used": m.last_used} for m in reversed(self.memory.all())],
        }

    def import_memory(self, data: dict[str, Any]) -> int:
        """Merge an export into this memory. Returns how many memories were added; skips ones already known."""
        if not isinstance(data, dict) or data.get("format") != EXPORT_FORMAT:
            raise ValueError("This file isn't a memory export from this app.")
        known_exchanges = {m.text for m in self.memory.all(EXCHANGE)}
        added = 0
        for item in data.get("memories", []):
            text, kind = str(item.get("text", "")).strip(), item.get("kind")
            if not text or kind not in (FACT, EXCHANGE):
                continue
            if kind == FACT:
                nearest = self.memory.search(text, kind=FACT, k=1, rerank=False)
                if nearest and nearest[0].distance <= self.settings.duplicate_distance:
                    continue
            elif text in known_exchanges:
                continue
            self.memory.add(text, kind, created=item.get("created"), uses=int(item.get("uses") or 0),
                            last_used=item.get("last_used"))
            added += 1
        if not self.summaries.summary and data.get("summary"):
            self.summaries.set(str(data["summary"]))
        return added


EXPORT_FORMAT = "local-ai-agent-memory"


def parse_number_list(raw: str, upper: int) -> list[int]:
    """Pull a JSON list of 1-based positions out of a model reply, ignoring anything out of range."""
    match = re.search(r"\[[^\[\]]*\]", raw)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for i in items:
        if isinstance(i, (int, float, str)) and str(i).strip().isdigit() and 1 <= int(str(i).strip()) <= upper:
            if int(str(i).strip()) not in out:
                out.append(int(str(i).strip()))
    return out


def parse_fact_list(raw: str) -> list[str]:
    """Local models don't always return clean JSON, so pull the first [...] block and be forgiving."""
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [i.strip() for i in items if isinstance(i, str) and i.strip()][:5]


def _clip(text: str, limit: int = 500) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"
