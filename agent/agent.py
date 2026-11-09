"""The agent loop: recall -> respond -> remember."""

from __future__ import annotations

import json
import re
from typing import Iterator

from . import prompts
from .config import Settings
from .llm import LLM, Message, make_llm
from .memory import EXCHANGE, FACT, Memory, SummaryStore, VectorMemory


class Agent:
    def __init__(self, settings: Settings | None = None, llm: LLM | None = None,
                 memory: VectorMemory | None = None, summaries: SummaryStore | None = None):
        self.settings = settings or Settings()
        self.llm = llm or make_llm(self.settings)
        self.memory = memory or VectorMemory(self.settings.chroma_dir, embed=self.llm.embed)
        self.summaries = summaries or SummaryStore(self.settings.summary_file)
        self.history: list[Message] = []          # this session only
        self.last_recalled: list[Memory] = []     # what was pulled into the most recent reply
        self.last_learned: list[str] = []         # facts saved after the most recent reply

    # ---------- recall ----------
    def recall(self, query: str) -> list[Memory]:
        s = self.settings
        facts = self.memory.search(query, kind=FACT, k=s.facts_k, max_distance=s.max_distance)
        exchanges = self.memory.search(query, kind=EXCHANGE, k=s.exchanges_k, max_distance=s.max_distance)
        return facts + exchanges

    def build_messages(self, user_text: str) -> list[Message]:
        recalled = self.recall(user_text)
        self.last_recalled = recalled
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

        self.last_learned = []
        for fact in self.extract_facts(user_text):
            if self.memory.add_fact(fact, self.settings.duplicate_distance):
                self.last_learned.append(fact)

        if self.summaries.bump() % self.settings.summarize_every == 0:
            self.refresh_summary()

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

    def reset_memory(self) -> None:
        self.memory.clear()
        self.summaries.clear()
        self.history.clear()
        self.last_recalled, self.last_learned = [], []


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
