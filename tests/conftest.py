"""A tiny fake LLM so the tests run in seconds with no Ollama server."""

import hashlib
import json
import math
import re

import pytest

from agent import Agent, Settings


def bag_of_words(text: str, dims: int = 256) -> list[float]:
    """Deterministic embedding: hashed word counts, normalised. Similar wording -> similar vectors."""
    vec = [0.0] * dims
    for word in re.findall(r"[a-z]+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeLLM:
    def __init__(self):
        self.calls: list[list[dict]] = []

    def embed(self, texts):
        return [bag_of_words(t) for t in texts]

    def chat(self, messages):
        self.calls.append(messages)
        prompt = messages[-1]["content"]
        if "durable facts" in prompt:
            facts = []
            if m := re.search(r"my name is (\w+)", prompt, re.I):
                facts.append(f"The user's name is {m.group(1)}.")
            if m := re.search(r"i work as an? ([\w ]+?)[.,]", prompt, re.I):
                facts.append(f"The user works as a {m.group(1)}.")
            return "Here you go:\n" + json.dumps(facts)
        if "running summary" in prompt:
            return "The user has been chatting about their work."
        return "Noted."

    def stream(self, messages):
        for word in self.chat(messages).split(" "):
            yield word + " "


@pytest.fixture
def make_agent(tmp_path):
    def _make():
        settings = Settings(data_dir=tmp_path / "data")
        settings.summarize_every = 2
        settings.max_distance = 0.9  # the fake word-count embeddings are much cruder than real ones
        return Agent(settings=settings, llm=FakeLLM())
    return _make
