"""Thin wrapper around a local Ollama server: chat (streaming or not) and embeddings."""

from __future__ import annotations

from typing import Iterator, Protocol

from .config import Settings


Message = dict[str, str]


class LLM(Protocol):
    """Anything that can chat and embed. The agent only depends on this interface."""

    def chat(self, messages: list[Message]) -> str: ...
    def stream(self, messages: list[Message]) -> Iterator[str]: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaLLM:
    def __init__(self, settings: Settings):
        import ollama  # imported here so tests can run without the package talking to a server

        self.settings = settings
        self.client = ollama.Client(host=settings.ollama_host)

    def chat(self, messages: list[Message]) -> str:
        res = self.client.chat(model=self.settings.chat_model, messages=messages)
        return res["message"]["content"]

    def stream(self, messages: list[Message]) -> Iterator[str]:
        for chunk in self.client.chat(model=self.settings.chat_model, messages=messages, stream=True):
            piece = chunk["message"]["content"]
            if piece:
                yield piece

    def embed(self, texts: list[str]) -> list[list[float]]:
        res = self.client.embed(model=self.settings.embed_model, input=texts)
        return list(res["embeddings"])

    def status(self) -> tuple[bool, str]:
        """Check that Ollama is running and both models are pulled."""
        try:
            listed = self.client.list()
        except Exception:
            return False, f"Can't reach Ollama at {self.settings.ollama_host}. Start it with `ollama serve`."
        names = {m["model"].split(":")[0] for m in listed["models"]}
        missing = [m for m in (self.settings.chat_model, self.settings.embed_model) if m.split(":")[0] not in names]
        if missing:
            return False, "Missing models: " + ", ".join(missing) + ". Run `ollama pull <model>` for each."
        return True, f"{self.settings.chat_model} · {self.settings.embed_model}"


class HostedLLM:
    """Hosted model on Groq for the online demo, so visitors can try the agent without installing anything.

    Chat goes to Groq's OpenAI-compatible API. Embeddings run inside the app with ChromaDB's built-in
    MiniLM model, because Groq does not offer embeddings.
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, settings: Settings):
        import os

        from openai import OpenAI
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self.settings = settings
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        self.client = OpenAI(base_url=self.BASE_URL, api_key=self.api_key or "missing")
        self._embedder = DefaultEmbeddingFunction()

    def _create(self, messages: list[Message], stream: bool):
        return self.client.chat.completions.create(
            model=self.settings.groq_model, messages=messages, stream=stream,
            reasoning_effort="low", max_completion_tokens=1024,
        )

    def chat(self, messages: list[Message]) -> str:
        return self._create(messages, stream=False).choices[0].message.content or ""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        for chunk in self._create(messages, stream=True):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._embedder(texts)]

    def status(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "GROQ_API_KEY is not set."
        return True, f"{self.settings.groq_model} on Groq · MiniLM embeddings"


def make_llm(settings: Settings) -> "LLM":
    """Pick the model backend named in the settings."""
    if settings.backend == "groq":
        return HostedLLM(settings)
    return OllamaLLM(settings)
