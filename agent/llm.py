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
