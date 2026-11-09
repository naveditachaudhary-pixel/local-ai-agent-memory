"""Settings for the agent. Every value can be overridden with an environment variable."""

from dataclasses import dataclass, field
import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # Ollama
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    chat_model: str = field(default_factory=lambda: _env("AGENT_CHAT_MODEL", "llama3.2"))
    embed_model: str = field(default_factory=lambda: _env("AGENT_EMBED_MODEL", "nomic-embed-text"))

    # Where memory lives on disk
    data_dir: Path = field(default_factory=lambda: Path(_env("AGENT_DATA_DIR", "data")))

    # Retrieval
    facts_k: int = 5              # how many facts to pull into each prompt
    exchanges_k: int = 3          # how many past exchanges to pull into each prompt
    max_distance: float = field(default_factory=lambda: float(_env("AGENT_MAX_DISTANCE", "0.55")))  # ignore memories less similar than this (cosine distance)
    duplicate_distance: float = 0.12  # a new fact this close to an old one is treated as a duplicate

    # Short-term context and summaries
    recent_turns: int = 6         # user+assistant pairs kept verbatim in the prompt
    summarize_every: int = 4      # refresh the rolling summary every N turns

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def summary_file(self) -> Path:
        return self.data_dir / "summary.json"
