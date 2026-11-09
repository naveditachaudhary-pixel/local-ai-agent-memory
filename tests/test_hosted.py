"""Checks the hosted (online demo) backend against a mock of Groq's OpenAI-compatible API."""

import json

import httpx
import pytest
from openai import OpenAI

import chromadb.utils.embedding_functions as ef
from agent import Settings
from agent.llm import HostedLLM, OllamaLLM, make_llm
from tests.conftest import bag_of_words


class FakeEmbedder:
    def __call__(self, texts):
        return [bag_of_words(t) for t in texts]


def sse(*events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def chunk(text):
    return {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(ef, "DefaultEmbeddingFunction", FakeEmbedder)
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append((request, body))
        if body.get("stream"):
            text = sse(chunk("Hello "), chunk("there."),
                       {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": []})
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '["The user is a nurse."]'}}]})

    llm = HostedLLM(Settings(backend="groq"))
    llm.client = OpenAI(base_url=HostedLLM.BASE_URL, api_key="test-key",
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return llm, seen


def test_chat_sends_model_and_parses_reply(hosted):
    llm, seen = hosted
    assert llm.chat([{"role": "user", "content": "hi"}]) == '["The user is a nurse."]'
    request, body = seen[0]
    assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["reasoning_effort"] == "low"


def test_stream_yields_text_and_skips_empty_chunks(hosted):
    llm, _ = hosted
    assert "".join(llm.stream([{"role": "user", "content": "hi"}])) == "Hello there."


def test_status_and_backend_choice(hosted, monkeypatch):
    llm, _ = hosted
    assert llm.status()[0]
    monkeypatch.delenv("GROQ_API_KEY")
    assert not HostedLLM(Settings(backend="groq")).status()[0]
    assert isinstance(make_llm(Settings(backend="groq")), HostedLLM)
    assert isinstance(make_llm(Settings(backend="ollama")), OllamaLLM)
