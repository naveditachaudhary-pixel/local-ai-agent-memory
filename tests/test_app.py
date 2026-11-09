"""Runs the real Streamlit app with a fake model, locally and in online-demo mode."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import agent.llm as llm_module
from tests.conftest import FakeLLM

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class FakeBackend(FakeLLM):
    def __init__(self, settings):
        super().__init__()

    def status(self):
        return True, "fake-model · fake-embed"


def sidebar_text(at):
    return " ".join(m.value for m in at.sidebar.markdown)


@pytest.fixture
def local_app(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "OllamaLLM", FakeBackend)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


@pytest.fixture
def demo_app(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MAX_DISTANCE", "0.9")
    monkeypatch.setattr(llm_module, "HostedLLM", FakeBackend)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def test_chat_turn_and_learned_fact_shows_in_sidebar(local_app):
    assert not local_app.exception
    local_app.chat_input[0].set_value("Hi, my name is Priya.").run()
    assert not local_app.exception
    assert "The user's name is Priya." in sidebar_text(local_app)
    assert any("Noted." in md.value for md in local_app.markdown)
    assert not local_app.info  # no demo banner when running locally


def test_demo_new_session_keeps_memory(demo_app):
    at = demo_app
    assert not at.exception
    assert "Online demo" in at.info[0].value
    at.chat_input[0].set_value("Hi, my name is Priya and I work as a nurse.").run()
    assert "The user's name is Priya." in sidebar_text(at)

    at.sidebar.button[0].click().run()  # "Start a new session"
    assert not at.exception
    assert not at.chat_message            # chat cleared
    assert "New session started" in at.success[0].value
    assert "The user's name is Priya." in sidebar_text(at)  # memory kept

    at.chat_input[0].set_value("What is my name?").run()
    assert any("Recalled:" in c.value and "Priya" in c.value for c in at.caption)


def test_demo_message_limit(demo_app):
    at = demo_app
    at.session_state.sent = 25
    at.run()
    assert "message limit" in at.warning[0].value
