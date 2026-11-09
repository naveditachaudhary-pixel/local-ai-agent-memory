"""Runs the real Streamlit app with the fake model and checks one full chat turn."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import agent.agent as agent_module
from tests.conftest import FakeLLM

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class FakeOllama(FakeLLM):
    def __init__(self, settings):
        super().__init__()

    def status(self):
        return True, "fake-model · fake-embed"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(agent_module, "OllamaLLM", FakeOllama)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def test_chat_turn_and_learned_fact_shows_in_sidebar(app):
    assert not app.exception
    app.chat_input[0].set_value("Hi, my name is Priya.").run()
    assert not app.exception
    texts = " ".join(m.value for m in app.sidebar.markdown)
    assert "The user's name is Priya." in texts
    assert any("Noted." in md.value for md in app.markdown)
