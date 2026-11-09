from agent.agent import parse_fact_list
from agent.memory import EXCHANGE, FACT


def test_learns_a_fact_and_recalls_it_in_a_new_session(make_agent):
    first = make_agent()
    first.respond("Hi! My name is Priya. I work as a data analyst.")
    assert "The user's name is Priya." in first.last_learned

    # A brand-new agent pointed at the same folder still remembers.
    second = make_agent()
    recalled = second.recall("what is my name")
    assert any("Priya" in m.text for m in recalled if m.kind == FACT)


def test_recalled_memories_are_injected_into_the_prompt(make_agent):
    agent = make_agent()
    agent.respond("My name is Priya.")
    messages = agent.build_messages("Do you remember my name?")
    assert "Priya" in messages[0]["content"]


def test_duplicate_facts_are_not_stored_twice(make_agent):
    agent = make_agent()
    agent.respond("My name is Priya.")
    agent.respond("Again, my name is Priya.")
    facts = [m.text for m in agent.memory.all(FACT)]
    assert facts.count("The user's name is Priya.") == 1


def test_every_turn_is_saved_as_an_exchange(make_agent):
    agent = make_agent()
    agent.respond("Tell me a fun fact.")
    agent.respond("Another one please.")
    assert agent.memory.count(EXCHANGE) == 2


def test_summary_refreshes_on_schedule_and_persists(make_agent):
    agent = make_agent()                     # summarize_every = 2 in the fixture
    agent.respond("Hello there.")
    assert agent.summaries.summary == ""
    agent.respond("I work as a teacher.")
    assert "work" in agent.summaries.summary
    assert make_agent().summaries.summary == agent.summaries.summary


def test_streaming_updates_memory_after_the_reply(make_agent):
    agent = make_agent()
    reply = "".join(agent.stream("My name is Sam."))
    assert reply.strip() == "Noted."
    assert any("Sam" in m.text for m in agent.memory.all(FACT))


def test_forget_and_reset(make_agent):
    agent = make_agent()
    agent.respond("My name is Priya. I work as a nurse.")
    fact = agent.memory.all(FACT)[0]
    agent.memory.forget(fact.id)
    assert fact.id not in [m.id for m in agent.memory.all(FACT)]

    agent.reset_memory()
    assert agent.memory.count() == 0
    assert agent.summaries.summary == "" and agent.summaries.turns == 0


def test_parse_fact_list_is_forgiving():
    assert parse_fact_list('Sure! ["The user likes tea."]') == ["The user likes tea."]
    assert parse_fact_list("[]") == []
    assert parse_fact_list("no json here") == []
    assert parse_fact_list('["ok", 3, "  "]') == ["ok"]
    assert parse_fact_list("[not valid json]") == []


# ---------- keeping facts current ----------

def test_new_fact_replaces_the_one_it_makes_outdated(make_agent):
    agent = make_agent()
    agent.respond("My name is Priya. I live in Cincinnati.")
    agent.respond("Big news: I moved to Austin.")
    facts = [m.text for m in agent.memory.all(FACT)]
    assert "The user lives in Austin." in facts
    assert "The user lives in Cincinnati." not in facts
    assert "The user's name is Priya." in facts            # unrelated facts are untouched
    assert agent.last_updated == [("The user lives in Cincinnati.", "The user lives in Austin.")]


def test_fact_updating_can_be_switched_off(make_agent):
    agent = make_agent()
    agent.settings.update_facts = False
    agent.respond("I live in Cincinnati.")
    agent.respond("I moved to Austin.")
    facts = [m.text for m in agent.memory.all(FACT)]
    assert {"The user lives in Cincinnati.", "The user lives in Austin."} <= set(facts)


def test_parse_number_list_is_forgiving():
    from agent.agent import parse_number_list
    assert parse_number_list("Sure! [2, 1]", 3) == [2, 1]
    assert parse_number_list('["3", 3, 9, 0]', 3) == [3]   # out of range and repeats dropped
    assert parse_number_list("none of them", 3) == []
    assert parse_number_list("[oops", 3) == []


# ---------- reinforcement and decay ----------

def test_recalled_memories_are_reinforced(make_agent):
    agent = make_agent()
    agent.respond("My name is Priya.")
    agent.build_messages("what is my name")
    fact = next(m for m in agent.memory.all(FACT) if "Priya" in m.text)
    assert fact.uses == 1 and fact.last_used


def test_rank_score_prefers_useful_and_recent_memories():
    from datetime import datetime, timedelta, timezone
    from agent.memory import rank_score
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fresh = (now - timedelta(days=1)).isoformat()
    stale = (now - timedelta(days=365)).isoformat()
    # Same similarity: a memory used often beats one never used, a recent one beats a forgotten one.
    assert rank_score(0.3, 8, fresh, fresh, now=now) < rank_score(0.3, 0, fresh, fresh, now=now)
    assert rank_score(0.3, 0, None, fresh, now=now) < rank_score(0.3, 0, None, stale, now=now)
    # But a much closer match still wins: decay and reinforcement only nudge the order.
    assert rank_score(0.10, 0, None, stale, now=now) < rank_score(0.40, 10, fresh, fresh, now=now)


# ---------- export / import ----------

def test_export_and_import_round_trip(make_agent, tmp_path):
    import json
    from agent import Agent, Settings
    from tests.conftest import FakeLLM

    source = make_agent()
    source.respond("My name is Priya. I live in Cincinnati.")
    exported = json.loads(json.dumps(source.export_memory()))   # must survive a trip through a JSON file
    assert exported["format"] == "local-ai-agent-memory"

    other = Agent(settings=Settings(data_dir=tmp_path / "other"), llm=FakeLLM())
    added = other.import_memory(exported)
    assert added == source.memory.count()
    assert {m.text for m in other.memory.all(FACT)} == {m.text for m in source.memory.all(FACT)}
    assert other.import_memory(exported) == 0                    # importing twice adds nothing


def test_import_rejects_other_files(make_agent):
    import pytest
    with pytest.raises(ValueError):
        make_agent().import_memory({"hello": "world"})
