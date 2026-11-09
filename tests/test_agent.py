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
