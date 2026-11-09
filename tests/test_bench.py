"""Runs the benchmark harness end to end with the fake model, so it can't silently break."""

import json

from agent import Settings
from bench.run import SCENARIOS, mentions, run_benchmark, to_markdown
from tests.conftest import FakeLLM


def test_keyword_matching():
    assert mentions("The user runs marathons.", "marathon")
    assert not mentions("The user's name is John.", "jon")
    assert mentions("The user's name is Jon.", "jon")


def test_scenarios_file_is_well_formed():
    scenarios = json.loads(SCENARIOS.read_text())
    assert len(scenarios) >= 10
    for s in scenarios:
        assert s["type"] in ("recall", "update") and s["messages"] and s["probes"]
        for p in s["probes"]:
            assert p["question"] and p["expect"]
        if s["type"] == "update":
            assert any(p.get("stale") for p in s["probes"])


def test_benchmark_runs_and_fact_updating_removes_stale_facts(tmp_path):
    moved = [{"name": "Moved", "type": "update",
              "messages": ["I live in Chicago.", "I moved to Seattle."],
              "probes": [{"question": "Where do I live?", "expect": ["seattle"], "stale": ["chicago"]}]}]
    settings = Settings(data_dir=tmp_path)
    settings.max_distance = 1.01  # the fake word-count embeddings barely overlap; let every fact through
    results = run_benchmark(lambda s: FakeLLM(), settings=settings, scenarios=moved, verbose=False)
    assert results["Without fact updating"]["total"].stale == [True]
    assert results["With fact updating"]["total"].stale == [False]
    assert results["With fact updating"]["total"].recall == [True]
    table = to_markdown(results, "fake", 1)
    assert "| With fact updating | 100% | 0% |" in table
