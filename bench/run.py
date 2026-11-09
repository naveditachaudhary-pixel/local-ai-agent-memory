"""Memory benchmark: does the agent remember the right things, and forget the outdated ones?

Each scenario is a short conversation. Then the agent starts a new session (the chat is cleared,
long-term memory is kept) and answers questions about it. For every question we check:

  * recall    - is the right fact among the memories pulled into the prompt?
  * stale     - is an outdated fact among them (for example the city the user moved away from)?
  * answer    - does the reply contain the right answer and not the outdated one?

Everything runs twice: with fact updating switched off and on, so the table shows what it changes.

    python -m bench                      # local Ollama models from your settings
    python -m bench --update-readme      # also refresh the results table in README.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Callable

from agent import Agent, Settings
from agent.llm import LLM, make_llm
from agent.memory import FACT

ROOT = Path(__file__).resolve().parent
SCENARIOS = ROOT / "scenarios.json"
README = ROOT.parent / "README.md"
START, END = "<!-- bench:start -->", "<!-- bench:end -->"


def mentions(text: str, keyword: str) -> bool:
    """Whole-word-start match, so 'marathon' finds 'marathons' but 'jon' doesn't find 'john'."""
    return re.search(rf"\b{re.escape(keyword.lower())}", text.lower()) is not None


@dataclass
class Score:
    recall: list[bool] = field(default_factory=list)
    stale: list[bool] = field(default_factory=list)   # only for questions that have an outdated answer
    answer: list[bool] = field(default_factory=list)

    @staticmethod
    def pct(xs: list[bool]) -> str:
        return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "–"


def run_scenario(scn: dict, make_agent: Callable[[], Agent], details: list[str]) -> Score:
    agent = make_agent()
    for message in scn["messages"]:
        agent.respond(message)
    agent.new_session()   # the questions come in a fresh session, so only long-term memory can help

    score = Score()
    for probe in scn["probes"]:
        q, expect, stale = probe["question"], probe["expect"], probe.get("stale", [])
        messages = agent.build_messages(q)
        recalled = " | ".join(m.text for m in agent.last_recalled if m.kind == FACT)
        reply = agent.llm.chat(messages)

        hit = any(all(mentions(m.text, k) for k in expect) for m in agent.last_recalled if m.kind == FACT)
        stale_hit = any(mentions(m.text, k) for m in agent.last_recalled if m.kind == FACT for k in stale)
        correct = all(mentions(reply, k) for k in expect) and not any(mentions(reply, k) for k in stale)
        score.recall.append(hit)
        score.answer.append(correct)
        if stale:
            score.stale.append(stale_hit)
        details.append(f"| {scn['name']} | {q} | {'✓' if hit else '✗'} | "
                       f"{('✗ stale' if stale_hit else '✓') if stale else '–'} | {'✓' if correct else '✗'} |")
    return score


def run_benchmark(make_llm_fn: Callable[[Settings], LLM] = make_llm, settings: Settings | None = None,
                  scenarios: list[dict] | None = None, verbose: bool = True) -> dict:
    base = settings or Settings()
    scenarios = scenarios if scenarios is not None else json.loads(SCENARIOS.read_text())
    results = {}
    for label, updating in (("Without fact updating", False), ("With fact updating", True)):
        totals, by_type, details = Score(), {}, []
        started = time.time()
        for scn in scenarios:
            def make_agent() -> Agent:
                s = Settings(**{**base.__dict__, "data_dir": Path(tempfile.mkdtemp(prefix="bench-")),
                                "update_facts": updating})
                return Agent(settings=s, llm=make_llm_fn(s))
            sc = run_scenario(scn, make_agent, details)
            for target in (totals, by_type.setdefault(scn["type"], Score())):
                target.recall += sc.recall; target.stale += sc.stale; target.answer += sc.answer
            if verbose:
                print(f"  [{label}] {scn['name']}: recall {Score.pct(sc.recall)}, answers {Score.pct(sc.answer)}")
        results[label] = {"total": totals, "by_type": by_type, "details": details, "seconds": time.time() - started}
    return results


def to_markdown(results: dict, model_note: str, n_scenarios: int) -> str:
    n_questions = len(next(iter(results.values()))["total"].recall)
    lines = [
        f"{n_scenarios} scenarios, {n_questions} questions, asked in a new session after each conversation. "
        f"Model: {model_note}. Run on {date.today():%B %d, %Y}.",
        "",
        "| | Right fact recalled | Outdated fact recalled (lower is better) | Correct answer |",
        "|---|---|---|---|",
    ]
    for label, r in results.items():
        t = r["total"]
        lines.append(f"| {label} | {Score.pct(t.recall)} | {Score.pct(t.stale)} | {Score.pct(t.answer)} |")
    lines += ["", "<details><summary>Per question, with fact updating</summary>", "",
              "| Scenario | Question | Recalled | Stale | Answer |", "|---|---|---|---|---|",
              *results["With fact updating"]["details"], "", "</details>"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "results.md"), help="where to write the results table")
    parser.add_argument("--update-readme", action="store_true", help="also replace the results table in README.md")
    args = parser.parse_args()

    settings = Settings()
    llm = make_llm(settings)
    ok, status = llm.status() if hasattr(llm, "status") else (True, "custom")
    if not ok:
        raise SystemExit(status)
    model_note = (f"`{settings.groq_model}` on Groq" if settings.backend == "groq"
                  else f"`{settings.chat_model}` with `{settings.embed_model}` embeddings, via Ollama")

    scenarios = json.loads(SCENARIOS.read_text())
    print(f"Running {len(scenarios)} scenarios twice with {status} …")
    results = run_benchmark(settings=settings, scenarios=scenarios)
    table = to_markdown(results, model_note, len(scenarios))
    Path(args.out).write_text(table + "\n")
    print("\n" + table.split("<details>")[0])
    print(f"Saved to {args.out}")

    if args.update_readme:
        text = README.read_text()
        if START not in text or END not in text:
            raise SystemExit("README.md has no benchmark markers to replace.")
        before, rest = text.split(START, 1)
        after = rest.split(END, 1)[1]
        README.write_text(f"{before}{START}\n{table}\n{END}{after}")
        print("README.md updated.")


if __name__ == "__main__":
    main()
