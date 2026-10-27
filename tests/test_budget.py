"""Budget guard: worst-case reservation (input + max output), parallel calls, unknown usage."""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miner_reference import pipeline  # noqa: E402
from miner_reference.budget import PRICING, Budget, Planned  # noqa: E402
from test_pipeline_stub import JS, IMAGE, SEED, StubCalls, StubRenderer, choice, response  # noqa: E402

SOL_IN = Decimal(str(PRICING["models"]["gpt-6.1-sol"]["input_credits_per_token"]))
SOL_OUT = Decimal(str(PRICING["models"]["gpt-6.1-sol"]["output_credits_per_token"]))


def test_planned_worst_case_covers_input_and_max_output():
    p = Planned("gpt-6.1-sol", text_chars=3200, images=2, max_output_tokens=1000)
    assert p.input_tokens == 1000 + 2 * 1700
    assert p.worst_credits() == SOL_IN * p.input_tokens + SOL_OUT * 1000


def test_parallel_reservation_blocks_an_unaffordable_fourth_call():
    b = Budget(credits_limit=1_000_000, calls_limit=60, safety_credits=0)
    p = Planned("gpt-6.1-sol", 0, 0, 30_000)  # 300,000 credits worst case each
    assert b.affordable([p, p, p])
    b.reserve([p, p, p])  # three calls in flight
    assert not b.affordable([p])  # 1,200,000 > 1,000,000
    b.release([p, p, p])
    b.charge("gpt-6.1-sol", SimpleNamespace(input_tokens=0, output_tokens=1000), p)  # actual 10,000
    assert b.affordable([p, p, p])


def test_unknown_usage_is_charged_at_worst_case_not_zero():
    b = Budget(credits_limit=1_000_000, calls_limit=60, safety_credits=0)
    p = Planned("gpt-6.1-sol", 0, 0, 50_000)  # 500,000 worst case
    b.charge("gpt-6.1-sol", None, p)
    assert b.spent == Decimal(500_000) and b.calls == 1
    assert not b.affordable([p, p])


def test_call_cap_is_enforced():
    b = Budget(credits_limit=10**9, calls_limit=2, safety_credits=0)
    p = Planned("gpt-6.1-sol", 0, 0, 1)
    b.charge("gpt-6.1-sol", SimpleNamespace(input_tokens=1, output_tokens=1), p)
    assert b.affordable([p]) and not b.affordable([p, p])


def test_pipeline_skips_refinements_when_budget_cannot_cover_them(monkeypatch):
    # Limit just above the three initial calls + judge worst case: refinements (5, 6) must be skipped
    # deterministically and the round-1 winner returned.
    astra = pipeline.candidate_plan(pipeline.ASTRA, 1, len(pipeline.STRATEGIES[1]))
    sol = pipeline.candidate_plan(pipeline.SOL, 1, len(pipeline.STRATEGIES[2]))
    judge = Planned(pipeline.SOL, len(pipeline.JUDGE_INSTRUCTIONS) + 200, 4, pipeline.JUDGE_OUTPUT)
    limit = int(astra.worst_credits() + 2 * sol.worst_credits() + 2 * judge.worst_credits()) + 1
    monkeypatch.setattr(pipeline, "Budget", lambda: Budget(credits_limit=limit, calls_limit=60, safety_credits=0))

    def heavy(r, model):  # actual usage close to the worst case: the model's max output really consumed
        r.usage.output_tokens = pipeline.max_output(model)
        r.usage.input_tokens = 14000
        r.usage.total_tokens = r.usage.input_tokens + r.usage.output_tokens
        return r

    plan = {1: heavy(response(JS + "//a"), pipeline.ASTRA), 2: heavy(response(JS + "//b"), pipeline.SOL),
            3: heavy(response(JS + "//c"), pipeline.SOL), 4: choice("A")}
    calls = StubCalls(plan)
    out = pipeline.run(IMAGE, SEED, calls, StubRenderer())
    assert out in {pipeline.normalize(JS + s) for s in ("//a", "//b", "//c")}
    assert sorted(r for r, _ in calls.log) == [1, 2, 3, 4]


def test_pipeline_returns_none_when_even_initial_calls_are_unaffordable(monkeypatch):
    monkeypatch.setattr(pipeline, "Budget", lambda: Budget(credits_limit=1000, calls_limit=60, safety_credits=0))
    calls = StubCalls({})
    assert pipeline.run(IMAGE, SEED, calls, StubRenderer()) is None
    assert calls.log == []


def test_worst_case_of_full_path_is_bounded_by_the_guard():
    """Documentation test: the printed bounds used in the plan come from the same code."""
    astra = pipeline.candidate_plan(pipeline.ASTRA, 1)
    sol = pipeline.candidate_plan(pipeline.SOL, 1)
    judge = Planned(pipeline.SOL, len(pipeline.JUDGE_INSTRUCTIONS) + 200, 4, pipeline.JUDGE_OUTPUT)
    refine = pipeline.candidate_plan(pipeline.SOL, 2, 8000)
    happy = astra.worst_credits() + 2 * sol.worst_credits() + 2 * judge.worst_credits() + 2 * refine.worst_credits()
    assert happy < Budget().credits_limit  # happy path always fits
    worst_full = happy + 5 * pipeline.candidate_plan(pipeline.SOL, 1, 9000).worst_credits()
    assert worst_full <= 5_000_000  # with the Astra cap even the maximal repair path stays nominal
    print(json.dumps({"happy_worst": int(happy), "max_path_worst": int(worst_full), "limit": int(Budget().credits_limit)}))
