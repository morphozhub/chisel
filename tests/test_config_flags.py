"""EXTRA_VIEWS and JUDGE_SWAP behaviour (deterministic, replay-safe)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miner_reference import config, pipeline  # noqa: E402
from test_pipeline_stub import IMAGE, JS, SEED, StubCalls, StubRenderer, choice, response  # noqa: E402


def plan_happy(final_label="A"):
    return {1: response(JS + "//a"), 2: response(JS + "//b"), 3: response(JS + "//c"), 4: choice("A"),
            5: response(JS + "//d"), 6: response(JS + "//e"), 7: choice(final_label)}


def test_extra_views_double_renders_and_reach_the_judge(monkeypatch):
    monkeypatch.setattr(config, "EXTRA_VIEWS", True)
    renderer = StubRenderer()
    seen = {}

    class Calls(StubCalls):
        def create(self, *, request_id=0, **params):
            if request_id in (4, 7):
                content = params["input"][0]["content"]
                seen[request_id] = sum(1 for c in content if c.get("type") == "input_image")
            return super().create(request_id=request_id, **params)

    out = pipeline.run(IMAGE, SEED, Calls(plan_happy()), renderer)
    assert out is not None
    assert len(renderer.log) == 10  # 5 grids + 5 side/back/top views
    assert seen[4] == 1 + 3 * 2  # target + (grid + views) per candidate
    assert seen[7] == 1 + 3 * 2


def test_extra_views_are_deterministic_across_runs(monkeypatch):
    monkeypatch.setattr(config, "EXTRA_VIEWS", True)
    outs, logs = [], []
    for _ in range(2):
        r = StubRenderer()
        outs.append(pipeline.run(IMAGE, SEED, StubCalls(plan_happy()), r))
        logs.append(r.log)
    assert outs[0] == outs[1] and logs[0] == logs[1]


def test_judge_swap_agreement_uses_ids_13_and_14(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_SWAP", True)
    plan = plan_happy()
    # The reversed-label judge must name the same candidate: label mapping is reversed, so
    # compute which label the first judge's winner gets in the reversed ordering.
    calls = StubCalls(plan)

    def swapped(request_id, labels_count=3):
        return chr(65 + labels_count - 1 - (ord("A") - 65))  # 'A' in reversed order of 3 -> 'C'

    plan[13] = choice(swapped(13))
    plan[14] = choice("C")  # final: 3 candidates (winner + 2 refinements); 'A' reversed -> 'C'
    out = pipeline.run(IMAGE, SEED, calls, StubRenderer())
    assert out is not None
    ids = sorted(r for r, _ in calls.log)
    assert ids == [1, 2, 3, 4, 5, 6, 7, 13, 14]


def test_judge_swap_disagreement_keeps_incumbent(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_SWAP", True)
    plan = plan_happy()
    plan[13] = choice("A")  # reversed order: 'A' is a different candidate than first judge's 'A'
    plan[14] = choice("A")
    calls = StubCalls(plan)
    out = pipeline.run(IMAGE, SEED, calls, StubRenderer())
    # Round 1 disagreement -> incumbent = candidates[0] (Astra); the refinements still run on it,
    # final disagreement -> keep the round-1 winner (incumbent) => output is the Astra source.
    assert out == pipeline.normalize(JS + "//a")


def test_fourth_candidate_uses_ids_15_and_16(monkeypatch):
    monkeypatch.setattr(config, "FOURTH_CANDIDATE", True)
    plan = plan_happy()
    plan[15] = response(JS + "//f")
    calls = StubCalls(plan)
    out = pipeline.run(IMAGE, SEED, calls, StubRenderer())
    assert out is not None
    assert sorted(r for r, _ in calls.log) == [1, 2, 3, 4, 5, 6, 7, 15]


def test_fourth_candidate_repair_uses_id_16(monkeypatch):
    monkeypatch.setattr(config, "FOURTH_CANDIDATE", True)
    plan = plan_happy()
    plan[15] = response(JS + "//bad4")
    plan[16] = response(JS + "//fixed4")
    calls = StubCalls(plan)
    out = pipeline.run(IMAGE, SEED, calls, StubRenderer(reject={"//bad4"}))
    assert out is not None and (16, pipeline.SOL) in calls.log
