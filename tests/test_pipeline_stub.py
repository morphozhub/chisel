"""Offline checks of pipeline control flow with stubbed witness/renderer.

Run: PYTHONPATH=. python -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest
from openai import RateLimitError, BadRequestError
from openai.types.responses import Response
from renderer_client import RenderRejected

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miner_reference import pipeline  # noqa: E402

IMAGE = b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes"
SEED = 1145844865
JS = "export default function generate(THREE) { return new THREE.Group(); }"


def response(text: str, status: str = "completed") -> Response:
    return Response.model_validate(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-6.1-sol",
            "status": status,
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )


class StubCalls:
    """Deterministic canned responses keyed by request_id; records the call sequence."""

    def __init__(self, plan: dict[int, object]) -> None:
        self.plan, self.log = plan, []

    def create(self, *, request_id: int = 0, **params):
        self.log.append((request_id, params["model"]))
        item = self.plan[request_id]
        if isinstance(item, Exception):
            raise item
        return item


class StubRenderer:
    renderer_id = "sha256:stub"

    def __init__(self, reject: set[str] = frozenset()) -> None:
        self.reject, self.log = reject, []

    def render(self, source: str, options) -> bytes:
        self.log.append(source)
        if any(marker in source for marker in self.reject):
            raise RenderRejected(json.dumps({"error": "validation failed", "failures": [{"rule": "BAD"}]}))
        return b"png:" + source.encode()[:16]


def http_error(cls, code):
    req = httpx.Request("POST", "https://x")
    return cls("m", response=httpx.Response(code, request=req), body=None)


def choice(label: str) -> Response:
    return response(json.dumps({"winner": label, "reason": "r", "discrepancies": [{"part": "body", "issue": "too tall", "fix": "scale y 0.8"}], "improvements": "i"}))


def test_happy_path_is_deterministic():
    plan = {1: response(JS + "//a"), 2: response(JS + "//b"), 3: response(JS + "//c"),
            4: choice("A"), 5: response(JS + "//d"), 6: response(JS + "//e"), 7: choice("B")}
    outs, logs = [], []
    for _ in range(2):
        calls, renderer = StubCalls(plan), StubRenderer()
        outs.append(pipeline.run(IMAGE, SEED, calls, renderer))
        logs.append((calls.log, renderer.log))
    assert outs[0] == outs[1] and outs[0] is not None
    # Concurrent calls inside a stage may *start* in any thread order (allowed by replay);
    # the set of calls and the render consumption order must be identical.
    assert sorted(logs[0][0]) == sorted(logs[1][0])
    assert logs[0][1] == logs[1][1]
    assert sorted(r for r, _ in logs[0][0]) == [1, 2, 3, 4, 5, 6, 7]


def test_rate_limit_aborts_prompt():
    plan = {1: response(JS), 2: http_error(RateLimitError, 429), 3: response(JS)}
    with pytest.raises(pipeline.ProviderUnavailable):
        pipeline.run(IMAGE, SEED, StubCalls(plan), StubRenderer())


def test_bad_request_is_a_failed_candidate_not_abort():
    plan = {1: http_error(BadRequestError, 400), 2: response(JS + "//b"), 3: response(JS + "//c"),
            4: choice("A"), 5: response(JS + "//d"), 6: response(JS + "//e"), 7: choice("A")}
    assert pipeline.run(IMAGE, SEED, StubCalls(plan), StubRenderer()) is not None


def test_selector_failure_falls_back_to_first_candidate():
    plan = {1: response(JS + "//a"), 2: response(JS + "//b"), 3: response(JS + "//c"),
            4: http_error(BadRequestError, 400)}
    out = pipeline.run(IMAGE, SEED, StubCalls(plan), StubRenderer())
    assert out == pipeline.normalize(JS + "//a")


def test_final_selector_failure_keeps_round1_winner():
    plan = {1: response(JS + "//a"), 2: response(JS + "//b"), 3: response(JS + "//c"),
            4: choice("A"), 5: response(JS + "//d"), 6: response(JS + "//e"), 7: response("not json")}
    calls = StubCalls(plan)
    out = pipeline.run(IMAGE, SEED, calls, StubRenderer())
    # winner label A maps to a shuffled candidate; it must be one of the round-1 sources
    assert out in {pipeline.normalize(JS + s) for s in ("//a", "//b", "//c")}


def test_render_rejection_triggers_one_repair_with_fixed_ids():
    plan = {1: response(JS + "//bad"), 2: response(JS + "//b"), 3: response(JS + "//c"),
            8: response(JS + "//fixed"), 4: choice("A"), 5: response(JS + "//d"), 6: response(JS + "//e"), 7: choice("A")}
    calls = StubCalls(plan)
    out = pipeline.run(IMAGE, SEED, calls, StubRenderer(reject={"//bad"}))
    assert out is not None
    assert (8, pipeline.SOL) in calls.log


def test_all_candidates_fail_returns_none():
    plan = {1: http_error(BadRequestError, 400), 2: http_error(BadRequestError, 400), 3: http_error(BadRequestError, 400)}
    assert pipeline.run(IMAGE, SEED, StubCalls(plan), StubRenderer()) is None
