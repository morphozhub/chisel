"""Replay / recovery of our miner through the organizer's execution boundary.

Modelled on miner-reference/tests/test_replay.py: ``main.generate`` runs under
``miner_runtime.runner.run_generation`` with the live ``Witness`` fixture and a
``RecordingRenderer``; the retained execution becomes a verified bundle, and
``miner_runtime.validator.run_replay`` must reproduce the same final bytes with
``ReplayWitness`` + ``ReplayRenderer`` while consuming every observation.

Run: cd /home/user/172/miner && PYTHONPATH=/home/user/172/miner python -m pytest tests -q
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from miner_runtime.observations import Execution, ReplayMismatch
from miner_runtime.runner import run_generation
from miner_runtime.validator import ReplayWitness, run_replay
from renderer_client import RecordingRenderer, ReplayRenderer
from test_batch import PNG, Calls, Renderer, Witness
from test_batch import no_network  # noqa: F401 - autouse guard, upstream keeps it in conftest.py
from witness_client.calls import canonical
from witness_client.pods import CallView

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402 - the public entry point the sandbox imports

from miner_reference.pipeline import ProviderUnavailable, normalize  # noqa: E402

generate = main.generate
SEED = 42


def source_of(request_id: int) -> str:
    return normalize(Calls().create(request_id=request_id, model="fixture").output_text)


def live_execution(live: Witness) -> bytes:
    """The one non-aborted execution retained by the witness state directory."""
    paths = list(live._state.glob("executions/*/*/execution.json"))
    assert len(paths) == 1
    return paths[0].read_bytes()


def build_bundle(live: Witness, directory: Path) -> Path:
    """Operator-verified bundle (version 2) from the retained execution, as upstream builds it."""
    execution = live_execution(live)
    bundle = directory / "bundle"
    bundle.mkdir()
    captures = {}
    for step in Execution.model_validate_json(execution).observations:
        captured = live.recorded[step.request.id]
        exchange = bundle / captured.exchange_id
        exchange.mkdir(exist_ok=True)
        manifest = canonical(
            {
                "request_key": step.request.model_dump(),
                "call_id": step.request.id,
                "provider_status": 200,
                "response_sha256": hashlib.sha256(captured.body).hexdigest(),
            }
        )
        (exchange / "manifest.json").write_bytes(manifest)
        (exchange / "response.sse").write_bytes(captured.body)
        call = CallView(
            request=step.request,
            exchange_id=captured.exchange_id,
            deadline=2000000000,
            phase="issued",
            authorized=True,
            manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        )
        captures[captured.exchange_id] = {
            "call": call.model_dump(),
            "model": "fixture",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
    pricing = canonical(
        {
            "calls_per_prompt": 20,
            "credits_per_prompt": 1000,
            "long_context_above_input_tokens": 272000,
            "models": {"fixture": {"input_credits_per_token": 1, "output_credits_per_token": 1}},
        }
    )
    (bundle / "execution.json").write_bytes(execution)
    (bundle / "pricing.json").write_bytes(pricing)
    (bundle / "verified.json").write_bytes(
        canonical(
            {
                "version": 2,
                "execution_sha256": hashlib.sha256(execution).hexdigest(),
                "pricing_sha256": hashlib.sha256(pricing).hexdigest(),
                "captures": captures,
            }
        )
    )
    return bundle


def replay_twice(bundle: Path, previews: RecordingRenderer, expected: str | None, image: bytes = PNG) -> None:
    """Replay must give byte-identical output and consume every model and render observation."""
    for _ in range(2):
        replay = ReplayWitness(bundle)
        rendered = ReplayRenderer(previews.records, renderer_id=previews.renderer_id)
        assert run_replay(generate, image, SEED, replay, rendered) == expected
        rendered.assert_consumed()
        assert replay.schedule.done == set(replay.schedule.steps)


def finished(live: Witness, expected: str | None) -> Execution:
    execution = Execution.model_validate_json(live_execution(live))
    assert execution.finished and not execution.aborted
    assert execution.output_is_none == (expected is None)
    assert execution.output_sha256 == hashlib.sha256((expected or "").encode()).hexdigest()
    return execution


def test_happy_path_reproduces_identical_bytes(tmp_path: Path) -> None:
    live, renderer = Witness(), Renderer()
    previews = RecordingRenderer(renderer)
    expected = run_generation(generate, PNG, SEED, live, previews)
    assert expected is not None and expected in renderer.sources
    assert live.paid == 7 and len(renderer.sources) == 5
    assert sorted(request_id for request_id, _ in live.calls.requests) == [1, 2, 3, 4, 5, 6, 7]
    execution = finished(live, expected)
    assert len(execution.observations) == 7 and all(step.action == "create" for step in execution.observations)
    bundle = build_bundle(live, tmp_path)
    replay_twice(bundle, previews, expected)
    assert live.paid == 7 and len(renderer.sources) == 5


def test_repair_path_reproduces(tmp_path: Path) -> None:
    live, renderer = Witness(), Renderer()
    renderer.reject = {source_of(i) for i in (1, 2, 3, 5, 6)}
    previews = RecordingRenderer(renderer)
    expected = run_generation(generate, PNG, SEED, live, previews)
    assert expected in {source_of(i) for i in (8, 9, 10, 11, 12)}
    assert live.paid == 12 and len(renderer.sources) == 10
    assert sum(record.rejection_json is not None for record in previews.records) == 5
    finished(live, expected)
    bundle = build_bundle(live, tmp_path)
    replay_twice(bundle, previews, expected)
    assert live.paid == 12 and len(renderer.sources) == 10


@pytest.mark.parametrize("kind", ["invalid", "incomplete"])
def test_round_one_selector_fallback_reproduces(tmp_path: Path, kind: str) -> None:
    live, renderer = Witness(), Renderer()
    getattr(live.calls, kind).add(4)
    previews = RecordingRenderer(renderer)
    expected = run_generation(generate, PNG, SEED, live, previews)
    # Our deterministic fallback: candidates keep submission order, so Astra's candidate wins.
    assert expected == source_of(1)
    assert live.paid == 4 and len(renderer.sources) == 3
    finished(live, expected)
    bundle = build_bundle(live, tmp_path)
    replay_twice(bundle, previews, expected)
    assert live.paid == 4


@pytest.mark.parametrize("kind", ["invalid", "incomplete"])
def test_final_selector_fallback_keeps_round_one_winner(tmp_path: Path, kind: str) -> None:
    live, renderer = Witness(), Renderer()
    getattr(live.calls, kind).add(7)
    previews = RecordingRenderer(renderer)
    expected = run_generation(generate, PNG, SEED, live, previews)
    assert expected in {source_of(i) for i in (1, 2, 3)}
    assert live.paid == 7 and len(renderer.sources) == 5
    finished(live, expected)
    bundle = build_bundle(live, tmp_path)
    replay_twice(bundle, previews, expected)


def test_rate_limit_aborts_then_a_fresh_execution_recovers(tmp_path: Path) -> None:
    live, renderer = Witness(), Renderer()
    live.provider_status[2] = 429
    with pytest.raises(ProviderUnavailable):
        run_generation(generate, PNG, SEED, live, RecordingRenderer(renderer))
    # The 429 was captured (paid) but is not a result: the execution is aborted, not finished.
    assert live.paid == 3  # the 429 exchange itself was captured and counted
    aborted = Execution.model_validate_json(live_execution(live))
    assert aborted.aborted and not aborted.finished and aborted.output_sha256 is None
    statuses = {step.request.request_id: step.outcome for step in aborted.observations}
    assert statuses == {1: "return", 2: "return", 3: "return"}
    directory = next(live._state.glob("executions/*/*/execution.json")).parent
    assert not list(directory.glob("aborted-*.json"))

    previews = RecordingRenderer(renderer)
    expected = run_generation(generate, PNG, SEED, live, previews)
    assert expected is not None
    # Requests 1 and 3 are reused from capture; only request 2 and the later calls are paid again.
    assert live.paid == 3 + 1 + 4
    retained = list(directory.glob("aborted-*.json"))
    assert len(retained) == 1
    assert Execution.model_validate_json(retained[0].read_bytes()).aborted
    execution = finished(live, expected)
    assert len(execution.observations) == 7
    bundle = build_bundle(live, tmp_path)
    replay_twice(bundle, previews, expected)
    assert live.paid == 8


def test_rate_limit_in_selector_also_aborts(tmp_path: Path) -> None:
    live, renderer = Witness(), Renderer()
    live.provider_status[4] = 429
    with pytest.raises(ProviderUnavailable):
        run_generation(generate, PNG, SEED, live, RecordingRenderer(renderer))
    assert live.paid == 4
    assert Execution.model_validate_json(live_execution(live)).aborted
    previews = RecordingRenderer(renderer)
    expected = run_generation(generate, PNG, SEED, live, previews)
    assert expected is not None and live.paid == 8
    replay_twice(build_bundle(live, tmp_path), previews, expected)


def test_replay_rejects_changed_image_seed_or_evidence(tmp_path: Path) -> None:
    live, renderer = Witness(), Renderer()
    previews = RecordingRenderer(renderer)
    expected = run_generation(generate, PNG, SEED, live, previews)
    bundle = build_bundle(live, tmp_path)

    def rendered() -> ReplayRenderer:
        return ReplayRenderer(previews.records, renderer_id=previews.renderer_id)

    with pytest.raises(ReplayMismatch, match="image or generation seed"):
        run_replay(generate, PNG + b"x", SEED, ReplayWitness(bundle), rendered())
    with pytest.raises(ReplayMismatch, match="image or generation seed"):
        run_replay(generate, PNG, SEED + 1, ReplayWitness(bundle), rendered())
    assert live.paid == 7
    # Still reproducible after the rejected attempts, then a tampered capture is refused.
    replay_twice(bundle, previews, expected)
    exchange = next(path for path in bundle.iterdir() if path.is_dir())
    (exchange / "response.sse").write_bytes(b"data: {}\n\n")
    # ReplayWitness raises "evidence changed" inside Schedule.step, which surfaces it as an outcome mismatch.
    with pytest.raises(ReplayMismatch, match="different outcome|evidence changed"):
        run_replay(generate, PNG, SEED, ReplayWitness(bundle), rendered())
    assert live.paid == 7


def test_main_generate_is_the_pipeline_entry_point(tmp_path: Path) -> None:
    """The sandbox calls main.generate(image, seed, witness, renderer) directly; it must match the boundary run."""
    live, renderer = Witness(), Renderer()
    direct = main.generate(PNG, SEED, live, renderer)
    assert direct is not None and direct in renderer.sources and live.paid == 7
    previews = RecordingRenderer(renderer)
    assert run_generation(main.generate, PNG, SEED, live, previews) == direct
    assert live.paid == 7  # every call was recovered from capture, nothing paid twice
    replay_twice(build_bundle(live, tmp_path), previews, direct)
