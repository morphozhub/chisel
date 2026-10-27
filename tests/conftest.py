"""Shared test setup: upstream-compatible render/call counts unless a test opts in to the extras."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miner_reference import config  # noqa: E402


@pytest.fixture(autouse=True)
def _upstream_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    # Upstream fixtures count 5 renders / 7 calls on the happy path; the extra judge views
    # and the swapped double judge are exercised by tests/test_config_flags.py explicitly.
    monkeypatch.setattr(config, "EXTRA_VIEWS", False)
    monkeypatch.setattr(config, "JUDGE_SWAP", False)
