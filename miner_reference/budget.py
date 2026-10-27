"""Deterministic per-prompt budget guard evaluated before every dispatch.

The contest charges, at replay, every call consumed by the prompt's execution:
``input_tokens × in_rate + output_tokens × out_rate`` from the frozen pricing file,
with cached input at full price and reasoning already inside ``output_tokens``.
Exceeding 5,000,000 credits or 60 calls rejects the whole prompt, and there is no
live ledger, so the miner has to stay inside the limit on its own.

This guard only uses committed data (pricing.json), the request we are about to
send and usage reported by earlier responses of the same prompt. It therefore
makes the same decisions during generation and during replay.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

PRICING = json.loads(Path(__file__).resolve().parents[1].joinpath("budgets", "pricing.json").read_text())

# Conservative input-token estimate: ~1 token per 3.2 characters of text and a
# flat allowance per image (a 1024² "auto" image costs ≈1.1k tokens; the 4-view
# grid is about the same). Over-estimating only makes the guard more cautious.
CHARS_PER_TOKEN = Decimal("3.2")
IMAGE_TOKENS = 1700


@dataclass(frozen=True)
class Planned:
    model: str
    text_chars: int
    images: int
    max_output_tokens: int

    @property
    def input_tokens(self) -> int:
        return int(Decimal(self.text_chars) / CHARS_PER_TOKEN) + self.images * IMAGE_TOKENS

    def worst_credits(self) -> Decimal:
        rate = PRICING["models"][self.model]
        prefix = "long_" if self.input_tokens > int(PRICING["long_context_above_input_tokens"]) else ""
        return Decimal(str(rate[f"{prefix}input_credits_per_token"])) * self.input_tokens + Decimal(
            str(rate[f"{prefix}output_credits_per_token"])
        ) * self.max_output_tokens


class BudgetExhausted(RuntimeError):
    """Raised only for the mandatory path; optional stages are skipped instead."""


class Budget:
    def __init__(
        self,
        credits_limit: int = int(PRICING["credits_per_prompt"]),
        calls_limit: int = int(PRICING["calls_per_prompt"]),
        safety_credits: int = 150_000,
    ) -> None:
        self.credits_limit = Decimal(credits_limit) - Decimal(safety_credits)
        self.calls_limit = calls_limit
        self.spent = Decimal(0)
        self.calls = 0
        self.reserved = Decimal(0)
        self.lock = threading.Lock()

    def charge(self, model: str, usage: object, planned: Planned | None = None) -> None:
        """Record usage of a dispatched call.

        ``usage`` is the provider-reported usage object. When it is missing (HTTP
        failure, incomplete delivery) the call is charged at its planned worst case:
        unknown usage is never treated as zero.
        """
        with self.lock:
            self.calls += 1
            if usage is None:
                self.spent += planned.worst_credits() if planned is not None else Decimal(0)
                return
            inp = int(getattr(usage, "input_tokens", 0) or 0)
            out = int(getattr(usage, "output_tokens", 0) or 0)
            rate = PRICING["models"][model]
            prefix = "long_" if inp > int(PRICING["long_context_above_input_tokens"]) else ""
            self.spent += Decimal(str(rate[f"{prefix}input_credits_per_token"])) * inp + Decimal(
                str(rate[f"{prefix}output_credits_per_token"])
            ) * out

    def affordable(self, planned: list[Planned], keep: Decimal = Decimal(0)) -> bool:
        """Can every call in ``planned`` run in the worst case while ``keep`` credits stay free?"""
        with self.lock:
            need = sum((p.worst_credits() for p in planned), Decimal(0))
            return (
                self.calls + len(planned) <= self.calls_limit
                and self.spent + self.reserved + need + keep <= self.credits_limit
            )

    def reserve(self, planned: list[Planned]) -> None:
        with self.lock:
            self.reserved += sum((p.worst_credits() for p in planned), Decimal(0))

    def release(self, planned: list[Planned]) -> None:
        with self.lock:
            self.reserved -= sum((p.worst_credits() for p in planned), Decimal(0))
            if self.reserved < 0:
                self.reserved = Decimal(0)
