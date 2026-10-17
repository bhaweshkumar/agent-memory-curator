"""Reusable Decider-style decision pattern: score (question, context, option) -> Decision.

This module is intentionally free of any Bedrock/Strands/AWS imports so the
pattern itself (the typed question/option set + scoring interface) can be
dropped into any agent, regardless of which generative backend it pairs with.

Usage note (for other builders):
    from decider_client import DeciderClient, MockDeciderClient, Decision

    decider: DeciderClient = MockDeciderClient()
    decision = decider.score(
        question="How do I retry a failed network call?",
        context="We fixed a bug where retryOperation() swallowed the last exception.",
        option="relevant",
    )
    if decision.confidence >= 0.6:
        # use this piece of context
        ...

Swap MockDeciderClient for a real Strands Decider 2B-backed implementation by
implementing the same DeciderClient Protocol -- no caller code needs to change.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

# Simulated per-call latency (seconds) for the mock scorer, matching the
# ~115ms latency Strands Decider 2B is advertised to run at.
SIMULATED_LATENCY_SECONDS = 0.115

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Decision:
    """A single Decider decision: a label plus a calibrated confidence."""

    label: str
    confidence: float


class DeciderClient(Protocol):
    """Typed decision interface: (question, context, option) -> Decision.

    Any backend -- mock heuristic, local model, or real Strands Decider 2B --
    can implement this Protocol and be used interchangeably by callers.
    """

    def score(self, question: str, context: str, option: str) -> Decision:
        """Score how well `option` applies to `context` given `question`."""
        ...


class MockDeciderClient:
    """Deterministic keyword/token-overlap scorer standing in for Decider 2B.

    Same (question, context, option) inputs always produce the same
    confidence. Simulates Decider 2B's ~115ms latency per call.
    """

    def score(self, question: str, context: str, option: str) -> Decision:
        time.sleep(SIMULATED_LATENCY_SECONDS)

        question_tokens = set(_WORD_RE.findall(question.lower()))
        context_tokens = set(_WORD_RE.findall(context.lower()))

        if not question_tokens or not context_tokens:
            confidence = 0.0
        else:
            overlap = question_tokens & context_tokens
            confidence = len(overlap) / len(question_tokens)

        confidence = max(0.0, min(1.0, confidence))
        return Decision(label=option, confidence=confidence)
