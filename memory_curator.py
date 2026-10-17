"""The intervention wiring: rank+filter stored memories via a DeciderClient.

This is where the reusable `decider_client.py` pattern gets placed in the
agent's path -- the "Context and memory" decision pattern from the hackathon
brief (score which memory/context chunks are relevant before injecting them).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from decider_client import DeciderClient
from memory_store import Memory

logger = logging.getLogger(__name__)

THRESHOLD = 0.6
MAX_MEMORIES = 5


@dataclass(frozen=True)
class ScoredMemory:
    """A memory paired with its relevance confidence and selection outcome."""

    memory: Memory
    confidence: float
    selected: bool
    failed: bool = False


def curate(
    question: str,
    memories: list[Memory],
    decider: DeciderClient,
    threshold: float = THRESHOLD,
    max_memories: int = MAX_MEMORIES,
) -> list[ScoredMemory]:
    """Score every memory for relevance to `question`, rank, and filter.

    Returns ScoredMemory entries for ALL input memories (so callers can show
    a full relevance table), ordered by confidence descending. `selected`
    marks the ones that pass `threshold` and fit within `max_memories`.
    """
    scored: list[ScoredMemory] = []
    for memory in memories:
        failed = False
        try:
            decision = decider.score(question=question, context=memory.text, option="relevant")
            confidence = decision.confidence
            if (
                decision.label != "relevant"
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
            ):
                raise ValueError("Invalid relevance decision")
        except Exception:
            logger.warning("Decider scoring failed for memory %s; excluding entry", memory.id)
            confidence = 0.0
            failed = True
        scored.append(ScoredMemory(memory=memory, confidence=confidence, selected=False, failed=failed))

    scored.sort(key=lambda sm: sm.confidence, reverse=True)

    selected_count = 0
    ranked: list[ScoredMemory] = []
    for sm in scored:
        selected = not sm.failed and sm.confidence >= threshold and selected_count < max_memories
        if selected:
            selected_count += 1
        ranked.append(ScoredMemory(memory=sm.memory, confidence=sm.confidence, selected=selected, failed=sm.failed))

    return ranked


def selected_memories(scored_memories: list[ScoredMemory]) -> list[Memory]:
    """Convenience accessor: just the memories that passed curation."""
    return [sm.memory for sm in scored_memories if sm.selected]
