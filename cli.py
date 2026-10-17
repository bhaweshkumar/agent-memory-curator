"""REPL entrypoint: curate memories via Decider pattern, then ask Bedrock."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from decider_client import MockDeciderClient
from memory_curator import curate, selected_memories
from memory_store import MemoryStore

logging.basicConfig(level=logging.WARNING)

DATA_PATH = Path(__file__).parent / "data" / "memories.json"


def _chars_and_tokens(text: str) -> tuple[int, int]:
    chars = len(text)
    # rough token estimate: ~4 chars per token
    tokens = max(1, chars // 4) if chars else 0
    return chars, tokens


def print_relevance_table(scored_memories, elapsed_seconds: float) -> None:
    print(f"\nRelevance scoring ({elapsed_seconds * 1000:.0f} ms for {len(scored_memories)} memories):")
    print(f"{'id':<10} {'confidence':<12} {'selected':<10} text")
    for sm in scored_memories:
        truncated = sm.memory.text[:60] + ("..." if len(sm.memory.text) > 60 else "")
        selected_mark = "y" if sm.selected else "n"
        print(f"{sm.memory.id:<10} {sm.confidence:<12.2f} {selected_mark:<10} {truncated}")


def print_comparison(all_memories, curated_memories) -> None:
    all_text = "\n".join(m.text for m in all_memories)
    curated_text = "\n".join(m.text for m in curated_memories)

    all_chars, all_tokens = _chars_and_tokens(all_text)
    curated_chars, curated_tokens = _chars_and_tokens(curated_text)

    print("\nContext size comparison:")
    print(f"  all memories:     {len(all_memories)} memories, {all_chars} chars, ~{all_tokens} tokens")
    print(f"  curated subset:   {len(curated_memories)} memories, {curated_chars} chars, ~{curated_tokens} tokens")


def run_repl() -> None:
    store = MemoryStore(DATA_PATH)
    memories = store.load()
    decider = MockDeciderClient()

    print(f"Loaded {len(memories)} memories from {DATA_PATH}")
    print("Agent Memory Curator -- type a question, or 'exit' to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        start = time.monotonic()
        scored_memories = curate(question=question, memories=memories, decider=decider)
        elapsed = time.monotonic() - start

        print_relevance_table(scored_memories, elapsed)

        curated = selected_memories(scored_memories)
        if not curated:
            print("\nNo relevant memories found above threshold; answering with no memory context.")

        try:
            from agent import answer_question

            answer = answer_question(question, curated)
            print(f"\nAnswer:\n{answer}\n")
        except Exception as exc:
            print(f"\nBedrock call failed: {exc}\n")

        print_comparison(memories, curated)
        print()


def main() -> None:
    run_repl()


if __name__ == "__main__":
    main()
