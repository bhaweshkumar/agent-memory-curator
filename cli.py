"""REPL entrypoint: curate memories via Decider pattern, then ask Bedrock."""

from __future__ import annotations

import logging
import argparse
import os
import time
from pathlib import Path

from decider_client import DEFAULT_DECIDER_URL, DeciderClient, MockDeciderClient, StrandsDeciderClient
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
    print(f"{'id':<10} {'score':<12} {'selected':<10} {'failed':<8} text")
    for sm in scored_memories:
        truncated = sm.memory.text[:60] + ("..." if len(sm.memory.text) > 60 else "")
        selected_mark = "y" if sm.selected else "n"
        print(f"{sm.memory.id:<10} {sm.confidence:<12.2f} {selected_mark:<10} {str(sm.failed):<8} {truncated}")


def print_comparison(all_memories, curated_memories) -> None:
    all_text = "\n".join(m.text for m in all_memories)
    curated_text = "\n".join(m.text for m in curated_memories)

    all_chars, all_tokens = _chars_and_tokens(all_text)
    curated_chars, curated_tokens = _chars_and_tokens(curated_text)

    print("\nContext size comparison:")
    print(f"  all memories:     {len(all_memories)} memories, {all_chars} chars, ~{all_tokens} tokens")
    print(f"  curated subset:   {len(curated_memories)} memories, {curated_chars} chars, ~{curated_tokens} tokens")


def run_repl(decider: DeciderClient | None = None, questions: list[str] | None = None) -> bool:
    store = MemoryStore(DATA_PATH)
    memories = store.load()
    if decider is None:
        decider = StrandsDeciderClient()
    scripted_questions = iter(questions) if questions is not None else None
    successful = True

    print(f"Loaded {len(memories)} memories from {DATA_PATH}")
    if isinstance(decider, MockDeciderClient):
        print("Backend: MOCK keyword overlap with simulated delay; not a qualifying Decider run.")
    else:
        print("Backend: Decider 2B; score is P(relevant), not choice-confidence.")
    print("Agent Memory Curator -- type a question, or 'exit' to quit.\n")

    while True:
        try:
            question = (next(scripted_questions) if scripted_questions is not None else input("> ")).strip()
            if scripted_questions is not None:
                print(f"> {question}")
        except StopIteration:
            break
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
        failures = sum(entry.failed for entry in scored_memories)
        if failures:
            print(f"\nWarning: {failures} memory decisions failed and were excluded.")
        if scored_memories and failures == len(scored_memories):
            print("Skipping Bedrock: every Decider request failed. Check the server, model, and URL.")
            print_comparison(memories, curated)
            successful = False
            continue
        if not curated:
            print("\nNo relevant memories found above threshold; answering with no memory context.")

        try:
            from agent import answer_question

            generation_start = time.monotonic()
            answer = answer_question(question, curated)
            print(f"\nBedrock answer latency: {(time.monotonic() - generation_start) * 1000:.0f} ms")
            print(f"\nAnswer:\n{answer}\n")
        except Exception as exc:
            print(f"\nBedrock call failed: {exc}\n")
            successful = False

        print_comparison(memories, curated)
        print(f"Decision-loop latency: {elapsed * 1000:.0f} ms")
        print()
    return successful


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Curate coding memories with Decider 2B before asking Bedrock.")
    parser.add_argument("--backend", choices=("decider", "mock"), default="decider")
    parser.add_argument("--decider-url", default=os.environ.get("DECIDER_URL", DEFAULT_DECIDER_URL))
    parser.add_argument("--question", action="append", help="Run a question without the REPL; repeat for a smoke run.")
    args = parser.parse_args(argv)
    decider = MockDeciderClient() if args.backend == "mock" else StrandsDeciderClient(base_url=args.decider_url)
    if run_repl(decider=decider, questions=args.question) is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
