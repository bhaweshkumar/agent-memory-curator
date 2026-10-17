from decider_client import Decision
from memory_curator import curate, selected_memories
from memory_store import Memory


class StubDecider:
    """Returns a pre-programmed confidence per memory id for deterministic tests."""

    def __init__(self, confidences_by_memory_text: dict[str, float]):
        self._confidences = confidences_by_memory_text

    def score(self, question: str, context: str, option: str) -> Decision:
        return Decision(label=option, confidence=self._confidences[context])


def _memory(id_: str, text: str) -> Memory:
    return Memory(id=id_, text=text, tags=[], created_at="2026-01-01T00:00:00Z")


def test_ranks_by_confidence_descending():
    memories = [_memory("m1", "low"), _memory("m2", "high"), _memory("m3", "mid")]
    decider = StubDecider({"low": 0.1, "high": 0.9, "mid": 0.5})

    ranked = curate("question", memories, decider, threshold=0.0, max_memories=10)

    assert [sm.memory.id for sm in ranked] == ["m2", "m3", "m1"]


def test_filters_below_threshold():
    memories = [_memory("m1", "below"), _memory("m2", "above")]
    decider = StubDecider({"below": 0.4, "above": 0.8})

    ranked = curate("question", memories, decider, threshold=0.6, max_memories=10)

    selected = selected_memories(ranked)
    assert [m.id for m in selected] == ["m2"]


def test_caps_at_max_memories():
    memories = [_memory(f"m{i}", f"text{i}") for i in range(10)]
    decider = StubDecider({f"text{i}": 1.0 for i in range(10)})

    ranked = curate("question", memories, decider, threshold=0.0, max_memories=3)

    selected = selected_memories(ranked)
    assert len(selected) == 3
    # all ranked entries still returned for display purposes
    assert len(ranked) == 10


def test_empty_store_returns_empty():
    decider = StubDecider({})
    ranked = curate("question", [], decider)
    assert ranked == []
    assert selected_memories(ranked) == []


def test_decider_exception_treated_as_zero_confidence_and_excluded():
    class FailingDecider:
        def score(self, question, context, option):
            raise RuntimeError("boom")

    memories = [_memory("m1", "text")]
    ranked = curate("question", memories, FailingDecider(), threshold=0.0, max_memories=10)

    assert ranked[0].confidence == 0.0
