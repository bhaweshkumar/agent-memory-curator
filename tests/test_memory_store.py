import json

import pytest

from memory_store import Memory, MemoryStore


def test_load_valid_file(tmp_path):
    path = tmp_path / "memories.json"
    path.write_text(
        json.dumps(
            [
                {"id": "m1", "text": "hello", "tags": ["a"], "created_at": "2026-01-01T00:00:00Z"},
                {"id": "m2", "text": "world", "tags": [], "created_at": "2026-01-02T00:00:00Z"},
            ]
        )
    )

    store = MemoryStore(path)
    memories = store.load()

    assert memories == [
        Memory(id="m1", text="hello", tags=["a"], created_at="2026-01-01T00:00:00Z"),
        Memory(id="m2", text="world", tags=[], created_at="2026-01-02T00:00:00Z"),
    ]


def test_load_skips_malformed_entries(tmp_path, caplog):
    path = tmp_path / "memories.json"
    path.write_text(
        json.dumps(
            [
                {"id": "m1", "text": "ok entry", "tags": ["a"], "created_at": "2026-01-01T00:00:00Z"},
                {"id": "m2", "text": "missing tags"},
                "not even a dict",
                {"id": "m3", "text": 123, "tags": [], "created_at": "2026-01-01T00:00:00Z"},
            ]
        )
    )

    store = MemoryStore(path)
    with caplog.at_level("WARNING"):
        memories = store.load()

    assert memories == [Memory(id="m1", text="ok entry", tags=["a"], created_at="2026-01-01T00:00:00Z")]
    assert len(caplog.records) == 3


def test_load_missing_file_returns_empty(tmp_path):
    store = MemoryStore(tmp_path / "does_not_exist.json")
    assert store.load() == []


def test_save_round_trip(tmp_path):
    path = tmp_path / "memories.json"
    store = MemoryStore(path)
    memories = [
        Memory(id="m1", text="hello", tags=["a"], created_at="2026-01-01T00:00:00Z"),
    ]

    store.save(memories)

    reloaded = MemoryStore(path).load()
    assert reloaded == memories
