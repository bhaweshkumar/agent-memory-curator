"""Flat-JSON memory storage: load, validate, and persist a list of memories."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("id", "text", "tags", "created_at")


@dataclass(frozen=True)
class Memory:
    """A single stored memory entry."""

    id: str
    text: str
    tags: list[str]
    created_at: str


def _is_valid_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    if not all(field in entry for field in REQUIRED_FIELDS):
        return False
    if not isinstance(entry["id"], str) or not isinstance(entry["text"], str):
        return False
    if not isinstance(entry["tags"], list):
        return False
    if not isinstance(entry["created_at"], str):
        return False
    return True


class MemoryStore:
    """Loads/saves a flat JSON file of memory entries."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._memories: list[Memory] = []

    def load(self) -> list[Memory]:
        """Load memories from disk, skipping and logging malformed entries."""
        if not self.path.exists():
            self._memories = []
            return self._memories

        raw_text = self.path.read_text(encoding="utf-8")
        raw_entries = json.loads(raw_text) if raw_text.strip() else []

        memories: list[Memory] = []
        for index, entry in enumerate(raw_entries):
            if not _is_valid_entry(entry):
                logger.warning("Skipping malformed memory entry at index %d: %r", index, entry)
                continue
            memories.append(
                Memory(
                    id=entry["id"],
                    text=entry["text"],
                    tags=list(entry["tags"]),
                    created_at=entry["created_at"],
                )
            )

        self._memories = memories
        return self._memories

    def save(self, memories: list[Memory]) -> None:
        """Persist memories to disk as a flat JSON array."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(memory) for memory in memories]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._memories = list(memories)

    @property
    def memories(self) -> list[Memory]:
        return self._memories
