"""Running set of every paper a scoring model has actually scored, for the scanned count.

IDs are stored without their version suffix, so a revised paper counts once. Only the
aggregate count leaves this module: stats.json carries no IDs, titles, or profile text.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from .cache import write_json_atomic
from .config import Config

if TYPE_CHECKING:
    from .score import ScoredPaper


def base_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def _seed_from_cache(cache_path: Path) -> set[str]:
    """Every paper the conference cache holds a screening score for, in any namespace."""
    if not cache_path.exists():
        return set()
    namespaces = json.loads(cache_path.read_text(encoding="utf-8"))
    return {
        base_id(arxiv_id)
        for entries in namespaces.values()
        for arxiv_id, entry in entries.items()
        if "screening" in entry
    }


class Ledger:
    def __init__(self, path: Path, ids: set[str], dirty: bool = False):
        self.path = path
        self.ids = ids
        self.dirty = dirty

    @classmethod
    def open(cls, config: Config) -> Ledger:
        """Load the ledger, or seed it from the conference cache on first use.

        Opening never writes; a seeded ledger is saved on the first record or stats write.
        """
        if config.ledger_path.exists():
            ids = json.loads(config.ledger_path.read_text(encoding="utf-8"))
            return cls(config.ledger_path, set(ids))
        return cls(config.ledger_path, _seed_from_cache(config.conference_cache_path), dirty=True)

    @property
    def count(self) -> int:
        return len(self.ids)

    def record(self, scored: list[ScoredPaper]) -> None:
        new = {base_id(s.paper.arxiv_id) for s in scored} - self.ids
        if new or self.dirty:
            self.ids |= new
            self.save()

    def save(self) -> None:
        write_json_atomic(self.path, sorted(self.ids))
        self.dirty = False


def write_stats(config: Config, ledger: Ledger, today: date | None = None) -> None:
    if ledger.dirty:
        ledger.save()
    stats = {"papers_scanned": ledger.count, "updated": (today or date.today()).isoformat()}
    write_json_atomic(config.stats_path, stats, indent=2)


def read_stats(config: Config) -> dict | None:
    if not config.stats_path.exists():
        return None
    return json.loads(config.stats_path.read_text(encoding="utf-8"))
