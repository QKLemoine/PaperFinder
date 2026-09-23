"""Per-paper score cache for conference mode.

Entries live under a namespace key derived from everything that can change a score, so
editing the profile, switching a model, or bumping SCORING_VERSION starts a fresh
namespace instead of serving stale scores. Old namespaces are kept, so reverting a
change makes its scores free again.

Layout: {namespace: {arxiv_id: {"screening": {...}, "rescored": {...}, "summary": str}}}
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def namespace_key(profile: str, screening_model: str, strong_model: str, version: int) -> str:
    payload = json.dumps([profile, screening_model, strong_model, version])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ScoreCache:
    def __init__(self, path: Path, key: str):
        self.path = path
        self.key = key
        self._all: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.entries: dict[str, dict] = self._all.setdefault(key, {})

    def get(self, arxiv_id: str, slot: str):
        return self.entries.get(arxiv_id, {}).get(slot)

    def put(self, arxiv_id: str, slot: str, value) -> None:
        self.entries.setdefault(arxiv_id, {})[slot] = value

    def save(self) -> None:
        """Write atomically: a crash mid-write leaves the previous file intact."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".cache-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._all, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            os.unlink(tmp)
            raise
