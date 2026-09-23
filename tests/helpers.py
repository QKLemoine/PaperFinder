from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from paperfinder.config import Config
from paperfinder.fetch import Paper

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def make_paper(i: int, comment: str | None = "CVPR 2026", journal_ref: str | None = None) -> Paper:
    arxiv_id = f"2603.{i:05d}v1"
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {i}",
        abstract=f"Abstract {i}",
        authors=[f"Author {i}"],
        categories=["cs.CV"],
        published=NOW - timedelta(hours=i),
        abs_url=f"http://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        comment=comment,
        journal_ref=journal_ref,
    )


def arxiv_result(i: int, comment: str | None = "CVPR 2026", journal_ref: str | None = None):
    return SimpleNamespace(
        title=f"Paper {i}",
        summary=f"Abstract {i}",
        authors=[SimpleNamespace(name=f"Author {i}")],
        categories=["cs.CV"],
        published=NOW - timedelta(hours=i),
        entry_id=f"http://arxiv.org/abs/2603.{i:05d}v1",
        pdf_url=None,
        comment=comment,
        journal_ref=journal_ref,
        get_short_id=lambda: f"2603.{i:05d}v1",
    )


class FakeArxivClient:
    """Stands in for arxiv.Client; honors Search.max_results like the real one."""

    results_to_return: list = []
    searches: list = []

    def __init__(self, **kwargs):
        pass

    def results(self, search):
        FakeArxivClient.searches.append(search)
        return iter(FakeArxivClient.results_to_return[: search.max_results])


def make_config(tmp: Path, profile: str = "I study seizure video.", **overrides) -> Config:
    profile_path = tmp / "research_profile.md"
    profile_path.write_text(profile, encoding="utf-8")
    fields = dict(
        categories=["cs.CV"],
        lookback_days=2,
        max_results=400,
        screening_model="screen-model",
        screening_effort="",
        strong_model="strong-model",
        strong_effort="",
        rescore_top=3,
        batch_size=2,
        top_n=2,
        score_threshold=6,
        digest_dir=tmp / "digests",
        write_json_archive=False,
        profile_path=profile_path,
        venue_aliases={"NeurIPS": ["NIPS"]},
    )
    fields.update(overrides)
    return Config(**fields)


class TempDirMixin:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()
