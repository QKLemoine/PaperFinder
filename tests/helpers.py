from __future__ import annotations

import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from paperfinder import score
from paperfinder.config import Config
from paperfinder.fetch import Paper

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


class FakeAnthropic:
    """Scores paper i as 9 - i on the screening model and 10 - i on the strong model.

    `omit` leaves those paper numbers out of every response; `fail_parse_calls` makes
    those (0-based) parse calls come back unparseable.
    """

    def __init__(self, omit: set[int] = frozenset(), fail_parse_calls: set[int] = frozenset()):
        self.calls = []
        self.omit = omit
        self.fail_parse_calls = fail_parse_calls
        self.messages = SimpleNamespace(parse=self._parse, create=self._create)

    def _parse(self, model, messages, output_format, **_):
        content = messages[0]["content"]
        found = re.findall(r'<paper index="(\d+)">\n<title>Paper (\d+)</title>', content)
        call_no = sum(c[0] == "parse" for c in self.calls)
        self.calls.append(("parse", model, [int(n) for _, n in found]))
        if call_no in self.fail_parse_calls:
            return SimpleNamespace(parsed_output=None, stop_reason="max_tokens")
        base = 10 if model == "strong-model" else 9
        assessments = [
            score.Assessment(index=int(idx), score=max(0, base - int(n)), reason=f"{model} on {n}")
            for idx, n in found
            if int(n) not in self.omit
        ]
        return SimpleNamespace(parsed_output=score.Screening(assessments=assessments), stop_reason="end_turn")

    def _create(self, model, messages, **_):
        titles = re.findall(r"<title>Paper (\d+)</title>", messages[0]["content"])
        self.calls.append(("create", model, [int(n) for n in titles]))
        text = "\n\n".join(f"Summary of paper {n}." for n in titles)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def make_paper(
    i: int,
    comment: str | None = "CVPR 2026",
    journal_ref: str | None = None,
    version: int = 1,
) -> Paper:
    arxiv_id = f"2603.{i:05d}v{version}"
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
        stats_path=tmp / "stats.json",
    )
    fields.update(overrides)
    return Config(**fields)


class TempDirMixin:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()
