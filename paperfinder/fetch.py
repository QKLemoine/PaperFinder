"""The Net: broad retrieval of recent arXiv papers.

Pulls everything submitted in the lookback window across the configured categories.
No filtering happens here beyond the date window — narrowing is the scoring model's
job, and keyword pre-filtering at this stage is exactly the title-matching failure
mode this pipeline exists to avoid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import arxiv

from .config import Config


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: datetime
    abs_url: str
    pdf_url: str
    comment: str | None = None
    journal_ref: str | None = None

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "published": self.published.isoformat(),
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
        }


def _clean(text: str) -> str:
    """arXiv abstracts arrive with hard-wrapped newlines; collapse to flowing text."""
    return re.sub(r"\s+", " ", text).strip()


def build_category_query(categories: list[str]) -> str:
    return " OR ".join(f"cat:{c}" for c in categories)


def fetch_recent(config: Config) -> list[Paper]:
    """Return papers submitted within the lookback window, newest first."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.lookback_days)
    papers, _ = _retrieve(build_category_query(config.categories), config.max_results, cutoff)
    return papers


def _retrieve(
    query: str,
    max_results: int,
    cutoff: datetime | None = None,
    detect_truncation: bool = False,
) -> tuple[list[Paper], bool]:
    """Page through `query` newest-first. Returns (papers, truncated).

    With `detect_truncation`, one extra result is requested so a full page can be told
    apart from a cap that actually cut results off.
    """
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        # Results stream newest-first, so we can stop as soon as one predates the
        # cutoff rather than paging through the whole category history.
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
        max_results=max_results + 1 if detect_truncation else max_results,
    )

    papers: list[Paper] = []
    seen: set[str] = set()
    truncated = False

    for n, result in enumerate(client.results(search)):
        if n == max_results:
            truncated = True
            break
        if cutoff is not None and result.published < cutoff:
            break

        arxiv_id = result.get_short_id()
        # Strip the version suffix so v1 and v2 of the same paper dedupe together.
        base_id = arxiv_id.split("v")[0]
        if base_id in seen:
            continue
        seen.add(base_id)

        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=_clean(result.title),
                abstract=_clean(result.summary),
                authors=[a.name for a in result.authors],
                categories=list(result.categories),
                published=result.published,
                abs_url=result.entry_id,
                pdf_url=result.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                comment=result.comment,
                journal_ref=result.journal_ref,
            )
        )

    return papers, truncated
