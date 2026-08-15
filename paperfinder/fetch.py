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


def fetch_recent(config: Config) -> list[Paper]:
    """Return papers submitted within the lookback window, newest first."""
    query = " OR ".join(f"cat:{c}" for c in config.categories)
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.lookback_days)

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        # Results stream newest-first, so we can stop as soon as one predates the
        # cutoff rather than paging through the whole category history.
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
        max_results=config.max_results,
    )

    papers: list[Paper] = []
    seen: set[str] = set()

    for result in client.results(search):
        if result.published < cutoff:
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
            )
        )

    return papers
