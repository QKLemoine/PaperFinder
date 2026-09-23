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


FALSE_POSITIVE_MARKERS = ["workshop", "submitted to", "under review", "rejected"]


def build_category_query(categories: list[str]) -> str:
    return " OR ".join(f"cat:{c}" for c in categories)


def venue_names(venue: str, aliases: dict[str, list[str]]) -> list[str]:
    """`venue` plus every alias grouped with it, matched case-insensitively either way."""
    names = [venue]
    for canonical, group in aliases.items():
        if venue.lower() in {n.lower() for n in [canonical, *group]}:
            names += [canonical, *group]
    return list({n.lower(): n for n in names}.values())


def build_conference_query(names: list[str], year: int) -> str:
    yyyy, yy = str(year), str(year)[-2:]
    phrases = []
    for name in names:
        name = name.replace('"', "")
        phrases += [f"{name} {yyyy}", f"{name}{yyyy}", f"{name}'{yy}", f"{name}{yy}"]
    return " OR ".join(f'{field}:"{p}"' for p in phrases for field in ("co", "jr"))


def venue_pattern(names: list[str], year: int) -> re.Pattern:
    """Venue and year within a few non-alphanumeric characters, in either order.

    Letter boundaries keep "CVPRW"/"ICCVW" (workshops) from matching; the two-digit year
    is only accepted glued on or apostrophed ("CVPR26", "CVPR'26"), since a bare "CVPR, 26"
    is as likely a page count.
    """
    yyyy, yy = str(year), str(year)[-2:]
    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    venue = rf"(?<![A-Za-z])(?:{alt})(?![A-Za-z])"
    gap = r"[^A-Za-z0-9\n]{0,4}"
    return re.compile(
        rf"{venue}(?:{gap}{yyyy}|\s?['’]{yy}|{yy})(?!\d)|(?<!\d){yyyy}{gap}{venue}",
        re.IGNORECASE,
    )


@dataclass
class ConferenceFilterResult:
    kept: list[Paper]
    no_proximity: int
    by_marker: dict[str, int]

    @property
    def excluded(self) -> int:
        return self.no_proximity + sum(self.by_marker.values())


def filter_conference(papers: list[Paper], names: list[str], year: int) -> ConferenceFilterResult:
    """Drop papers whose venue/year hit isn't a real acceptance.

    Each excluded paper is counted once: under proximity if venue and year never appear
    together, otherwise under the first false-positive marker it contains.
    """
    pattern = venue_pattern(names, year)
    result = ConferenceFilterResult(kept=[], no_proximity=0, by_marker={m: 0 for m in FALSE_POSITIVE_MARKERS})
    for paper in papers:
        text = "\n".join(t for t in (paper.comment, paper.journal_ref) if t)
        if not pattern.search(text):
            result.no_proximity += 1
            continue
        lowered = text.lower()
        marker = next((m for m in FALSE_POSITIVE_MARKERS if m in lowered), None)
        if marker:
            result.by_marker[marker] += 1
            continue
        result.kept.append(paper)
    return result


def fetch_conference(config: Config, names: list[str], year: int) -> tuple[list[Paper], bool]:
    """Every paper whose comment or journal-ref names the venue and year, newest first.

    No lookback window. Returns (papers, truncated) — truncated means max_results cut
    off real matches.
    """
    return _retrieve(
        build_conference_query(names, year), config.max_results, detect_truncation=True
    )


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
