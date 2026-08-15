"""The Deliverable: render the top papers to a dated Markdown digest."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .config import Config
from .score import ScoredPaper


def _authors(names: list[str], limit: int = 4) -> str:
    if not names:
        return "Unknown"
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f", +{len(names) - limit} more"


def render(
    config: Config,
    top: list[ScoredPaper],
    summaries: list[str],
    total_retrieved: int,
    run_date: date,
) -> str:
    lines = [
        f"# arXiv digest — {run_date.isoformat()}",
        "",
        f"Screened **{total_retrieved}** papers from "
        f"{', '.join(config.categories)} over the last "
        f"{config.lookback_days} day{'s' if config.lookback_days != 1 else ''}.",
        "",
    ]

    if not top:
        lines += [
            f"Nothing cleared the relevance bar (score ≥ {config.score_threshold:g}) today.",
            "",
            "A run of empty days usually means the sweep is looking in the wrong place, not "
            "that the field went quiet — widen `categories` or revisit `research_profile.md`.",
            "",
        ]
        return "\n".join(lines)

    for i, (item, summary) in enumerate(zip(top, summaries), 1):
        paper = item.paper
        lines += [
            f"## {i}. {paper.title}",
            "",
            f"**Relevance {item.score}/10** · {_authors(paper.authors)} · "
            f"{paper.published.date().isoformat()} · {', '.join(paper.categories[:3])}",
            "",
            summary,
            "",
            f"[abstract]({paper.abs_url}) · [pdf]({paper.pdf_url}) · `{paper.arxiv_id}`",
            "",
        ]

    return "\n".join(lines)


def write(config: Config, content: str, run_date: date) -> Path:
    config.digest_dir.mkdir(parents=True, exist_ok=True)
    path = config.digest_dir / f"{run_date.isoformat()}.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_archive(config: Config, scored: list[ScoredPaper], run_date: date) -> Path:
    config.digest_dir.mkdir(parents=True, exist_ok=True)
    path = config.digest_dir / f"{run_date.isoformat()}.json"
    path.write_text(
        json.dumps([s.to_dict() for s in scored], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
