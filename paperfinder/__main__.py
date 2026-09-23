"""Entry point: python -m paperfinder"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import anthropic

from . import config as config_mod
from . import digest, fetch, score

API_ERRORS = (anthropic.AuthenticationError, anthropic.RateLimitError, anthropic.BadRequestError)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _anthropic_client() -> anthropic.Anthropic | None:
    try:
        return anthropic.Anthropic()
    except Exception as exc:
        _log(f"Could not initialize the Anthropic client: {exc}")
        _log(
            "Set ANTHROPIC_API_KEY, or run `ant auth login` to store a profile the SDK "
            "picks up automatically."
        )
        return None


def _report_api_error(exc: Exception) -> int:
    if isinstance(exc, anthropic.AuthenticationError):
        _log("Authentication failed. Check ANTHROPIC_API_KEY, or run `ant auth status`.")
    elif isinstance(exc, anthropic.RateLimitError):
        _log("Rate limited. Retry later, or lower max_results in config.toml.")
    else:
        _log(f"Request rejected: {exc}")
        _log(
            "If this names `effort`, the model does not accept that parameter — "
            "clear screening_effort/strong_effort in config.toml."
        )
    return 1


def _progress(done: int, total: int) -> None:
    _log(f"  scored {done}/{total}")


def _warn_truncated(cfg: config_mod.Config) -> None:
    _log(
        f"\n!!! WARNING: TRUNCATED. More than {cfg.max_results} papers matched; only the "
        f"newest {cfg.max_results} were kept.\n"
        "!!! Rerun with a higher --limit to cover the whole venue.\n"
    )


def _print_paper(p: fetch.Paper, prefix: str) -> None:
    print(f"  {prefix} {p.arxiv_id}  {p.title[:90]}")
    if p.comment:
        print(f"      comment:     {p.comment}")
    if p.journal_ref:
        print(f"      journal-ref: {p.journal_ref}")


def _run_conference(cfg: config_mod.Config, args: argparse.Namespace) -> int:
    names = fetch.venue_names(args.venue, cfg.venue_aliases)
    _log(
        f"Searching arXiv comments and journal refs for {' / '.join(names)} {args.year} "
        f"(up to {cfg.max_results} papers, newest first)..."
    )
    _log(f"Query: {fetch.build_conference_query(names, args.year)}")
    papers, truncated = fetch.fetch_conference(cfg, names, args.year)
    _log(f"Retrieved {len(papers)} papers.")
    if truncated:
        _warn_truncated(cfg)

    filtered = fetch.filter_conference(papers, names, args.year)
    _log(f"Excluded {filtered.excluded} as likely false positives:")
    _log(f"  {filtered.no_proximity:>4}  venue and year not written together")
    for marker, n in filtered.by_marker.items():
        _log(f"  {n:>4}  mentions {marker!r}")

    if args.show_excluded:
        print(f"\nExcluded papers ({filtered.excluded}):")
        for p, reason in filtered.excluded_papers:
            label = reason if reason == fetch.NO_PROXIMITY else f"marker: {reason!r}"
            _print_paper(p, f"[{label}]")
        print()
        sys.stdout.flush()

    _log(f"{len(filtered.kept)} papers left to screen.")

    if args.dry_run:
        for p in filtered.kept:
            _print_paper(p, f"[{p.published.date()}]")
        sys.stdout.flush()
        if truncated:
            _warn_truncated(cfg)
        return 0

    if not filtered.kept:
        return 0

    client = _anthropic_client()
    if client is None:
        return 1

    cache = score.open_cache(cfg)
    cached = sum(cache.get(p.arxiv_id, "screening") is not None for p in filtered.kept)
    _log(
        f"Stage 1: screening {len(filtered.kept)} papers against {cfg.profile_path.name} "
        f"with {cfg.screening_model} ({cached} cached)..."
    )
    try:
        final = score.score_conference(client, cfg, filtered.kept, cache, _progress, _log)
        top = [s for s in final if s.score >= cfg.score_threshold][: cfg.top_n]
        _log(f"{len(top)} paper(s) cleared the bar (score >= {cfg.score_threshold:g}).")
        summaries = score.conference_summaries(client, cfg, top, cache)
    except API_ERRORS as exc:
        return _report_api_error(exc)

    content = digest.render_conference(
        cfg, args.venue, args.year, top, summaries, len(papers), truncated, filtered
    )
    path = digest.write_conference(cfg, content, args.venue, args.year)
    _log(f"\nWrote {path}")
    if truncated:
        _warn_truncated(cfg)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="paperfinder",
        description="Pull a daily batch of recent arXiv papers and rank them against research_profile.md.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and list papers without calling the scoring model (no API cost).",
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Override lookback_days for this run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Override max_results for this run. Useful for a cheap first test.",
    )
    subparsers = parser.add_subparsers(dest="command")
    conference = subparsers.add_parser(
        "conference",
        help="Rank papers accepted to one venue and year instead of recent submissions.",
        description="Find arXiv papers whose comment or journal ref names the venue and "
        "year, and rank them against research_profile.md. Matching is best-effort.",
    )
    conference.add_argument("--venue", required=True, help="e.g. CVPR, NeurIPS")
    conference.add_argument("--year", required=True, type=int, help="e.g. 2026")
    conference.add_argument(
        "--dry-run",
        action="store_true",
        help="Show match and exclusion counts with each paper's comment (no API cost).",
    )
    conference.add_argument("--limit", type=int, help="Override max_results for this run.")
    conference.add_argument(
        "--show-excluded",
        action="store_true",
        help="List each excluded paper with its comment/journal ref and why it was excluded.",
    )
    args = parser.parse_args()

    cfg = config_mod.load()
    if args.days is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "lookback_days": args.days})
    if args.limit is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "max_results": args.limit})

    if args.command == "conference":
        return _run_conference(cfg, args)

    print(
        f"Fetching up to {cfg.max_results} papers from {', '.join(cfg.categories)} "
        f"(last {cfg.lookback_days}d)...",
        file=sys.stderr,
    )
    papers = fetch.fetch_recent(cfg)
    print(f"Retrieved {len(papers)} papers.", file=sys.stderr)

    if not papers:
        print(
            "Nothing in the window. arXiv does not publish on weekends — try --days 3.",
            file=sys.stderr,
        )
        return 0

    if args.dry_run:
        for p in papers:
            print(f"  [{p.published.date()}] {p.arxiv_id}  {p.title[:90]}")
        return 0

    client = _anthropic_client()
    if client is None:
        return 1

    print(
        f"Stage 1: screening {len(papers)} papers against {cfg.profile_path.name} "
        f"with {cfg.screening_model}...",
        file=sys.stderr,
    )
    try:
        scored = score.score_papers(
            client, cfg, papers, cfg.screening_model, cfg.screening_effort, _progress
        )

        # Stage 2 re-ranks only the shortlist. Screening scores and rescored scores
        # come from different models and are not comparable, so once stage 2 runs the
        # final selection is drawn from the rescored set alone.
        if cfg.rescore_top > 0 and scored:
            shortlist = [s.paper for s in scored[: cfg.rescore_top]]
            print(
                f"Stage 2: re-scoring top {len(shortlist)} with {cfg.strong_model}"
                f"{f' (effort={cfg.strong_effort})' if cfg.strong_effort else ''}...",
                file=sys.stderr,
            )
            scored = score.score_papers(
                client, cfg, shortlist, cfg.strong_model, cfg.strong_effort
            )
    except API_ERRORS as exc:
        return _report_api_error(exc)

    top = [s for s in scored if s.score >= cfg.score_threshold][: cfg.top_n]
    print(
        f"{len(top)} paper(s) cleared the bar (score >= {cfg.score_threshold:g}).",
        file=sys.stderr,
    )

    summaries = score.write_summaries(client, cfg, top)
    run_date = date.today()
    content = digest.render(cfg, top, summaries, len(papers), run_date)
    path = digest.write(cfg, content, run_date)
    print(f"\nWrote {path}", file=sys.stderr)

    if cfg.write_json_archive:
        archive = digest.write_archive(cfg, scored, run_date)
        print(f"Wrote {archive}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
