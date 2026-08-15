"""Entry point: python -m paperfinder"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import anthropic

from . import config as config_mod
from . import digest, fetch, score


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
    args = parser.parse_args()

    cfg = config_mod.load()
    if args.days is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "lookback_days": args.days})
    if args.limit is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "max_results": args.limit})

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

    try:
        client = anthropic.Anthropic()
    except Exception as exc:
        print(f"Could not initialize the Anthropic client: {exc}", file=sys.stderr)
        print(
            "Set ANTHROPIC_API_KEY, or run `ant auth login` to store a profile the SDK "
            "picks up automatically.",
            file=sys.stderr,
        )
        return 1

    def progress(done: int, total: int) -> None:
        print(f"  scored {done}/{total}", file=sys.stderr)

    print(
        f"Stage 1: screening {len(papers)} papers against {cfg.profile_path.name} "
        f"with {cfg.screening_model}...",
        file=sys.stderr,
    )
    try:
        scored = score.score_papers(
            client, cfg, papers, cfg.screening_model, cfg.screening_effort, progress
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
    except anthropic.AuthenticationError:
        print(
            "Authentication failed. Check ANTHROPIC_API_KEY, or run `ant auth status`.",
            file=sys.stderr,
        )
        return 1
    except anthropic.RateLimitError:
        print("Rate limited. Retry later, or lower max_results in config.toml.", file=sys.stderr)
        return 1
    except anthropic.BadRequestError as exc:
        print(f"Request rejected: {exc}", file=sys.stderr)
        print(
            "If this names `effort`, the model does not accept that parameter — "
            "clear screening_effort/strong_effort in config.toml.",
            file=sys.stderr,
        )
        return 1

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
