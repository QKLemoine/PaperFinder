"""The Filter: semantic evaluation of each paper against the research profile.

Two passes. The first scores every retrieved paper on its abstract, in batches, using
structured outputs so the scores come back parseable rather than as prose we have to
regex. The second takes only the survivors and writes the digest.

Scores are absolute (each paper judged against the profile, not against the other
papers in its batch), so results from separate batches are directly comparable and can
be merged and re-sorted afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic
from pydantic import BaseModel, Field

from .cache import ScoreCache, namespace_key
from .config import Config
from .fetch import Paper

# Bump when the prompts or scoring semantics change, so cached scores are not reused.
SCORING_VERSION = 1

SCORING_SYSTEM = """\
You are screening a day's worth of new arXiv papers for one researcher, against the \
research profile below.

Judge each paper on what its abstract says it actually does — the mechanism it proposes, \
the phenomenon it measures, the study it runs. A paper whose title and framing come from a \
different subfield can still be highly relevant if its substance bears on the profile, and a \
paper that shares the profile's vocabulary can be irrelevant if the overlap is only in \
terminology. Surface-level keyword overlap with the profile is not evidence of relevance; \
treat it as neutral and read for substance.

Score each paper 0-10 on relevance to the profile:

- 9-10: Directly advances one of the profile's stated ideas. The researcher would want to \
read this today.
- 7-8: Clearly bears on the profile — a useful method, result, or framing — but is not \
squarely on one of the stated ideas.
- 4-6: Adjacent. Shares a subfield or a technique, but the connection needs an argument.
- 1-3: Same broad area, no real bearing on the profile.
- 0: Unrelated, or explicitly ruled out by the profile's exclusions.

Be willing to use the full range, including the low end. On a typical day most papers in a \
broad category sweep score below 4, and a run where everything lands at 6+ means the bar \
has drifted rather than that the day was unusually good.

For `reason`, write one sentence naming the specific link to the profile — the idea it \
touches and how — or, for a low score, the specific reason it misses. "Relevant to the \
profile" and "not relevant" carry no information; name the mechanism.

Score every paper you are given, and use each paper's `index` exactly as provided.

<research_profile>
{profile}
</research_profile>"""

DIGEST_SYSTEM = """\
You are writing a short daily reading digest for one researcher, from the papers that \
scored highest against the research profile below.

For each paper, write a tight paragraph covering what it does, and what specifically \
connects it to the profile — the idea it touches and why it earns the researcher's time \
today. Ground every claim in the abstract you were given; if the abstract does not say \
whether something was evaluated with users, do not assert that it was. Where a paper is a \
partial fit, say so plainly rather than overselling it — a digest the researcher learns to \
trust is worth more than one that flatters every entry.

Write flowing prose, not bullet fragments. No preamble, no closing summary, and do not \
restate the scores or repeat the titles as headers — the surrounding document already \
carries those. Return only the paragraphs, one per paper, separated by a blank line, in \
the order the papers were given.

<research_profile>
{profile}
</research_profile>"""


class Assessment(BaseModel):
    index: int = Field(description="The paper's index, exactly as given in the input.")
    score: int = Field(description="Relevance to the research profile, 0-10.")
    reason: str = Field(description="One sentence naming the specific link to the profile.")


class Screening(BaseModel):
    assessments: list[Assessment]


@dataclass
class ScoredPaper:
    paper: Paper
    score: int
    reason: str

    def to_dict(self) -> dict:
        return {**self.paper.to_dict(), "score": self.score, "reason": self.reason}


def _render_batch(papers: list[Paper], offset: int) -> str:
    blocks = []
    for i, paper in enumerate(papers):
        blocks.append(
            f"<paper index=\"{offset + i}\">\n"
            f"<title>{paper.title}</title>\n"
            f"<categories>{', '.join(paper.categories)}</categories>\n"
            f"<abstract>{paper.abstract}</abstract>\n"
            f"</paper>"
        )
    return "\n\n".join(blocks)


def score_papers(
    client: anthropic.Anthropic,
    config: Config,
    papers: list[Paper],
    model: str,
    effort: str = "",
    on_progress=None,
) -> list[ScoredPaper]:
    """Score papers against the profile with `model`. Returns results sorted best-first.

    `effort` is omitted when empty. Some models (claude-haiku-4-5) reject the
    parameter outright, so it is opt-in rather than defaulted.
    """
    system = [
        {
            "type": "text",
            "text": SCORING_SYSTEM.format(profile=config.profile),
            # The profile is stable across batches and across runs, so it is worth a
            # cache breakpoint. Below the model's minimum cacheable prefix this is a
            # silent no-op, which is the right failure mode for a short profile.
            "cache_control": {"type": "ephemeral"},
        }
    ]

    scored: list[ScoredPaper] = []

    # The SDK merges output_format into output_config, so an effort-only
    # output_config here keeps the structured-output schema intact.
    extra = {"output_config": {"effort": effort}} if effort else {}

    for start in range(0, len(papers), config.batch_size):
        batch = papers[start : start + config.batch_size]
        response = client.messages.parse(
            model=model,
            max_tokens=16000,
            **extra,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Score these {len(batch)} papers against the profile.\n\n"
                        f"{_render_batch(batch, start)}"
                    ),
                }
            ],
            output_format=Screening,
        )

        result = response.parsed_output
        if result is None:
            raise RuntimeError(
                f"Scoring returned no parseable output for papers {start}-{start + len(batch) - 1} "
                f"(stop_reason={response.stop_reason}). Lower batch_size in config.toml and retry."
            )

        for assessment in result.assessments:
            local = assessment.index - start
            if not 0 <= local < len(batch):
                continue  # Model returned an index outside this batch; drop it.
            scored.append(
                ScoredPaper(
                    paper=batch[local],
                    score=max(0, min(10, assessment.score)),
                    reason=assessment.reason.strip(),
                )
            )

        if on_progress:
            on_progress(min(start + config.batch_size, len(papers)), len(papers))

    scored.sort(key=lambda s: (-s.score, s.paper.title))
    return scored


def write_summaries(
    client: anthropic.Anthropic,
    config: Config,
    top: list[ScoredPaper],
) -> list[str]:
    """Write one digest paragraph per top paper. Falls back to the scoring reason."""
    return _generate_summaries(client, config, top) or [item.reason for item in top]


def _generate_summaries(
    client: anthropic.Anthropic,
    config: Config,
    top: list[ScoredPaper],
) -> list[str] | None:
    """One paragraph per paper, or None when the output can't be trusted."""
    if not top:
        return []

    blocks = []
    for i, item in enumerate(top, 1):
        blocks.append(
            f"<paper n=\"{i}\">\n"
            f"<title>{item.paper.title}</title>\n"
            f"<abstract>{item.paper.abstract}</abstract>\n"
            f"<screening_note>{item.reason}</screening_note>\n"
            f"</paper>"
        )

    response = client.messages.create(
        model=config.strong_model,
        max_tokens=4000,
        system=[{"type": "text", "text": DIGEST_SYSTEM.format(profile=config.profile)}],
        messages=[{"role": "user", "content": "\n\n".join(blocks)}],
    )

    if response.stop_reason == "refusal":
        return None

    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # If the model returned a different number of paragraphs than papers, we can no
    # longer match them up safely — fall back rather than misattribute a summary.
    if len(paragraphs) != len(top):
        return None
    return paragraphs


def open_cache(config: Config) -> ScoreCache:
    key = namespace_key(
        config.profile, config.screening_model, config.strong_model, SCORING_VERSION
    )
    return ScoreCache(config.digest_dir / ".conference_cache.json", key)


def _score_cached(
    client: anthropic.Anthropic,
    config: Config,
    cache: ScoreCache,
    papers: list[Paper],
    slot: str,
    model: str,
    effort: str,
    on_progress=None,
) -> list[ScoredPaper]:
    """Score only papers missing `slot`, saving after every batch. Returns all of them."""
    missing = [p for p in papers if cache.get(p.arxiv_id, slot) is None]
    for start in range(0, len(missing), config.batch_size):
        batch = missing[start : start + config.batch_size]
        for s in score_papers(client, config, batch, model, effort):
            cache.put(s.paper.arxiv_id, slot, {"score": s.score, "reason": s.reason})
        cache.save()
        if on_progress:
            on_progress(min(start + config.batch_size, len(missing)), len(missing))

    scored = [
        ScoredPaper(paper=p, **cache.get(p.arxiv_id, slot))
        for p in papers
        if cache.get(p.arxiv_id, slot) is not None
    ]
    scored.sort(key=lambda s: (-s.score, s.paper.title))
    return scored


def score_conference(
    client: anthropic.Anthropic,
    config: Config,
    papers: list[Paper],
    cache: ScoreCache,
    on_progress=None,
    log=None,
) -> list[ScoredPaper]:
    """Two-stage scoring where every model result is cached per paper.

    The shortlist is always the top `rescore_top` of the full pool by screening score, so
    it is the same set a cold run would pick; only its members lacking a cached rescore
    go to the strong model. An unchanged re-run therefore makes no model calls.
    """
    log = log or (lambda _msg: None)
    screened = _score_cached(
        client, config, cache, papers, "screening",
        config.screening_model, config.screening_effort, on_progress,
    )
    if config.rescore_top <= 0:
        return screened

    shortlist = [s.paper for s in screened[: config.rescore_top]]
    uncached = sum(cache.get(p.arxiv_id, "rescored") is None for p in shortlist)
    log(f"Stage 2: shortlist of {len(shortlist)}, {uncached} not yet rescored.")
    return _score_cached(
        client, config, cache, shortlist, "rescored",
        config.strong_model, config.strong_effort,
    )


def conference_summaries(
    client: anthropic.Anthropic,
    config: Config,
    top: list[ScoredPaper],
    cache: ScoreCache,
) -> list[str]:
    """Cached per paper; falls back to the scoring reason without caching the fallback."""
    missing = [s for s in top if cache.get(s.paper.arxiv_id, "summary") is None]
    if missing:
        paragraphs = _generate_summaries(client, config, missing)
        if paragraphs is not None:
            for s, text in zip(missing, paragraphs):
                cache.put(s.paper.arxiv_id, "summary", text)
            cache.save()
    return [cache.get(s.paper.arxiv_id, "summary") or s.reason for s in top]
