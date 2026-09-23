# PaperFinder

A daily arXiv digest that ranks new papers by whether their *substance* matches your
research ideas, not whether their titles share your keywords.

Three stages:

1. **The net** — pull every paper submitted to your chosen arXiv categories in the
   lookback window. No keyword filtering here; narrowing at this stage is exactly the
   title-matching failure the tool exists to avoid.
2. **The filter** — send each abstract to Claude alongside `research_profile.md` and get
   back a 0-10 relevance score with a one-sentence rationale, via structured outputs.
3. **The deliverable** — write the top papers to `digests/YYYY-MM-DD.md` with a short
   write-up and links.

## Setup

Dependencies are already installed in `.venv` (`arxiv`, `anthropic`). To recreate:

```sh
python3 -m venv .venv
.venv/bin/pip install arxiv anthropic
```

Then set credentials — the SDK finds either automatically:

```sh
export ANTHROPIC_API_KEY="sk-ant-..."
# or, for a stored profile instead of a static key:
ant auth login
```

## Before the first real run

**Create `research_profile.md` from the template**, then edit it:

```sh
cp research_profile.example.md research_profile.md
```

`research_profile.md` is gitignored — your actual research ideas stay local and are
never committed. It is also the single highest-leverage input in this pipeline: the
filter can only be as discriminating as the profile is specific. Vague profiles produce a
digest that reads plausible and ranks noise. Say what you're working on, list the
specific ideas you're tracking, and be explicit about what *doesn't* count; the
exclusions do as much work as the inclusions.

## Usage

```sh
# See what the net catches, with no model calls and no cost
.venv/bin/python -m paperfinder --dry-run

# Full run: fetch, score, write digests/YYYY-MM-DD.md
.venv/bin/python -m paperfinder

# Cheap first test on a small slice
.venv/bin/python -m paperfinder --limit 20

# Widen the window after a weekend or a gap
.venv/bin/python -m paperfinder --days 4
```

`--limit` and `--days` override config for one run only.

## Conference mode

Instead of the last few days of submissions, rank everything accepted to one venue:

```sh
# Match counts, exclusions, and each paper's matched comment — no model calls
.venv/bin/python -m paperfinder conference --venue CVPR --year 2026 --dry-run

# Full run: writes digests/conference-CVPR-2026.md
.venv/bin/python -m paperfinder conference --venue CVPR --year 2026 --limit 2500
```

**Venue matching is best-effort.** arXiv has no acceptance metadata, so the net searches
the free-text comment (`co:`) and journal-ref (`jr:`) fields for phrases like
`CVPR 2026`, `CVPR2026`, `CVPR'26`, and `CVPR26`. It then keeps only papers where the
venue and year are written close together, and drops any whose comment or journal ref
mentions `workshop`, `submitted to`, `under review`, or `rejected`. Expect it to miss
accepted papers whose authors never updated their arXiv comment, and to let through the
occasional oddly worded false positive. `--dry-run` shows the exclusion counts per
reason and the matched text for every paper kept, so you can check before paying.

Details:

- **Aliases.** Add venues known by more than one name under `[conference.aliases]` in
  `config.toml` (NeurIPS/NIPS is there already); any name in a group searches all of them.
- **Truncation.** Results are newest-first and capped at `max_results`. Big venues have
  thousands of matches, so if the cap cuts any off, the run warns loudly — including in
  `--dry-run` and in the digest — and you should rerun with a higher `--limit`. Every
  screened paper costs a scoring call, so dry-run first to see the size.
- **Free re-runs.** Screening scores, rescores, and digest paragraphs are cached per
  paper in `digests/.conference_cache.json`. The cache is keyed on the research profile,
  both model names, and `SCORING_VERSION` in `score.py` (bump it when you change the
  prompts), so an edit to any of them rescores from scratch. Re-running unchanged makes
  no model calls; adding papers or raising `rescore_top` only pays for what's new.
- Uses the same two-stage scoring and `[scoring]` settings as daily mode; `lookback_days`
  and `--days` don't apply.

## Configuration

`config.toml` holds everything tunable; each field is commented there. The ones you'll
actually touch:

| Field | What it does |
|---|---|
| `categories` | arXiv categories to sweep. Broad is fine — the filter does the narrowing. |
| `lookback_days` | arXiv doesn't publish weekends; `1` will return nothing on a Sunday. |
| `max_results` | Cost ceiling. Every retrieved abstract goes to the model. |
| `top_n` | How many papers reach the digest. |
| `score_threshold` | Floor for inclusion, so a thin day yields a short digest rather than three weak papers. |
| `write_json_archive` | Also dump *all* scored papers to JSON for re-ranking or analysis. |

## Cost and the two-stage cascade

Scoring runs in two stages so the expensive model never reads the whole feed:

- **Stage 1** screens all ~200 papers with `claude-haiku-4-5`. Its only job is recall —
  deciding which 25 are plausibly worth a closer look. That is a far easier call than
  final ranking, which is what makes a cheap model appropriate here.
- **Stage 2** re-scores that shortlist with `claude-sonnet-5` and writes the digest.
  Better judgment on the boundary cases, applied to 25 papers instead of 200.

Because the two models' scores aren't on a comparable scale, the final digest is drawn
from the rescored set alone. Set `rescore_top = 0` to skip stage 2 and rank on screening
scores only — cheapest, and noticeably blunter.

Rough per-run cost on a 200-paper sweep (character-based estimates, not measured):

| Setup | Per run | ~Monthly (weekdays) |
|---|---|---|
| Opus 5 for everything | $1.00–1.50 | $22–33 |
| Sonnet 5 for everything | $0.30–0.50 | $7–11 |
| **Haiku screen + Sonnet rescore (default)** | **~$0.10** | **~$2** |
| Haiku only (`rescore_top = 0`) | ~$0.05 | ~$1 |

Narrowing `categories` scales all of these roughly linearly — `cs.AI` alone is 147 of the
203 papers in a typical 2-day window.

There is no free Anthropic tier. A local model via Ollama is free in dollars, but this
task is precisely where small models fail: they fall back to keyword matching, which is
the thing this tool exists to avoid. If you want to go that route, use it as a stage-1
screener only and keep a hosted model for stage 2 — the cascade is already shaped for it.

**A note on `effort`:** `claude-haiku-4-5` rejects the parameter, which is why
`screening_effort` defaults to empty. If you switch `screening_model` to something that
accepts it, you can set it then.

## Running it daily

Once you've confirmed a manual run looks right, add a cron entry (use absolute paths, and
note that cron gets a minimal environment, so set the key explicitly):

```sh
crontab -e
# 8am on weekdays
0 8 * * 1-5 cd "/Users/quintenlemoine/Desktop/Personal Projects/PaperSort/PaperFinder" && ANTHROPIC_API_KEY="sk-ant-..." .venv/bin/python -m paperfinder >> digests/run.log 2>&1
```

On macOS, cron needs Full Disk Access to write under `~/Desktop`; a `launchd` agent is the
sturdier option if that gets in the way.

## Tuning the filter

The scores are only as good as the profile, and the fastest way to calibrate is to read
the rationales rather than the ranks. Set `write_json_archive = true`, run for a few days,
and look at the papers scoring 5-7 — the boundary cases are where a profile's vagueness
shows up. If a paper you'd have wanted scores low, the profile is usually missing the idea
rather than the model missing the paper.

If everything scores 6+, the bar has drifted and the digest stops being a filter; tighten
the exclusions in the profile before raising `score_threshold`.

## Layout

```
config.toml            # all tunable settings
research_profile.md    # your ideas — the scoring target
paperfinder/
  config.py            # config loading
  fetch.py             # the net: arXiv retrieval, conference query + filters
  score.py             # the filter: scoring + digest prose
  cache.py             # per-paper score cache for conference mode
  digest.py            # the deliverable: Markdown rendering
  __main__.py          # CLI
tests/                 # unittest, no network or API calls
digests/               # output (gitignored)
```

Run the tests with `.venv/bin/python -m unittest discover -s tests -t .`

## Possible extensions

- **Semantic Scholar recommendations** as a second net, seeded from papers you've already
  saved — better recall for work outside your usual categories, at the cost of a second
  API and a seed list to maintain.
- **Cross-run dedupe**, so a paper that appeared in an earlier digest doesn't resurface
  when it's cross-listed or revised.
