"""Configuration loading from config.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    categories: list[str]
    lookback_days: int
    max_results: int
    screening_model: str
    screening_effort: str
    strong_model: str
    strong_effort: str
    rescore_top: int
    batch_size: int
    top_n: int
    score_threshold: float
    digest_dir: Path
    write_json_archive: bool
    profile_path: Path

    @property
    def profile(self) -> str:
        text = self.profile_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(
                f"{self.profile_path.name} is empty. The scoring model has nothing to "
                "score against — describe your research ideas there first."
            )
        return text


def load(path: Path | None = None) -> Config:
    path = path or ROOT / "config.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    retrieval = raw.get("retrieval", {})
    scoring = raw.get("scoring", {})
    output = raw.get("output", {})

    categories = retrieval.get("categories", [])
    if not categories:
        raise ValueError("config.toml: [retrieval] categories must list at least one arXiv category.")

    digest_dir = Path(output.get("digest_dir", "digests"))
    if not digest_dir.is_absolute():
        digest_dir = ROOT / digest_dir

    return Config(
        categories=list(categories),
        lookback_days=int(retrieval.get("lookback_days", 2)),
        max_results=int(retrieval.get("max_results", 200)),
        screening_model=scoring.get("screening_model", "claude-haiku-4-5"),
        screening_effort=str(scoring.get("screening_effort", "")).strip(),
        strong_model=scoring.get("strong_model", "claude-sonnet-5"),
        strong_effort=str(scoring.get("strong_effort", "medium")).strip(),
        rescore_top=int(scoring.get("rescore_top", 25)),
        batch_size=int(scoring.get("batch_size", 50)),
        top_n=int(scoring.get("top_n", 3)),
        score_threshold=float(scoring.get("score_threshold", 6)),
        digest_dir=digest_dir,
        write_json_archive=bool(output.get("write_json_archive", False)),
        profile_path=ROOT / "research_profile.md",
    )
