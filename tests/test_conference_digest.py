import unittest

from paperfinder import digest
from paperfinder.fetch import ConferenceFilterResult
from paperfinder.score import ScoredPaper

from .helpers import TempDirMixin, make_config, make_paper


class ConferenceDigestTest(TempDirMixin, unittest.TestCase):
    def render(self, truncated):
        paper = make_paper(1, "Accepted to CVPR 2026", journal_ref="Proc. CVPR 2026")
        filtered = ConferenceFilterResult(
            kept=[paper], no_proximity=2, by_marker={"workshop": 3, "rejected": 0}
        )
        return digest.render_conference(
            make_config(self.tmp), "CVPR", 2026, [ScoredPaper(paper, 9, "r")], ["Summary."],
            6, truncated, filtered,
        )

    def test_includes_matched_text_and_counts(self):
        text = self.render(truncated=False)
        self.assertIn("> Comment: Accepted to CVPR 2026", text)
        self.assertIn("> Journal ref: Proc. CVPR 2026", text)
        self.assertIn("excluded 5 (2 without venue and year together; workshop: 3)", text)
        self.assertIn("best-effort", text)
        self.assertNotIn("Warning", text)

    def test_truncation_warning(self):
        self.assertIn("**Warning:**", self.render(truncated=True))

    def test_filename_slug(self):
        cfg = make_config(self.tmp)
        path = digest.write_conference(cfg, "x", "../ICCV W/s", 2026)
        self.assertEqual(path, cfg.digest_dir / "conference-ICCV-W-s-2026.md")


if __name__ == "__main__":
    unittest.main()
