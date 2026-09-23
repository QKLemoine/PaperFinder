import unittest
from unittest import mock

import arxiv

from paperfinder import fetch

from .helpers import FakeArxivClient, TempDirMixin, arxiv_result, make_config, make_paper


class QueryBuilderTest(unittest.TestCase):
    def test_category_query_unchanged(self):
        self.assertEqual(
            fetch.build_category_query(["cs.CV", "eess.IV"]), "cat:cs.CV OR cat:eess.IV"
        )

    def test_conference_query_phrase_variants_in_both_fields(self):
        q = fetch.build_conference_query(["CVPR"], 2026)
        for phrase in ["CVPR 2026", "CVPR2026", "CVPR'26", "CVPR26"]:
            self.assertIn(f'co:"{phrase}"', q)
            self.assertIn(f'jr:"{phrase}"', q)
        self.assertEqual(q.count(" OR "), 7)

    def test_conference_query_strips_quotes(self):
        self.assertNotIn('""', fetch.build_conference_query(['CV"PR'], 2026))

    def test_venue_names_resolve_aliases_both_directions(self):
        aliases = {"NeurIPS": ["NIPS"]}
        self.assertEqual(fetch.venue_names("neurips", aliases), ["NeurIPS", "NIPS"])
        self.assertEqual(fetch.venue_names("NIPS", aliases), ["NIPS", "NeurIPS"])
        self.assertEqual(fetch.venue_names("CVPR", aliases), ["CVPR"])


class VenuePatternTest(unittest.TestCase):
    def setUp(self):
        self.pattern = fetch.venue_pattern(["CVPR"], 2026)

    def test_matches(self):
        for text in [
            "Accepted to CVPR 2026",
            "cvpr2026 camera ready",
            "Accepted by CVPR'26",
            "Accepted by CVPR’26",
            "CVPR26",
            "IEEE/CVF (CVPR), 2026",
            "2026 CVPR",
        ]:
            self.assertTrue(self.pattern.search(text), text)

    def test_rejects(self):
        for text in [
            "CVPR 2025",
            "CVPR 2025, 26 pages",
            "CVPR, 26 pages",
            "CVPRW 2026",
            "CVPR 20261",
            "Accepted to CVPR. Code released 2026",
            "",
        ]:
            self.assertFalse(self.pattern.search(text), text)


class FilterTest(unittest.TestCase):
    def test_counts_each_exclusion_once(self):
        papers = [
            make_paper(1, "Accepted to CVPR 2026"),
            make_paper(2, "Submitted to CVPR 2026"),
            make_paper(3, "CVPR 2026 Workshop, under review elsewhere"),
            make_paper(4, None, journal_ref="CVPR 2026 WORKSHOP on Gait"),
            make_paper(5, "Mentions CVPR and later 2026"),
            make_paper(6, "10 pages", journal_ref="Proc. CVPR 2026"),
            make_paper(7, "CVPR 2026 (rejected from ICCV)"),
        ]
        result = fetch.filter_conference(papers, ["CVPR"], 2026)
        self.assertEqual([p.arxiv_id for p in result.kept], ["2603.00001v1", "2603.00006v1"])
        self.assertEqual(result.no_proximity, 1)
        self.assertEqual(
            result.by_marker,
            {"workshop": 2, "submitted to": 1, "under review": 0, "rejected": 1},
        )
        self.assertEqual(result.excluded, 5)


class FetchConferenceTest(TempDirMixin, unittest.TestCase):
    def fetch(self, n_available, max_results):
        FakeArxivClient.results_to_return = [arxiv_result(i) for i in range(n_available)]
        FakeArxivClient.searches = []
        cfg = make_config(self.tmp, max_results=max_results)
        with mock.patch.object(arxiv, "Client", FakeArxivClient):
            return fetch.fetch_conference(cfg, ["CVPR"], 2026)

    def test_truncation_detected_when_more_results_exist(self):
        papers, truncated = self.fetch(n_available=5, max_results=3)
        self.assertTrue(truncated)
        self.assertEqual(len(papers), 3)

    def test_no_truncation_when_results_fit_exactly(self):
        papers, truncated = self.fetch(n_available=3, max_results=3)
        self.assertFalse(truncated)
        self.assertEqual(len(papers), 3)

    def test_sorted_newest_first_without_lookback(self):
        papers, _ = self.fetch(n_available=3, max_results=10)
        search = FakeArxivClient.searches[0]
        self.assertEqual(search.sort_by, arxiv.SortCriterion.SubmittedDate)
        self.assertEqual(search.sort_order, arxiv.SortOrder.Descending)
        self.assertEqual(papers[0].comment, "CVPR 2026")


if __name__ == "__main__":
    unittest.main()
