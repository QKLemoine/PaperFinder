import json
import unittest
from types import SimpleNamespace
from unittest import mock

from paperfinder import score
from paperfinder.cache import ScoreCache

from .helpers import FakeAnthropic, TempDirMixin, make_config, make_paper


def run_conference(client, cfg, papers):
    cache = score.open_cache(cfg)
    final = score.score_conference(client, cfg, papers, cache)
    top = [s for s in final if s.score >= cfg.score_threshold][: cfg.top_n]
    return final, score.conference_summaries(client, cfg, top, cache)


class FreeRerunTest(TempDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.papers = [make_paper(i) for i in range(1, 6)]

    def test_unchanged_rerun_makes_zero_model_calls(self):
        cfg = make_config(self.tmp)
        client = FakeAnthropic()
        first = run_conference(client, cfg, self.papers)
        self.assertTrue(client.calls)

        client2 = FakeAnthropic()
        second = run_conference(client2, cfg, self.papers)
        self.assertEqual(client2.calls, [])
        self.assertEqual(
            [(s.paper.arxiv_id, s.score, s.reason) for s in first[0]],
            [(s.paper.arxiv_id, s.score, s.reason) for s in second[0]],
        )
        self.assertEqual(first[1], second[1])

    def test_first_run_shape(self):
        cfg = make_config(self.tmp)
        client = FakeAnthropic()
        final, summaries = run_conference(client, cfg, self.papers)
        # batch_size=2: 5 papers screen in 3 calls; shortlist of 3 rescored in 2 calls.
        self.assertEqual([c[1] for c in client.calls if c[0] == "parse"], ["screen-model"] * 3 + ["strong-model"] * 2)
        self.assertEqual([s.paper.title for s in final], ["Paper 1", "Paper 2", "Paper 3"])
        self.assertEqual([s.score for s in final], [9, 8, 7])
        self.assertEqual(summaries, ["Summary of paper 1.", "Summary of paper 2."])

    def test_rescore_only_shortlisted_papers_missing_a_rescore(self):
        run_conference(FakeAnthropic(), make_config(self.tmp, rescore_top=2), self.papers)

        client = FakeAnthropic()
        run_conference(client, make_config(self.tmp, rescore_top=3), self.papers)
        self.assertEqual(client.calls, [("parse", "strong-model", [3])])

    def test_screening_scores_survive_rescoring(self):
        cfg = make_config(self.tmp)
        run_conference(FakeAnthropic(), cfg, self.papers)
        entry = score.open_cache(cfg).entries["2603.00001v1"]
        self.assertEqual(entry["screening"], {"score": 8, "reason": "screen-model on 1"})
        self.assertEqual(entry["rescored"], {"score": 9, "reason": "strong-model on 1"})

    def test_new_paper_only_screens_the_new_one(self):
        cfg = make_config(self.tmp)
        run_conference(FakeAnthropic(), cfg, self.papers)
        client = FakeAnthropic()
        run_conference(client, cfg, self.papers + [make_paper(9)])
        self.assertEqual(client.calls, [("parse", "screen-model", [9])])

    def test_summary_fallback_is_not_cached(self):
        cfg = make_config(self.tmp)
        client = FakeAnthropic()
        client.messages.create = lambda **_: SimpleNamespace(stop_reason="refusal", content=[])
        _, summaries = run_conference(client, cfg, self.papers)
        self.assertEqual(summaries, ["strong-model on 1", "strong-model on 2"])
        self.assertIsNone(score.open_cache(cfg).get("2603.00001v1", "summary"))


class CacheInvalidationTest(TempDirMixin, unittest.TestCase):
    def warm(self, cfg):
        run_conference(FakeAnthropic(), cfg, [make_paper(1), make_paper(2)])

    def calls_for(self, cfg):
        client = FakeAnthropic()
        run_conference(client, cfg, [make_paper(1), make_paper(2)])
        return client.calls

    def test_profile_change_invalidates_and_revert_restores(self):
        self.warm(make_config(self.tmp, profile="A"))
        self.assertTrue(self.calls_for(make_config(self.tmp, profile="B")))
        self.assertEqual(self.calls_for(make_config(self.tmp, profile="A")), [])

    def test_either_model_change_invalidates(self):
        self.warm(make_config(self.tmp))
        self.assertTrue(self.calls_for(make_config(self.tmp, screening_model="other")))
        self.assertTrue(self.calls_for(make_config(self.tmp, strong_model="other")))

    def test_scoring_version_bump_invalidates(self):
        cfg = make_config(self.tmp)
        self.warm(cfg)
        with mock.patch.object(score, "SCORING_VERSION", score.SCORING_VERSION + 1):
            self.assertTrue(self.calls_for(cfg))


class AtomicWriteTest(TempDirMixin, unittest.TestCase):
    def test_saves_after_each_batch(self):
        cfg = make_config(self.tmp, rescore_top=0)
        client = FakeAnthropic()
        sizes = []
        real_save = ScoreCache.save

        def spy(cache_self):
            real_save(cache_self)
            sizes.append(len(json.loads(cache_self.path.read_text())[cache_self.key]))

        with mock.patch.object(ScoreCache, "save", spy):
            score.score_conference(client, cfg, [make_paper(i) for i in range(1, 6)], score.open_cache(cfg))
        self.assertEqual(sizes, [2, 4, 5])

    def test_failed_write_keeps_previous_file_and_no_temp_files(self):
        path = self.tmp / "cache.json"
        cache = ScoreCache(path, "k")
        cache.put("a", "screening", {"score": 1, "reason": "r"})
        cache.save()
        before = path.read_text()

        cache.put("b", "screening", {"score": 2, "reason": "r"})
        with mock.patch("paperfinder.cache.json.dump", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                cache.save()
        self.assertEqual(path.read_text(), before)
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["cache.json"])


if __name__ == "__main__":
    unittest.main()
