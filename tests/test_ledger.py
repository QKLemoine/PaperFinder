import contextlib
import io
import json
import unittest
from datetime import date
from unittest import mock

import anthropic

from paperfinder import __main__ as cli
from paperfinder import config as config_mod
from paperfinder import fetch, ledger, score

from .helpers import FakeAnthropic, TempDirMixin, make_config, make_paper


class LedgerRecordingTest(TempDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.cfg = make_config(self.tmp)

    def score(self, client, papers):
        led = ledger.Ledger.open(self.cfg)
        score.score_papers(client, self.cfg, papers, "screen-model", on_batch=led.record)
        return led

    def test_records_only_parsed_and_kept_papers(self):
        led = self.score(FakeAnthropic(omit={2}), [make_paper(i) for i in range(1, 5)])
        self.assertEqual(led.ids, {"2603.00001", "2603.00003", "2603.00004"})

    def test_batch_that_returns_but_fails_to_parse_records_nothing(self):
        led = ledger.Ledger.open(self.cfg)
        with self.assertRaises(RuntimeError):
            score.score_papers(
                FakeAnthropic(fail_parse_calls={0}), self.cfg, [make_paper(1), make_paper(2)],
                "screen-model", on_batch=led.record,
            )
        self.assertEqual(led.count, 0)
        self.assertFalse(self.cfg.ledger_path.exists())

    def test_only_batches_before_a_parse_failure_are_recorded(self):
        led = ledger.Ledger.open(self.cfg)
        with self.assertRaises(RuntimeError):
            score.score_papers(
                FakeAnthropic(fail_parse_calls={1}), self.cfg, [make_paper(i) for i in range(1, 5)],
                "screen-model", on_batch=led.record,
            )
        self.assertEqual(json.loads(self.cfg.ledger_path.read_text()), ["2603.00001", "2603.00002"])

    def test_rescoring_the_same_papers_does_not_double_count(self):
        papers = [make_paper(i) for i in range(1, 5)]
        self.score(FakeAnthropic(), papers)
        led = self.score(FakeAnthropic(), papers)
        self.assertEqual(led.count, 4)

    def test_versions_of_one_paper_count_once(self):
        led = self.score(FakeAnthropic(), [make_paper(1, version=1), make_paper(1, version=2)])
        self.assertEqual(led.ids, {"2603.00001"})

    def test_conference_rerun_from_cache_records_nothing(self):
        papers = [make_paper(i) for i in range(1, 6)]
        first = ledger.Ledger.open(self.cfg)
        score.score_conference(FakeAnthropic(), self.cfg, papers, score.open_cache(self.cfg), on_batch=first.record)
        self.assertEqual(first.count, 5)

        second = ledger.Ledger.open(self.cfg)
        client = FakeAnthropic()
        with mock.patch.object(second, "record", wraps=second.record) as record:
            score.score_conference(client, self.cfg, papers, score.open_cache(self.cfg), on_batch=record)
        self.assertEqual(client.calls, [])
        record.assert_not_called()
        self.assertEqual(second.count, 5)


class SeedingTest(TempDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.cfg = make_config(self.tmp)
        self.cfg.digest_dir.mkdir()
        self.cfg.conference_cache_path.write_text(json.dumps({
            "ns1": {
                "2603.00001v1": {"screening": {"score": 1, "reason": "r"}},
                "2603.00002v2": {"screening": {"score": 2, "reason": "r"}, "rescored": {"score": 3, "reason": "r"}},
            },
            "ns2": {
                "2603.00001v2": {"screening": {"score": 1, "reason": "r"}},
                "2603.00009v1": {"summary": "never screened"},
            },
        }))

    def test_first_open_seeds_from_every_cache_namespace_without_writing(self):
        led = ledger.Ledger.open(self.cfg)
        self.assertEqual(led.ids, {"2603.00001", "2603.00002"})
        self.assertFalse(self.cfg.ledger_path.exists())

    def test_seed_is_persisted_by_the_first_stats_write(self):
        ledger.write_stats(self.cfg, ledger.Ledger.open(self.cfg))
        self.assertEqual(json.loads(self.cfg.ledger_path.read_text()), ["2603.00001", "2603.00002"])

    def test_existing_ledger_is_not_reseeded(self):
        self.cfg.ledger_path.write_text(json.dumps(["2603.00005"]))
        self.assertEqual(ledger.Ledger.open(self.cfg).ids, {"2603.00005"})


class CliTest(TempDirMixin, unittest.TestCase):
    """Drives main() end to end with the arXiv fetch and Anthropic client mocked."""

    def setUp(self):
        super().setUp()
        self.cfg = make_config(self.tmp, rescore_top=0)
        self.papers = [make_paper(i) for i in range(1, 5)]

    def main(self, argv, client=None, **patches):
        def no_client():
            raise AssertionError("no model client should be created")

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch("sys.argv", ["paperfinder", *argv]))
            stack.enter_context(mock.patch.object(config_mod, "load", return_value=self.cfg))
            stack.enter_context(mock.patch.object(fetch, "fetch_recent", return_value=self.papers))
            stack.enter_context(mock.patch.object(fetch, "fetch_conference", return_value=(self.papers, False)))
            stack.enter_context(
                mock.patch.object(anthropic, "Anthropic", (lambda: client) if client else no_client)
            )
            for target, value in patches.items():
                stack.enter_context(mock.patch.object(ledger, target, value))
            out = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            code = cli.main()
        return code, out.getvalue()

    def assert_nothing_counted(self):
        self.assertFalse(self.cfg.ledger_path.exists())
        self.assertFalse(self.cfg.stats_path.exists())

    def test_daily_dry_run_counts_nothing(self):
        self.assertEqual(self.main(["--dry-run"])[0], 0)
        self.assert_nothing_counted()

    def test_conference_dry_run_counts_nothing(self):
        self.assertEqual(self.main(["conference", "--venue", "CVPR", "--year", "2026", "--dry-run"])[0], 0)
        self.assert_nothing_counted()

    def test_stats_json_is_aggregate_only_and_reruns_do_not_double_count(self):
        for _ in range(2):
            self.assertEqual(self.main([], client=FakeAnthropic())[0], 0)
            text = self.cfg.stats_path.read_text()
            self.assertEqual(
                json.loads(text), {"papers_scanned": 4, "updated": date.today().isoformat()}
            )
            self.assertNotIn("2603", text)
            self.assertNotIn("seizure", text)

    def test_conference_run_counts_and_cached_rerun_does_not(self):
        argv = ["conference", "--venue", "CVPR", "--year", "2026"]
        self.main(argv, client=FakeAnthropic())
        second = FakeAnthropic()
        self.main(argv, client=second)
        self.assertEqual(second.calls, [])
        self.assertEqual(json.loads(self.cfg.stats_path.read_text())["papers_scanned"], 4)

    def test_mid_run_failure_still_writes_stats(self):
        with self.assertRaises(RuntimeError):
            self.main([], client=FakeAnthropic(fail_parse_calls={1}))
        self.assertEqual(json.loads(self.cfg.stats_path.read_text())["papers_scanned"], 2)

    def test_stats_write_failure_cannot_hide_the_original_exception(self):
        broken = mock.Mock(side_effect=OSError("disk full"))
        with self.assertRaises(RuntimeError):
            self.main([], client=FakeAnthropic(fail_parse_calls={0}), write_stats=broken)
        broken.assert_called_once()

    def test_stats_write_failure_after_a_successful_run_is_raised(self):
        with self.assertRaises(OSError):
            self.main([], client=FakeAnthropic(), write_stats=mock.Mock(side_effect=OSError("disk full")))

    def test_stats_command_reports_totals_without_writing(self):
        self.main([], client=FakeAnthropic())
        before = self.cfg.stats_path.read_text()
        code, out = self.main(["stats"])
        self.assertEqual(code, 0)
        self.assertIn("Papers scanned: 4 unique", out)
        self.assertIn(f"4 as of {date.today().isoformat()}", out)
        self.assertEqual(self.cfg.stats_path.read_text(), before)


if __name__ == "__main__":
    unittest.main()
