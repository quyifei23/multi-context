"""Counterexamples to timing/completion claims from sequential USERD reads."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_userd_progress import make_plan, validate_samples


class ProgressEvidence(unittest.TestCase):
    def setUp(self):
        labels = (['idle_before'] * 8 + ['before_begin_event', 'after_begin_event', 'before_launch', 'after_launch'] +
                  ['launch_no_api'] * 8 + ['after_end_event', 'event_query', 'event_query', 'after_elapsed_time', 'after_dtoh'] +
                  ['idle_after'] * 8)
        api = {'after_begin_event', 'after_launch', 'after_end_event', 'event_query', 'after_elapsed_time', 'after_dtoh'}
        self.rows = []
        for i, label in enumerate(labels):
            begin = 1000 * (i + 1)
            self.rows.append(dict(kind='userd_sample', round=i, label=label, begin_ns=begin, end_ns=begin + 30,
                api_begin_ns=begin - 100 if label in api else 0, api_end_ns=begin - 10 if label in api else 0,
                api_result=600 if i == 21 else 0,
                channels=[dict(ordinal=0, begin_ns=begin + 1, end_ns=begin + 20,
                               get_first=0, put_first=0, get_second=0, put_second=0)]))

    def test_bounded_success_with_pending_then_complete(self):
        samples, queries = validate_samples(self.rows, 1)
        self.assertEqual(len(queries), 2)
        self.assertEqual(len(samples), 33)

    def test_event_never_ready(self):
        self.rows[22]['api_result'] = 600
        with self.assertRaisesRegex(ValueError, 'completion_not_proven'): validate_samples(self.rows, 1)

    def test_async_launch_error(self):
        self.rows[11]['api_result'] = 700
        with self.assertRaisesRegex(ValueError, 'cuda_api_error'): validate_samples(self.rows, 1)

    def test_api_mislabeled_as_no_api_window(self):
        self.rows[12]['api_begin_ns'] = 1
        with self.assertRaisesRegex(ValueError, 'unexpected_api'): validate_samples(self.rows, 1)

    def test_incomplete_channel_scan(self):
        self.rows[11]['channels'] = []
        with self.assertRaisesRegex(ValueError, 'member'): validate_samples(self.rows, 1)

    def test_missing_read_round(self):
        self.rows[12]['round'] += 1
        with self.assertRaisesRegex(ValueError, 'lost_read_round'): validate_samples(self.rows, 1)

    def test_nonatomic_change_is_retained(self):
        self.rows[11]['channels'][0]['get_second'] = 1
        # A changing pair is valid evidence, never silently filtered as noise.
        samples, _ = validate_samples(self.rows, 1)
        self.assertEqual(samples[11]['channels'][0]['get_first'], 0)
        self.assertEqual(samples[11]['channels'][0]['get_second'], 1)

    def test_field_bracket_outside_round(self):
        self.rows[11]['channels'][0]['end_ns'] += 100
        with self.assertRaisesRegex(ValueError, 'field_read_bracket'): validate_samples(self.rows, 1)

    def test_sample_before_api_return(self):
        self.rows[11]['api_end_ns'] = self.rows[11]['begin_ns'] + 1
        with self.assertRaisesRegex(ValueError, 'api_bracket'): validate_samples(self.rows, 1)

    def test_old_process_plan(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'ready').write_text('123 1500\n')
            with self.assertRaisesRegex(ValueError, 'live_owner_pid_mismatch'): make_plan(p, 456)

    def test_budget_overflow(self):
        with self.assertRaisesRegex(ValueError, 'read_budget'):
            validate_samples([dict(kind='userd_sample')] * 257, 1)


if __name__ == '__main__':
    unittest.main()
