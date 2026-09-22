"""Coverage counterexamples for queue-entry sets, independent of node count."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_graph_entry import compare_frames, ring_interval, same_publication, validate_frame


def frame(puts, counts):
    channels = [dict(ordinal=i, begin_ns=i + 1, end_ns=i + 1, get=p, put=p) for i, p in enumerate(puts)]
    after = [dict(c, begin_ns=i + len(puts) + 2, end_ns=i + len(puts) + 2) for i, c in enumerate(channels)]
    return dict(begin_ns=0, end_ns=100, userd_before=channels, userd_after=after,
                rings=[dict(ordinal=i, ring_first=['0' * 16] * n, ring_second=['0' * 16] * n) for i, n in enumerate(counts)])


class EntrySets(unittest.TestCase):
    def pair(self, old=(2, 0), new=(5, 0), counts=(8, 8)):
        a, b = frame(old, counts), frame(new, counts)
        for i, n in enumerate(counts):
            for slot in ring_interval(old[i], new[i], n):
                b['rings'][i]['ring_first'][slot] = b['rings'][i]['ring_second'][slot] = '1' * 16
        return a, b, counts

    def test_three_entry_interval(self):
        r = compare_frames(*self.pair())
        self.assertEqual(r[0]['interval_slots'], [2, 3, 4])
        self.assertTrue(all(c['exact_interval_coverage'] for c in r))

    def test_one_entry_does_not_require_three_nodes(self):
        self.assertEqual(compare_frames(*self.pair(new=(3, 0)))[0]['put_delta_mod'], 1)

    def test_more_entries_than_nodes(self):
        self.assertEqual(compare_frames(*self.pair(new=(7, 0)))[0]['put_delta_mod'], 5)

    def test_interval_crosses_wrap(self):
        r = compare_frames(*self.pair(old=(7, 0), new=(2, 0)))[0]
        self.assertEqual(r['interval_slots'], [7, 0, 1])
        self.assertTrue(r['exact_interval_coverage'] and r['wrap_observed'])

    def test_multiple_compute_channels_are_data(self):
        r = compare_frames(*self.pair(new=(5, 2)))
        self.assertEqual([c['put_delta_mod'] for c in r], [3, 2])
        self.assertTrue(all(c['exact_interval_coverage'] for c in r))

    def test_unchanged_slot_inside_interval_is_unresolved(self):
        a, b, counts = self.pair()
        b['rings'][0]['ring_first'][3] = b['rings'][0]['ring_second'][3] = '0' * 16
        r = compare_frames(a, b, counts)[0]
        self.assertEqual(r['unchanged_slots_inside'], [3])
        self.assertFalse(r['exact_interval_coverage'])

    def test_changed_slot_outside_interval_is_unresolved(self):
        a, b, counts = self.pair()
        b['rings'][0]['ring_first'][7] = b['rings'][0]['ring_second'][7] = '1' * 16
        r = compare_frames(a, b, counts)[0]
        self.assertEqual(r['changed_slots_outside'], [7])
        self.assertFalse(r['exact_interval_coverage'])

    def test_inactive_channel_ring_mutation_is_not_ignored(self):
        a, b, counts = self.pair()
        b['rings'][1]['ring_first'][1] = b['rings'][1]['ring_second'][1] = '1' * 16
        self.assertFalse(compare_frames(a, b, counts)[1]['exact_interval_coverage'])

    def test_zero_delta_is_not_a_full_ring_inference(self):
        r = compare_frames(*self.pair(new=(2, 0)))[0]
        self.assertEqual(r['interval_slots'], [])

    def test_torn_double_read_rejected(self):
        a, b, counts = self.pair(); b['rings'][0]['ring_second'][2] = '2' * 16
        with self.assertRaisesRegex(ValueError, 'double_read_unstable'): compare_frames(a, b, counts)

    def test_pointer_changes_during_snapshot_rejected(self):
        a, _, counts = self.pair(); a['userd_after'][0]['put'] = 3
        with self.assertRaisesRegex(ValueError, 'publication_changed'): validate_frame(a, counts)

    def test_ring_member_missing_rejected(self):
        a, _, counts = self.pair(); a['rings'].pop()
        with self.assertRaisesRegex(ValueError, 'members_missing'): validate_frame(a, counts)

    def test_deferred_publication_is_visible(self):
        a, b, counts = self.pair()
        self.assertFalse(same_publication(a, b, counts))
        self.assertTrue(same_publication(b, copy.deepcopy(b), counts))

    def test_out_of_range_pointer_rejected(self):
        with self.assertRaisesRegex(ValueError, 'invalid_ring_index'): ring_interval(8, 1, 8)


if __name__ == '__main__': unittest.main()
