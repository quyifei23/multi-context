"""Synthetic counterexamples: address equality alone never authorizes a ring read."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import test_userd_mapping as fixture
from analyze_userd_mapping import resolve_channel
from gpfifo_binding import resolve_ring
from analyze_gpfifo_entry import attribute_launch, stable_frame


class RingBinding(unittest.TestCase):
    def setUp(self):
        f = fixture.MappingEvidence(); f.setUp()
        self.identity, self.member, self.memories, self.vmas = f.identity, f.member, f.memories, f.vmas
        self.identity['gpu_uuid_hex'] = 'a' * 32
        cl, dev, sub, group, mem, channel, setup, mapping = f.events
        mem['seq'] = 7; channel['seq'] = 15; setup['seq'] = 11; mapping['seq'] = 12
        channel.update(context_share=8, vaspace=0, gpfifo_entries=256, userd_offsets=[3072, 0], fd=4,
                       fd_dev=1, fd_ino=22, fd_rdev=3, fd_path='/dev/nvidiactl')
        va = dict(seq=5, kind='alloc', client=1, object=7, parent=2, status=0, flags=0, rc=0, **{'class': 0x90f1})
        cs = dict(seq=6, kind='alloc', client=1, object=8, parent=4, status=0, flags=0, rc=0,
                  ctxshare_vaspace=7, **{'class': 0x9067})
        def uvm(seq, kind, **fields):
            return dict(seq=seq, kind=kind, fd=20, fd_dev=1, fd_ino=55, fd_rdev=5, fd_path='/dev/nvidia-uvm',
                        client=1, gpu_uuid_hex='a' * 32, rm_fd=4, rm_fd_dev=1, rm_fd_ino=22,
                        rm_fd_rdev=3, rm_fd_path='/dev/nvidiactl', status=0, rc=0, **fields)
        init = uvm(8, 'uvm_initialize', flags=0)
        mm = uvm(9, 'uvm_mm_initialize', uvm_fd=20); mm['fd'] = 21
        reg = uvm(10, 'uvm_register_vaspace', vaspace=7)
        area = uvm(13, 'uvm_create_range', base=0x800000, length=4096)
        gpu_map = uvm(14, 'uvm_map_external', base=0x800000, length=4096, offset=0, memory=5,
                      gpu_attributes=[dict(gpu_uuid_hex='a' * 32, mapping_type=1)])
        cr = uvm(16, 'uvm_register_channel', channel=6)
        self.events = [cl, dev, sub, group, va, cs, mem, init, mm, reg, setup, mapping, area, gpu_map, channel, cr]

    def bind(self):
        u = resolve_channel(self.events, self.identity, self.member, self.memories, self.vmas)
        return resolve_ring(self.events, self.identity, self.member, u, self.vmas)

    def test_complete_same_object_mapping(self):
        self.assertEqual(self.bind()['ring_cpu_address'], 0x100000)

    def test_equal_cpu_gpu_address_without_gpu_mapping(self):
        self.events = [e for e in self.events if e['kind'] != 'uvm_map_external']
        self.events[-2]['gpfifo_gpu_va'] = 0x100000
        with self.assertRaisesRegex(ValueError, 'gpu_va_mapping_missing'): self.bind()

    def test_wrong_gpu_uuid(self):
        self.events[13]['gpu_attributes'][0]['gpu_uuid_hex'] = 'b' * 32
        with self.assertRaisesRegex(ValueError, 'gpu_va_mapping_missing'): self.bind()

    def test_wrong_memory_object(self):
        self.events[13]['memory'] = 999
        with self.assertRaisesRegex(ValueError, 'validated_userd_memory'): self.bind()

    def test_wrong_rm_fd(self):
        self.events[13]['rm_fd'] = 999
        with self.assertRaisesRegex(ValueError, 'fd_mismatch'): self.bind()

    def test_gpu_unmap(self):
        self.events.append(dict(kind='uvm_unmap_external', seq=17, fd=20, base=0x800010, length=32, status=0))
        with self.assertRaisesRegex(ValueError, 'range_changed'): self.bind()

    def test_range_freed(self):
        self.events.append(dict(kind='uvm_free', seq=17, fd=20, base=0x800000, status=0))
        with self.assertRaisesRegex(ValueError, 'range_freed'): self.bind()

    def test_vaspace_freed(self):
        self.events.append(dict(kind='free', seq=17, client=1, object=7, status=0))
        with self.assertRaisesRegex(ValueError, 'object_freed'): self.bind()

    def test_mm_fd_closed(self):
        self.events.append(dict(kind='close', seq=17, fd=21, rc=0))
        with self.assertRaisesRegex(ValueError, 'uvm_fd_closed'): self.bind()

    def test_explicit_mm_not_required_status(self):
        self.events[8]['status'] = 0x10006
        self.events.append(dict(kind='close', seq=17, fd=21, rc=0))
        self.assertEqual(self.bind()['uvm_mm_initialize_status'], 0x10006)

    def test_mm_error_is_not_treated_as_optional(self):
        self.events[8]['status'] = 0x56
        with self.assertRaisesRegex(ValueError, 'uvm_mm_lifetime'): self.bind()

    def test_ring_outside_cpu_mapping(self):
        self.events[13]['offset'] = 4096
        with self.assertRaisesRegex(ValueError, 'outside_existing_cpu_mapping'): self.bind()

    def test_unknown_uvm_mutation(self):
        self.events.append(dict(kind='uvm_ioctl', seq=17, fd=20, uvm_request=999, rc=0))
        with self.assertRaisesRegex(ValueError, 'unreviewed_uvm_operation'): self.bind()


class EntryAttribution(unittest.TestCase):
    def pair(self, slot=2):
        values = ['0' * 16] * 8
        before = dict(userd_before=[dict(put=slot)], ring_first=values)
        changed = values.copy(); changed[slot] = '0' * 15 + '1'
        after = dict(userd_after=[dict(put=(slot + 1) % 8)], ring_first=changed)
        return before, after

    def test_single_changed_entry(self):
        self.assertEqual(attribute_launch(*self.pair(), 0, 8), 2)

    def test_modulo_wrap(self):
        self.assertEqual(attribute_launch(*self.pair(7), 0, 8), 7)
        self.assertEqual(attribute_launch(*self.pair(0), 0, 8), 0)

    def test_multiple_candidates(self):
        a, b = self.pair(); b['ring_first'][3] = '1' * 16
        with self.assertRaisesRegex(ValueError, 'uniquely_match'): attribute_launch(a, b, 0, 8)

    def test_wrong_slot(self):
        a, b = self.pair(); b['ring_first'][2], b['ring_first'][3] = b['ring_first'][3], b['ring_first'][2]
        with self.assertRaisesRegex(ValueError, 'uniquely_match'): attribute_launch(a, b, 0, 8)

    def test_same_entry_bytes_despite_put_delta(self):
        a, b = self.pair(); b['ring_first'] = a['ring_first'].copy()
        with self.assertRaisesRegex(ValueError, 'uniquely_match'): attribute_launch(a, b, 0, 8)

    def test_put_delta_two(self):
        a, b = self.pair(); b['userd_after'][0]['put'] = 4
        with self.assertRaisesRegex(ValueError, 'delta_not_one'): attribute_launch(a, b, 0, 8)

    def test_torn_ring_read(self):
        f = dict(ring_first=['0' * 16], ring_second=['1' * 16])
        with self.assertRaisesRegex(ValueError, 'double_read_unstable'): stable_frame(f, [8])


if __name__ == '__main__': unittest.main()
