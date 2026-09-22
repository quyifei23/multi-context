"""Counterexamples for observation-to-mapping claims; no CUDA/GPU required."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_userd_mapping import resolve_channel


class MappingEvidence(unittest.TestCase):
    def setUp(self):
        def alloc(seq, obj, parent, cls):
            return dict(seq=seq, kind='alloc', client=1, object=obj, parent=parent,
                        status=0, flags=0, rc=0, **{'class': cls})
        self.events = [alloc(1, 1, 0, 0x41), alloc(2, 2, 1, 0x80), alloc(3, 3, 2, 0x2080),
                       alloc(4, 4, 2, 0xa06c), alloc(5, 5, 2, 0x40), alloc(6, 6, 4, 0xc56f)]
        self.events[-1].update(userd_handles=[5, 0], userd_offsets=[512, 0], gpfifo_entries=1024, gpfifo_gpu_va=0x800000)
        self.events.append(dict(kind='map_memory', seq=7, client=1, memory=5, device=2, rc=0,
                                status=0, flags=0, offset=0, length=4096, map_fd=9,
                                map_fd_dev=1, map_fd_ino=11, map_fd_rdev=2, map_fd_path='/dev/nvidia0', tid=10))
        self.events.append(dict(kind='mmap', seq=8, fd=9, rc=0, address=0x100000, length=4096,
                                offset=0, prot=3, flags=1, fd_dev=1, fd_ino=11, fd_rdev=2,
                                fd_path='/dev/nvidia0', tid=10))
        self.identity = dict(hClient=1, client_generation=1, group=dict(handle=4, generation=4),
                             device=dict(handle=2, generation=2), subdevice=dict(handle=3, generation=3))
        self.member = dict(channel=dict(handle=6, generation=6, parent=4, parent_generation=4, **{'class': 0xc56f}))
        self.memories = [dict(client=1, handle=5, client_generation=1, generation=5,
                              parent=2, parent_generation=2, **{'class': 0x40})]
        self.vmas = [dict(start=0x100000, end=0x101000, perms='rw-s', file_offset=0,
                          path='/dev/nvidia0', inode=11)]

    def resolve(self):
        return resolve_channel(self.events, self.identity, self.member, self.memories, self.vmas)

    def test_complete_chain(self):
        self.assertEqual(self.resolve()['userd_cpu_address'], 0x100200)

    def test_freed_memory(self):
        self.events.append(dict(kind='free', seq=9, client=1, object=5, status=0))
        with self.assertRaisesRegex(ValueError, 'object_freed'): self.resolve()

    def test_recycled_handle(self):
        newer = copy.deepcopy(self.events[4]); newer['seq'] = 9; self.events.append(newer)
        with self.assertRaisesRegex(ValueError, 'reallocated'): self.resolve()

    def test_wrong_parent_generation(self):
        self.memories[0]['parent_generation'] += 1
        with self.assertRaisesRegex(ValueError, 'generation'): self.resolve()

    def test_wrong_owner_client(self):
        self.memories[0]['client'] = 2
        with self.assertRaisesRegex(ValueError, 'owned_memory'): self.resolve()

    def test_closed_and_reused_fd(self):
        self.events[-1]['seq'] = 9
        self.events.insert(-1, dict(kind='close', seq=8, fd=9, rc=0))
        with self.assertRaisesRegex(ValueError, 'mapping_missing'): self.resolve()

    def test_map_fd_replaced_by_other_device(self):
        self.events[-1]['fd_rdev'] = 3
        with self.assertRaisesRegex(ValueError, 'mapping_missing'): self.resolve()

    def test_partial_unmap(self):
        self.events.append(dict(kind='munmap', seq=9, address=0x100800, length=512, rc=0))
        with self.assertRaisesRegex(ValueError, 'mapping_missing'): self.resolve()

    def test_map_fixed_replacement(self):
        self.events.append(dict(kind='mmap', seq=9, fd=-1, flags=16, address=0x100000, length=4096, rc=0))
        with self.assertRaisesRegex(ValueError, 'mapping_missing'): self.resolve()

    def test_mremap_overwrites_destination(self):
        self.events.append(dict(kind='mremap', seq=9, address=0x200000, length=4096,
                                new_address=0x100000, new_length=4096, rc=0))
        with self.assertRaisesRegex(ValueError, 'mapping_missing'): self.resolve()

    def test_userd_outside_mapping(self):
        self.events[5]['userd_offsets'][0] = 4096
        with self.assertRaisesRegex(ValueError, 'mapping_missing'): self.resolve()

    def test_missing_read_permission(self):
        self.vmas[0]['perms'] = '-w-s'
        with self.assertRaisesRegex(ValueError, 'mapping_missing'): self.resolve()

    def test_ambiguous_aliases(self):
        setup, mapping = copy.deepcopy(self.events[-2:])
        setup.update(seq=9, map_fd=10); mapping.update(seq=10, fd=10, address=0x200000)
        self.events.extend([setup, mapping]); vma = dict(self.vmas[0]); vma.update(start=0x200000, end=0x201000)
        self.vmas.append(vma)
        with self.assertRaisesRegex(ValueError, 'ambiguous'): self.resolve()

    def test_other_subdevice_is_not_silently_selected(self):
        self.events[5]['userd_handles'] = [0, 5]
        with self.assertRaisesRegex(ValueError, 'ambiguous_userd'): self.resolve()

    def test_serialized_channel_allocation(self):
        self.events[5]['flags'] = 1
        with self.assertRaisesRegex(ValueError, 'encoding'): self.resolve()

    def test_failed_mmap(self):
        self.events[-1]['rc'] = -1
        with self.assertRaisesRegex(ValueError, 'mapping_missing'): self.resolve()


if __name__ == '__main__': unittest.main()
