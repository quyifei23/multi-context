#!/usr/bin/env python3
"""Offline, fail-closed join of owned RM objects and already-existing CPU mappings.

Reads evidence files only. Does not open a GPU, map memory, or dereference the
reported address. Public outputs contain no handles, addresses, PID, fd or UUID.
"""
import argparse
import csv
import json
from pathlib import Path

SOURCE = 'db0c4e65c8e34c678d745ddb1317f53f90d1072b'
CHANNEL = 0xc56f
USERD_BYTES = 512


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ok(e):
    return e.get('rc', 0) == 0 and e.get('status', 0) == 0


def birth(events, client, handle):
    candidates = [e for e in events if e['kind'] in ('alloc', 'alloc_memory') and ok(e)
                  and e.get('object') == handle and (e.get('client') == client or
                  (handle == client and e.get('client') == 0))]
    require(candidates, 'owned_allocation_missing')
    e = candidates[-1]
    require(not e.get('flags', 0) or e['kind'] == 'alloc_memory', 'unknown_allocation_encoding')
    require(not any(x['kind'] == 'free' and ok(x) and x.get('client') == client and
                    x.get('object') == handle and x['seq'] > e['seq'] for x in events), 'object_freed')
    return e


def fd_signature(e, prefix=''):
    return tuple(e.get(prefix + 'fd_' + key) for key in ('dev', 'ino', 'rdev', 'path'))


def overlaps(start, size, other, other_size):
    return start < other + other_size and other < start + size


def resolve_channel(events, identity, member, memories, vmas):
    client = identity['hClient']
    channel = member['channel']
    require(channel['class'] == CHANNEL and channel['generation'] > 0, 'channel_class_or_generation')
    require(channel['parent'] == identity['group']['handle'] and
            channel['parent_generation'] == identity['group']['generation'], 'channel_parent_generation')
    ca = birth(events, client, channel['handle'])
    require(ca['class'] == CHANNEL and ca['parent'] == channel['parent'], 'channel_allocation_binding')
    require('userd_handles' in ca and len(ca['userd_handles']) == len(ca['userd_offsets']), 'channel_payload_missing')
    # This probe is explicitly one physical GPU/subdevice; never choose a slot.
    require([i for i, h in enumerate(ca['userd_handles']) if h] == [0], 'ambiguous_userd_subdevice')
    memory, offset = ca['userd_handles'][0], ca['userd_offsets'][0]
    records = [m for m in memories if m['client'] == client and m['handle'] == memory]
    require(len(records) == 1, 'owned_memory_record_missing_or_ambiguous')
    m = records[0]
    require(m['client_generation'] == identity['client_generation'] and m['generation'] > 0 and
            m['parent'] == identity['device']['handle'] and
            m['parent_generation'] == identity['device']['generation'], 'memory_owner_generation_or_device')
    ma = birth(events, client, memory)
    require(ma['class'] == m['class'] and ma['parent'] == m['parent'] and ma['seq'] < ca['seq'], 'memory_reallocated_or_wrong_class')
    for handle in (client, identity['device']['handle'], identity['subdevice']['handle'], identity['group']['handle']):
        birth(events, client, handle)
    require(offset % USERD_BYTES == 0 and ca['gpfifo_entries'] > 0, 'invalid_channel_layout')
    candidates = []
    for setup in events:
        if setup['kind'] != 'map_memory' or not ok(setup) or setup.get('client') != client or setup.get('memory') != memory:
            continue
        if setup['seq'] <= ma['seq'] or setup['device'] not in (identity['device']['handle'], identity['subdevice']['handle']):
            continue
        # Restrict this first experiment to complete, page-aligned, readable
        # normal memory-object mappings. No physical-address interpretation.
        if setup['offset'] % 4096 or setup['length'] % 4096 or (setup['flags'] & 3) not in (0, 1):
            continue
        if not (setup['offset'] <= offset and offset + USERD_BYTES <= setup['offset'] + setup['length']):
            continue
        mm = None
        for e in events:
            if e['seq'] <= setup['seq']:
                continue
            if e['kind'] == 'close' and e.get('fd') == setup['map_fd']:
                break  # The fd number alone cannot survive close/reuse.
            if e['kind'] in ('map_memory', 'alloc_memory') and e.get('map_fd') == setup['map_fd'] and ok(e):
                break  # A different RM mmap context superseded this one.
            if e['kind'] == 'mmap' and e.get('fd') == setup['map_fd']:
                mm = e
                break
        if mm is None or not ok(mm) or fd_signature(mm) != fd_signature(setup, 'map_'):
            continue
        if mm['tid'] != setup['tid'] or mm['offset'] != 0 or mm['length'] != setup['length'] or not mm['prot'] & 1 or not mm['flags'] & 1:
            continue
        invalid = False
        for e in events:
            if e['seq'] <= mm['seq'] or not ok(e):
                continue
            if e['kind'] == 'unmap_memory' and e.get('client') == client and e.get('memory') == memory:
                invalid = True
            if e['kind'] == 'munmap' or (e['kind'] == 'mmap' and e['flags'] & 16):
                invalid |= overlaps(mm['address'], mm['length'], e['address'], e['length'])
            if e['kind'] == 'mremap':
                invalid |= overlaps(mm['address'], mm['length'], e['address'], e['length'])
                invalid |= overlaps(mm['address'], mm['length'], e['new_address'], e['new_length'])
        if invalid:
            continue
        cpu = mm['address'] + offset - setup['offset']
        valid_vmas = [v for v in vmas if v['start'] <= cpu and cpu + USERD_BYTES <= v['end']
                      and v['perms'].startswith('r') and v['perms'].endswith('s')
                      and v['path'] == mm['fd_path'] and v['inode'] == mm['fd_ino']
                      and v['file_offset'] + cpu - v['start'] == cpu - mm['address']]
        if len(valid_vmas) == 1:
            candidates.append((setup, mm, cpu))
    require(len(candidates) == 1, 'mapping_missing_stale_or_ambiguous')
    setup, mm, cpu = candidates[0]
    return {'channel_generation': channel['generation'], 'memory_generation': m['generation'],
            'channel_alloc_seq': ca['seq'], 'memory_alloc_seq': ma['seq'], 'rm_map_seq': setup['seq'],
            'mmap_seq': mm['seq'], 'userd_offset': offset, 'mapping_bytes': mm['length'],
            'gpfifo_entries': ca['gpfifo_entries'], 'userd_cpu_address': cpu,
            'gpfifo_gpu_va': ca['gpfifo_gpu_va'], 'mapping_verified': True}


def parse_maps(text):
    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        span, perms, offset, dev, inode, path = line.split(maxsplit=5)
        start, end = [int(x, 16) for x in span.split('-')]
        result.append({'start': start, 'end': end, 'perms': perms, 'file_offset': int(offset, 16),
                       'device': dev, 'inode': int(inode), 'path': path})
    return result


def snapshot(directory, label, rows):
    marks = [r for r in rows if r['kind'] == 'mark' and r.get('label') == label]
    require(len(marks) == 1, 'snapshot_marker_missing')
    events = [r for r in rows if r['seq'] <= marks[0]['seq']]
    load = lambda suffix: json.loads((directory / f'{label}.{suffix}.json').read_text())
    identity, memories, diagnostics = load('identity'), load('memory'), load('diagnostics')
    st = diagnostics['rm_state']
    require(all(st[x] for x in ('runtime_matches', 'static_abi_reviewed', 'observation_enabled', 'group_binding_valid')), 'bridge_identity_gate_failed')
    require(st['project_controls_attempted'] == 0 and not st['group_get_info_verified'] and
            not st['bridge_gpu_uuid_query_attempted'], 'unexpected_project_control')
    require(diagnostics['capture_inventory'].startswith('incomplete=no '), 'bridge_capture_incomplete')
    require(identity['source_commit'] == SOURCE and identity['profile'] == '595.58.03', 'profile_mismatch')
    require(memories['source_commit'] == SOURCE and memories['profile'] == '595.58.03' and
            memories['pid'] == identity['owner_pid'] == st['pid'] and
            memories['registry_instance'] == identity['registry_instance'], 'memory_snapshot_owner_mismatch')
    require(all(e['pid'] == identity['owner_pid'] for e in events), 'wrong_process_observation')
    original = [e for e in events if 'number' in e]
    known_mapping_abis = {0x27: 'alloc_memory', 0x4e: 'map_memory', 0x4f: 'unmap_memory'}
    require(all(e['kind'] == known_mapping_abis[e['number']] for e in original
                if e['number'] in known_mapping_abis and e['rc'] == 0), 'unknown_mapping_abi')
    bridged = diagnostics['observed_ioctls']
    require([(e['fd'], e['number'], e['size'], e['rc']) for e in original] ==
            [(e['fd'], e['number'], e['size'], e['syscall_return']) for e in bridged], 'observer_bridge_envelope_order_mismatch')
    # Independent physical identity from normal libcuda RM responses. The
    # operational FIFO gate remains untouched/unarmed; no query is added here.
    client, subdevice = identity['hClient'], identity['subdevice']['handle']
    sub_birth = birth(events, client, subdevice)
    gids = [e for e in events if e.get('client') == client and e.get('object') == subdevice and
            e.get('command') == 0x2080014a and ok(e) and e['seq'] > sub_birth['seq'] and
            e.get('flags') == 0 and e.get('gid_flags') == 2 and e.get('gid_length') == 16]
    require(gids and all(e['gid_bytes'] == identity['gpu_uuid_hex'] for e in gids), 'passive_physical_uuid_unverified')
    vmas = parse_maps((directory / f'{label}.maps').read_text())
    bindings = [resolve_channel(events, identity, member, memories['objects'], vmas) for member in identity['members']]
    require(bindings, 'empty_group')
    addresses = [b['userd_cpu_address'] for b in bindings]
    require(len(set(addresses)) == len(addresses), 'channel_userd_regions_alias')
    return identity, bindings, len(gids), len(original)


def analyze(directory):
    rows = [json.loads(x) for x in (directory / 'observations.jsonl').read_text().splitlines()]
    require(rows and rows[0]['kind'] == 'start' and rows[0]['bridge_hook_next'], 'observer_not_initialized')
    require([r['seq'] for r in rows] == list(range(1, len(rows) + 1)), 'lost_observation')
    require(not any(r['kind'].startswith('unknown') or r.get('label') == 'failed' for r in rows), 'capture_failed')
    require(all(r['end_ns'] >= r['begin_ns'] for r in rows), 'invalid_clock_boundaries')
    a, first, gid_a, num_a = snapshot(directory, 'after_warmup', rows)
    b, second, gid_b, num_b = snapshot(directory, 'after_tiny', rows)
    require(a == b and first == second, 'binding_or_mapping_changed_across_snapshots')
    public = [{k: r[k] for k in ('userd_offset', 'mapping_bytes', 'gpfifo_entries', 'mapping_verified')}
              | {'channel_ordinal': i, 'owner_generation_verified': True, 'physical_uuid_verified': True,
                 'live_readable_vma_verified': True} for i, r in enumerate(first)]
    summary = {'classification': 'already_mapped_state_available', 'scope': 'owned_A_compute_TSG_USERD_binding_only',
               'snapshots': 2, 'verified_channels': len(first), 'channels_in_target_group': len(a['members']),
               'same_mapping_across_snapshots': True, 'passive_uuid_responses_per_snapshot': [gid_a, gid_b],
               'observer_bridge_matching_ioctl_counts': [num_a, num_b], 'observations': len(rows),
               'project_rm_controls': 0, 'get_put_memory_reads': 0, 'sentinel_entry_mapping_established': False,
               'rewind_trials': 0, 'source_commit': SOURCE,
               'limits': ['libc-interposed calls only; direct/hidden syscalls are outside capture coverage',
                          'USERD GET/PUT contents, update semantics and RAMFC state not sampled',
                          'CUDA logical submission to GPFIFO entry mapping not established']}
    return summary, public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--public-dir', type=Path, required=True)
    args = parser.parse_args()
    summary, records = analyze(args.evidence)
    args.public_dir.mkdir(parents=True, exist_ok=True)
    with (args.public_dir / 'summary.json').open('x') as f:
        json.dump(summary, f, indent=2); f.write('\n')
    with (args.public_dir / 'bindings.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
