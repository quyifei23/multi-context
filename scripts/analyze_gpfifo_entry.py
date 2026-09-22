#!/usr/bin/env python3
"""Validate existing ring mapping before reads, then attribute opaque entry changes."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from analyze_userd_mapping import require, read_observations, snapshot, parse_maps, overlaps, SOURCE
from analyze_userd_progress import analyze as analyze_progress
from gpfifo_binding import resolve_ring


def active_channel(rows):
    pre = [r for r in rows if r['kind'] == 'userd_sample' and r['label'] == 'before_launch']
    post = [r for r in rows if r['kind'] == 'userd_sample' and r['label'] == 'after_launch']
    require(len(pre) == len(post) == 1, 'discovery_launch_missing')
    changed = []
    for i, (a, b) in enumerate(zip(pre[0]['channels'], post[0]['channels'])):
        require(a['put_first'] == a['put_second'] and b['put_first'] == b['put_second'], 'unstable_discovery_put')
        if a['put_first'] != b['put_first']:
            changed.append(i)
    require(len(changed) == 1, 'active_channel_missing_or_ambiguous')
    return changed[0]


def binding_at(directory, label, rows, ordinal):
    identity, userds, _, _ = snapshot(directory, label, rows)
    mark = next(r for r in rows if r['kind'] == 'mark' and r.get('label') == label)
    events = [r for r in rows if r['seq'] <= mark['seq']]
    binding = resolve_ring(events, identity, identity['members'][ordinal], userds[ordinal],
                           parse_maps((directory / f'{label}.maps').read_text()))
    require(all(not overlaps(binding['ring_cpu_address'], binding['ring_entries'] * 8,
                             b['userd_cpu_address'], 512) for b in userds), 'ring_aliases_userd_fields')
    return identity, userds, binding, mark


def publish_ring_plan(directory, expected_pid):
    owner, sequence = map(int, (directory / 'ring_ready').read_text().split())
    require(owner == expected_pid, 'ring_owner_pid_mismatch')
    # Reuse the original progress protocol and all ownership/generation/UUID gates.
    analyze_progress(directory)
    rows = read_observations(directory)
    require(rows[0].get('gpfifo_metadata') == 1, 'ring_metadata_not_enabled')
    ordinal = active_channel(rows)
    identity, _, b, mark = binding_at(directory, 'after_tiny', rows, ordinal)
    require(rows[-1] == mark and mark['seq'] == sequence and identity['owner_pid'] == owner, 'ring_snapshot_not_current')
    live = [line for line in Path(f'/proc/{owner}/maps').read_text().splitlines() if '/dev/nvidia' in line]
    require(live == [line for line in (directory / 'after_tiny.maps').read_text().splitlines() if line], 'ring_live_vma_changed')
    c = identity['members'][ordinal]['channel']
    last = [r for r in rows if r['kind'] == 'userd_sample'][-1]['channels'][ordinal]
    require(last['put_first'] == last['put_second'] and 0 <= last['put_first'] < b['ring_entries'], 'preparation_put_unknown')
    plan = (f"GPFIFO_READ_PLAN_V2 {owner} {sequence} {identity['registry_instance']} {ordinal} "
            f"{c['handle']} {c['generation']} {b['ring_cpu_address']} {b['ring_entries']} {last['put_first']}\n")
    require(not (directory / 'ring.plan').exists(), 'ring_plan_already_exists')
    with (directory / 'ring.binding.json').open('x') as f:
        json.dump(b, f, indent=2); f.write('\n')
    with (directory / 'ring.plan.tmp').open('x') as f:
        f.write(plan)
    (directory / 'ring.plan.tmp').rename(directory / 'ring.plan')


def stable_frame(frame, counts):
    n = len(frame['ring_first'])
    require(n and frame['ring_first'] == frame['ring_second'] and
            all(re.fullmatch('[0-9a-f]{16}', v) for v in frame['ring_first']), 'ring_double_read_unstable')
    puts = []
    previous_end = frame['begin_ns']
    for key in ('userd_before', 'userd_after'):
        fields = frame[key]
        require([c['ordinal'] for c in fields] == list(range(len(counts))), 'ring_userd_members_missing')
        for c, count in zip(fields, counts):
            require(previous_end <= c['begin_ns'] <= c['end_ns'] <= frame['end_ns'], 'ring_field_clock_invalid')
            previous_end = c['end_ns']
            require(0 <= c['get'] < count and 0 <= c['put'] < count, 'ring_pointer_out_of_range')
        puts.append([c['put'] for c in fields])
    require(puts[0] == puts[1], 'publication_changed_during_ring_snapshot')
    return puts[0]


def attribute_launch(pre, post, active, entries):
    before, after = pre['userd_before'], post['userd_after']
    require(len(pre['ring_first']) == len(post['ring_first']) == entries, 'ring_entry_count_mismatch')
    require(0 <= active < len(before) == len(after), 'active_ordinal_invalid')
    puts_a, puts_b = [c['put'] for c in before], [c['put'] for c in after]
    moved = [i for i, (a, b) in enumerate(zip(puts_a, puts_b)) if a != b]
    require(moved == [active], 'multiple_or_wrong_active_channels')
    slot = puts_a[active]
    require(puts_b[active] == (slot + 1) % entries, 'put_delta_not_one_entry')
    changed = [i for i, (a, b) in enumerate(zip(pre['ring_first'], post['ring_first'])) if a != b]
    require(changed == [slot], 'changed_slots_do_not_uniquely_match_put')
    return slot


def analyze(directory):
    analyze_progress(directory)
    rows = read_observations(directory)
    ordinal = active_channel(rows)
    a, userds, b, start = binding_at(directory, 'after_tiny', rows, ordinal)
    final, final_userds, final_b, _ = binding_at(directory, 'after_entries', rows, ordinal)
    require(a == final and userds == final_userds and b == final_b, 'ring_binding_changed')
    require(json.loads((directory / 'ring.binding.json').read_text()) == b, 'ring_plan_binding_mismatch')
    ev = json.loads((directory / 'entries.json').read_text())
    wrap = ev.get('wrap_requested', False)
    preparation = ev.get('preparation_launches', 0)
    plan = (directory / 'ring.plan').read_text().split()
    c = a['members'][ordinal]['channel']
    expected_plan = [a['owner_pid'], start['seq'], a['registry_instance'], ordinal,
                     c['handle'], c['generation'], b['ring_cpu_address'], b['ring_entries']]
    require(plan[0] in ('GPFIFO_READ_PLAN_V1', 'GPFIFO_READ_PLAN_V2') and
            list(map(int, plan[1:9])) == expected_plan and len(plan) == (10 if plan[0].endswith('V2') else 9), 'ring_plan_identity_mismatch')
    if plan[0].endswith('V2'):
        require(int(plan[9]) == ev['initial_put'], 'ring_plan_initial_put_mismatch')
    finish = [r for r in rows if r['kind'] == 'mark' and r.get('label') == 'ring_reads_finished']
    require(len(finish) == 1, 'ring_window_not_finished')
    window = [r for r in rows if start['seq'] < r['seq'] < finish[0]['seq']]
    expected = ([('ring_sample', 'before_fill')] if wrap else []) + [('ring_sample', 'idle')]
    for _ in range(2):
        expected += [('ring_sample', 'before_launch'), ('ring_sample', 'after_launch')]
        expected += [('ring_progress', 'entry_no_api')] * 8 + [('ring_sample', 'after_no_api')]
    require([(r['kind'], r.get('label')) for r in window] == expected, 'ring_window_sequence_or_intervening_syscall')
    require([r['round'] for r in window] == list(range(len(window))) and
            window[-1]['end_ns'] - window[0]['begin_ns'] < 3_000_000_000, 'ring_budget_or_sequence')
    require(all(x['end_ns'] <= y['begin_ns'] for x, y in zip(window, window[1:])), 'ring_windows_overlap')
    counts = [x['gpfifo_entries'] for x in userds]
    frames = [r for r in window if r['kind'] == 'ring_sample']
    entries = b['ring_entries']
    for f in frames:
        require(len(f['ring_first']) == entries and f['begin_ns'] <= f['end_ns'], 'ring_snapshot_length')
        stable_frame(f, counts)
    require(ev['last_result_verified'] and len(ev['launches']) == 2 and ev['rounds'] == len(window), 'isolated_launch_result_missing')
    if wrap:
        require(0 <= preparation < entries and preparation == entries - 1 - ev['initial_put'] and
                stable_frame(frames[0], counts)[ordinal] == ev['initial_put'] and
                stable_frame(frames[1], counts)[ordinal] == entries - 1, 'wrap_preparation_failed')
        frames = frames[1:]
    else:
        require(preparation == 0, 'unrequested_preparation')
    public = []
    origin = ev['launches'][0]['begin_ns']
    old = frames[0]
    digest = lambda values: hashlib.sha256(b''.join(bytes.fromhex(x)[::-1] for x in values)).hexdigest()
    for index in (1, 2):
        pre, post, settled = [f for f in frames if f['launch_index'] == index]
        require(pre['ring_first'] == old['ring_first'] and stable_frame(pre, counts) == stable_frame(old, counts), 'ring_changed_between_launches')
        call = ev['launches'][index - 1]
        require(call['index'] == index and call['launch_sequence'] == index + 2 + preparation and call['cuda_result'] == 0 and
                pre['end_ns'] <= call['begin_ns'] <= call['end_ns'] <= post['begin_ns'] and
                (call['begin_ns'], call['end_ns'], 0) == (post['api_begin_ns'], post['api_end_ns'], post['api_result']), 'isolated_api_boundary_mismatch')
        slot = attribute_launch(pre, post, ordinal, entries)
        require(settled['ring_first'] == post['ring_first'] and stable_frame(settled, counts) == stable_frame(post, counts), 'deferred_ring_or_put_changes')
        progress = [r for r in window if post['seq'] < r['seq'] < settled['seq']]
        get_values = set()
        for r in progress:
            require(r['api_begin_ns'] == r['api_end_ns'] == 0 and r['api_result'] == -1, 'unexpected_cuda_api_in_ring_gap')
            require([c['ordinal'] for c in r['channels']] == list(range(len(counts))), 'progress_members_missing')
            for c, count, put in zip(r['channels'], counts, stable_frame(post, counts)):
                require(c['put_first'] == c['put_second'] == put and
                        0 <= c['get_first'] < count and 0 <= c['get_second'] < count, 'progress_changed_put_or_invalid_get')
            c = r['channels'][ordinal]; get_values.update((c['get_first'], c['get_second']))
        public.append(dict(launch_index=index, channel_ordinal=ordinal, ring_entries=entries, slot_index=slot,
            put_before=slot, put_after=(slot + 1) % entries, put_delta_mod=1, changed_slot_count=1,
            ring_before_sha256=digest(pre['ring_first']), ring_after_sha256=digest(post['ring_first']),
            entry_before_sha256=digest([pre['ring_first'][slot]]), entry_after_sha256=digest([post['ring_first'][slot]]),
            api_begin_rel_ns=call['begin_ns'] - origin, api_end_rel_ns=call['end_ns'] - origin,
            pre_read_begin_rel_ns=pre['begin_ns'] - origin, pre_read_end_rel_ns=pre['end_ns'] - origin,
            post_read_begin_rel_ns=post['begin_ns'] - origin, post_read_end_rel_ns=post['end_ns'] - origin,
            no_api_get_values='|'.join(map(str, sorted(get_values))), unique_entry_binding=True))
        old = settled
    require(public[1]['slot_index'] == (public[0]['slot_index'] + 1) % entries, 'next_launch_not_next_slot')
    require(not wrap or [r['slot_index'] for r in public] == [entries - 1, 0], 'physical_wrap_not_observed')
    summary = dict(classification='isolated_launch_to_exact_entry_observed', source_commit=SOURCE,
        measured_launches=2, active_channel_ordinal=ordinal, channels_checked=len(userds), ring_entries=entries,
        mapped_slots=[r['slot_index'] for r in public], same_owner_generation_mapping=True,
        existing_ring_cpu_mapping_verified=True, gpu_va_memory_cpu_binding_verified=True,
        preparation_launches=preparation, ring_snapshots=len(frames) + int(wrap),
        ring_u64_loads=(len(frames) + int(wrap)) * entries * 2,
        project_rm_controls=0, extra_gpu_mappings=0, rewind_trials=0,
        physical_wrap_observed=public[1]['slot_index'] == 0, no_event_or_copy_between_measured_launches=True,
        last_tiny_result_verified=True, payload_semantics_decoded=False,
        limits=['Binding is for these isolated Driver launches, not arbitrary multi-operation epochs',
                'Sequential repeated reads are not atomic hardware snapshots',
                'GET does not establish kernel completion, lack of prefetch or rewind safety',
                'Only the captured libc paths and this client-created context-share/shared Memory layout are supported'])
    return summary, public


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence', type=Path)
    p.add_argument('--public-dir', type=Path, required=True)
    a = p.parse_args(); summary, rows = analyze(a.evidence)
    a.public_dir.mkdir(parents=True, exist_ok=True)
    with (a.public_dir / 'summary.json').open('x') as f:
        json.dump(summary, f, indent=2); f.write('\n')
    with (a.public_dir / 'entries.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
