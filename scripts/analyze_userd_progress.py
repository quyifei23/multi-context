#!/usr/bin/env python3
"""Validate a live owned read plan, or analyze bounded USERD field observations.

The parent validates metadata only: no GPU access, new mapping or remote memory
read. Only the originating process dereferences its already-mapped USERD.
"""
import argparse
import csv
import json
from pathlib import Path
from analyze_userd_mapping import require, read_observations, snapshot, SOURCE


def make_plan(directory, expected_pid):
    owner, seq = map(int, (directory / 'ready').read_text().split())
    require(owner == expected_pid, 'live_owner_pid_mismatch')
    rows = read_observations(directory)
    require(rows[-1]['kind'] == 'mark' and rows[-1].get('label') == 'after_warmup' and
            rows[-1]['seq'] == seq, 'live_snapshot_not_current')
    identity, bindings, _, _ = snapshot(directory, 'after_warmup', rows)
    require(identity['owner_pid'] == owner, 'live_identity_owner_mismatch')
    # Parent process is still waiting on this exact Popen child. Check its live
    # VMAs too, without reading any memory. The child checks sequence again.
    maps = Path(f'/proc/{owner}/maps').read_text()
    live_nvidia = [line for line in maps.splitlines() if '/dev/nvidia' in line]
    saved_nvidia = [line for line in (directory / 'after_warmup.maps').read_text().splitlines() if line]
    require(live_nvidia == saved_nvidia, 'live_vma_changed')
    lines = [f"USERD_READ_PLAN_V1 {owner} {seq} {identity['registry_instance']} {len(bindings)}"]
    for i, (member, b) in enumerate(zip(identity['members'], bindings)):
        c = member['channel']
        lines.append(f"{i} {c['handle']} {c['generation']} {b['userd_cpu_address']} {b['gpfifo_entries']}")
    return '\n'.join(lines) + '\n'


def publish_plan(directory, expected_pid):
    plan = make_plan(directory, expected_pid)
    require(not (directory / 'read.plan').exists(), 'read_plan_already_exists')
    with (directory / 'read.plan.tmp').open('x') as f:
        f.write(plan)
    (directory / 'read.plan.tmp').rename(directory / 'read.plan')


def validate_samples(rows, channels):
    samples = [r for r in rows if r['kind'] == 'userd_sample']
    require(0 < len(samples) <= 256, 'read_budget_or_missing_samples')
    require([s['round'] for s in samples] == list(range(len(samples))), 'lost_read_round')
    require(samples[-1]['end_ns'] - samples[0]['begin_ns'] < 3_000_000_000, 'read_deadline_exceeded')
    labels = [s['label'] for s in samples]
    expected = (['idle_before'] * 8 + ['before_begin_event', 'after_begin_event', 'before_launch', 'after_launch'] +
                ['launch_no_api'] * 8 + ['after_end_event'])
    require(labels[:len(expected)] == expected, 'unexpected_submission_sequence')
    queries = samples[len(expected):-10]
    require(queries and all(q['label'] == 'event_query' for q in queries) and
            all(q['api_result'] == 600 for q in queries[:-1]) and queries[-1]['api_result'] == 0,
            'completion_not_proven')
    require(labels[-10:] == ['after_elapsed_time', 'after_dtoh'] + ['idle_after'] * 8, 'missing_post_completion_reads')
    api_labels = {'after_begin_event', 'after_launch', 'after_end_event', 'event_query', 'after_elapsed_time', 'after_dtoh'}
    previous = 0
    for s in samples:
        require(previous <= s['begin_ns'] <= s['end_ns'], 'overlapping_sample_rounds')
        previous = s['end_ns']
        if s['label'] in api_labels:
            require(0 < s['api_begin_ns'] <= s['api_end_ns'] <= s['begin_ns'], 'invalid_api_bracket')
            require(s['api_result'] in ((0, 600) if s['label'] == 'event_query' else (0,)), 'cuda_api_error')
        else:
            require(s['api_begin_ns'] == s['api_end_ns'] == 0, 'unexpected_api_in_readonly_window')
        require([c['ordinal'] for c in s['channels']] == list(range(channels)), 'read_member_mismatch')
        end = s['begin_ns']
        for c in s['channels']:
            require(end <= c['begin_ns'] <= c['end_ns'] <= s['end_ns'], 'invalid_field_read_bracket')
            end = c['end_ns']
            require(all(isinstance(c[k], int) and 0 <= c[k] <= 0xffffffff for k in
                        ('get_first', 'put_first', 'get_second', 'put_second')), 'invalid_u32_read')
    return samples, queries


def analyze(directory):
    rows = read_observations(directory)
    a, first, _, num_a = snapshot(directory, 'after_warmup', rows)
    b, second, _, num_b = snapshot(directory, 'after_tiny', rows)
    require(a == b and first == second, 'binding_or_mapping_changed_across_read_window')
    start = next(r['seq'] for r in rows if r.get('label') == 'after_warmup')
    finish = [r for r in rows if r['kind'] == 'mark' and r.get('label') == 'reads_finished']
    require(len(finish) == 1, 'read_window_not_finished')
    middle = [r for r in rows if start < r['seq'] < finish[0]['seq']]
    require(middle and all(r['kind'] == 'userd_sample' for r in middle), 'audit_changed_while_reading')
    samples, queries = validate_samples(rows, len(first))
    require(samples == middle, 'read_outside_validated_window')
    tiny = json.loads((directory / 'tiny.json').read_text())
    launch = next(s for s in samples if s['label'] == 'after_launch')
    require(tiny['result_verified'] and tiny['launch_sequence'] == 2 and
            tiny['end_gpu_ns'] >= tiny['start_gpu_ns'] and tiny['event_ms'] > 0 and
            (tiny['launch_begin_ns'], tiny['launch_end_ns']) == (launch['api_begin_ns'], launch['api_end_ns']) and
            tiny['complete_observed_ns'] == queries[-1]['api_end_ns'], 'tiny_result_or_timing_mismatch')
    # Fixed public schema. All timestamps are relative to this run's launch API
    # entry; ring indices are data, while handles/addresses/UUID/PID stay local.
    origin = tiny['launch_begin_ns']
    public = []
    for s in samples:
        for c, binding in zip(s['channels'], first):
            public.append(dict(round=s['round'], phase=s['label'], channel_ordinal=c['ordinal'],
                read_begin_rel_ns=c['begin_ns'] - origin, read_end_rel_ns=c['end_ns'] - origin,
                get_first=c['get_first'], put_first=c['put_first'], get_second=c['get_second'], put_second=c['put_second'],
                repeated_values_equal=c['get_first'] == c['get_second'] and c['put_first'] == c['put_second'],
                gpfifo_entries=binding['gpfifo_entries'],
                api_begin_rel_ns=s['api_begin_ns'] - origin if s['api_begin_ns'] else '',
                api_end_rel_ns=s['api_end_ns'] - origin if s['api_end_ns'] else '', api_result=s['api_result']))
    channel_results = []
    for i, binding in enumerate(first):
        values = [s['channels'][i] for s in samples]
        phases = {}
        for label in dict.fromkeys(s['label'] for s in samples):
            selected = [s['channels'][i] for s in samples if s['label'] == label]
            phases[label] = {f'{field}_observed': sorted({c[key] for c in selected for key in (f'{field}_first', f'{field}_second')})
                             for field in ('get', 'put')}
        get_values = {c[k] for c in values for k in ('get_first', 'get_second')}
        put_values = {c[k] for c in values for k in ('put_first', 'put_second')}
        channel_results.append(dict(channel_ordinal=i, gpfifo_entries=binding['gpfifo_entries'],
            get_changed=len(get_values) > 1, put_changed=len(put_values) > 1,
            values_within_ring=all(v < binding['gpfifo_entries'] for v in get_values | put_values), phases=phases))
    summary = dict(classification='bounded_userd_fields_observed', source_commit=SOURCE,
        verified_channels=len(first), same_binding_across_read_window=True,
        rounds=len(samples), volatile_u32_reads=len(samples) * len(first) * 4,
        observed_window_ns=samples[-1]['end_ns'] - samples[0]['begin_ns'],
        project_rm_controls=0, extra_gpu_mappings=0, rewind_trials=0, explicit_context_destroy_calls=0,
        cuda_result_verified=True, event_query_not_ready=sum(q['api_result'] == 600 for q in queries),
        launch_api_ns=tiny['launch_end_ns'] - origin,
        complete_observed_rel_ns=tiny['complete_observed_ns'] - origin,
        event_ms_including_instrumentation_gaps=tiny['event_ms'],
        kernel_internal_duration_ns=tiny['end_gpu_ns'] - tiny['start_gpu_ns'],
        observer_bridge_matching_ioctl_counts=[num_a, num_b],
        repeated_value_changes=sum(not r['repeated_values_equal'] for r in public), channels=channel_results,
        sentinel_entry_mapping_established=False, hardware_consumption_semantics_verified=False,
        limits=['USERD reads are sequential, not atomic across fields or channels',
                'LFENCE orders CPU reads; it does not establish GPU freshness or RAMFC equivalence',
                'No ring contents decoded, no per-launch entry count or sentinel mapping established',
                'Instrumentation deliberately inserts host gaps; not a latency benchmark',
                'Lifecycle coverage is limited to interposed libc calls plus idle live VMA checks'])
    return summary, public


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence', type=Path)
    p.add_argument('--public-dir', type=Path, required=True)
    a = p.parse_args()
    summary, rows = analyze(a.evidence)
    a.public_dir.mkdir(parents=True, exist_ok=True)
    with (a.public_dir / 'summary.json').open('x') as f:
        json.dump(summary, f, indent=2); f.write('\n')
    with (a.public_dir / 'samples.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('channels', 'limits')}, indent=2))


if __name__ == '__main__':
    main()
