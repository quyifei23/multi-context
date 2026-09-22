#!/usr/bin/env python3
"""Bind the complete owned compute ring set, then compare isolated Graph launches.

No node-to-entry assumption; unusual deltas and multi-channel changes are data.
Only lifecycle/ABI/ownership or structurally incomplete evidence fails the gate.
"""
import argparse
import csv
import json
from pathlib import Path
from analyze_userd_mapping import require, snapshot, read_observations, parse_maps, overlaps, SOURCE
from analyze_userd_progress import analyze as analyze_progress
from analyze_gpfifo_entry import active_channel, stable_frame
from gpfifo_binding import resolve_ring

CASES = ('direct_one', 'direct_three', 'graph_one', 'graph_three', 'graph_three_repeat')


def bind_set(directory, label, rows):
    identity, userds, _, envelopes = snapshot(directory, label, rows)
    mark = next(r for r in rows if r['kind'] == 'mark' and r.get('label') == label)
    events = [r for r in rows if r['seq'] <= mark['seq']]
    maps = parse_maps((directory / f'{label}.maps').read_text())
    bindings = [resolve_ring(events, identity, member, u, maps) for member, u in zip(identity['members'], userds)]
    for i, b in enumerate(bindings):
        require(all(not overlaps(b['ring_cpu_address'], b['ring_entries'] * 8, u['userd_cpu_address'], 512)
                    for u in userds), 'ring_set_aliases_userd')
        require(all(not overlaps(b['ring_cpu_address'], b['ring_entries'] * 8, c['ring_cpu_address'], c['ring_entries'] * 8)
                    for c in bindings[:i]), 'ring_set_aliases_another_ring')
    return identity, userds, bindings, mark, envelopes


def plan_text(identity, userds, bindings, sequence, active):
    lines = [f"GRAPH_RING_PLAN_V1 {identity['owner_pid']} {sequence} {identity['registry_instance']} {len(bindings)} {active}"]
    for i, (m, u, b) in enumerate(zip(identity['members'], userds, bindings)):
        c = m['channel']
        lines.append(f"{i} {c['handle']} {c['generation']} {u['userd_cpu_address']} {b['ring_cpu_address']} {b['ring_entries']}")
    return '\n'.join(lines) + '\n'


def publish_graph_plan(directory, expected_pid, condition):
    require(condition in CASES, 'unknown_graph_condition')
    owner, sequence = map(int, (directory / f'{condition}.ready').read_text().split())
    require(owner == expected_pid, 'graph_live_owner_mismatch')
    analyze_progress(directory)
    rows = read_observations(directory)
    require(rows[0].get('gpfifo_metadata') == 1, 'ring_metadata_not_enabled')
    identity, userds, bindings, mark, _ = bind_set(directory, 'graph_before_' + condition, rows)
    require(rows[-1] == mark and mark['seq'] == sequence and identity['owner_pid'] == owner, 'graph_live_snapshot_not_current')
    live = [s for s in Path(f'/proc/{owner}/maps').read_text().splitlines() if '/dev/nvidia' in s]
    require(live == [s for s in (directory / f'graph_before_{condition}.maps').read_text().splitlines() if s], 'graph_live_vma_changed')
    plan = plan_text(identity, userds, bindings, sequence, active_channel(rows))
    require(not (directory / f'{condition}.plan').exists(), 'graph_plan_already_exists')
    with (directory / f'{condition}.binding.json').open('x') as f:
        json.dump(bindings, f, indent=2); f.write('\n')
    with (directory / f'{condition}.plan.tmp').open('x') as f: f.write(plan)
    (directory / f'{condition}.plan.tmp').rename(directory / f'{condition}.plan')


def validate_frame(frame, counts):
    require(frame['begin_ns'] <= frame['end_ns'] and
            [r['ordinal'] for r in frame['rings']] == list(range(len(counts))), 'graph_ring_set_members_missing')
    for r, n in zip(frame['rings'], counts):
        require(len(r['ring_first']) == n, 'graph_ring_size_mismatch')
        stable_frame(dict(frame, **r), counts)
    return [u['put'] for u in frame['userd_after']]


def ring_interval(old, new, size):
    require(size > 0 and 0 <= old < size and 0 <= new < size, 'invalid_ring_index')
    return [(old + j) % size for j in range((new - old) % size)]


def compare_frames(pre, post, counts):
    pa, pb = validate_frame(pre, counts), validate_frame(post, counts)
    result = []
    for i, n in enumerate(counts):
        before, after = pre['rings'][i]['ring_first'], post['rings'][i]['ring_first']
        expected = ring_interval(pa[i], pb[i], n)
        changed = [j for j, (a, b) in enumerate(zip(before, after)) if a != b]
        missing, outside = sorted(set(expected) - set(changed)), sorted(set(changed) - set(expected))
        result.append(dict(channel_ordinal=i, ring_entries=n, put_before=pa[i], put_after=pb[i],
            put_delta_mod=len(expected), interval_slots=expected, changed_slots=changed,
            unchanged_slots_inside=missing, changed_slots_outside=outside,
            exact_interval_coverage=not missing and not outside,
            wrap_observed=bool(expected) and pb[i] < pa[i]))
    return result


def same_publication(a, b, counts):
    return validate_frame(a, counts) == validate_frame(b, counts) and all(
        x['ring_first'] == y['ring_first'] for x, y in zip(a['rings'], b['rings']))


def analyze(directory):
    analyze_progress(directory)
    rows = read_observations(directory); active = active_channel(rows)
    state = json.loads((directory / 'graph_state.json').read_text())
    require(state == json.loads((directory / 'graph_state_final.json').read_text()) and
            state['capture_instantiate_upload_warmup_complete'] and state['graph_one_nodes'] == 1 and
            state['graph_three_nodes'] == 3 and state['graph_three_edges'] == [[0, 1], [1, 2]] and
            state['graph_one_exec'] and state['graph_three_exec'] and state['graph_one_exec'] != state['graph_three_exec'], 'graph_identity_or_topology_mismatch')
    public, details, common, last, origin = [], [], None, None, None
    previous_gpu_starts = {}
    for index, name in enumerate(CASES):
        before = bind_set(directory, 'graph_before_' + name, rows)
        after = bind_set(directory, 'graph_after_' + name, rows)
        require(before[:3] == after[:3] and (common is None or common == before[:3]), 'graph_binding_changed')
        common = before[:3]; identity, userds, bindings = common
        require(json.loads((directory / f'{name}.binding.json').read_text()) == bindings and
                (directory / f'{name}.plan').read_text() == plan_text(identity, userds, bindings, before[3]['seq'], active), 'graph_plan_mismatch')
        counts = [b['ring_entries'] for b in bindings]
        scoped = [r for r in rows if before[3]['seq'] < r['seq'] < after[3]['seq']]
        finish = [r for r in scoped if r['kind'] == 'mark' and r.get('label') == 'graph_reads_finished']
        require(len(finish) == 1, 'graph_read_window_unfinished')
        window = [r for r in scoped if r['seq'] < finish[0]['seq']]
        require([(r['kind'], r.get('label'), r.get('case_index'), r.get('round')) for r in window] ==
                [('ring_set_sample', phase, index, j) for j, phase in enumerate(('idle', 'before_launch', 'after_launch', 'settled'))], 'graph_window_changed_or_incomplete')
        idle, pre, post, settled = window
        require(all(a['end_ns'] <= b['begin_ns'] for a, b in zip(window, window[1:])) and
                settled['end_ns'] - idle['begin_ns'] < 3_000_000_000, 'graph_window_budget_or_clock')
        for frame in window: validate_frame(frame, counts)
        launch = json.loads((directory / f'{name}.launch.json').read_text())
        output = json.loads((directory / f'{name}.result.json').read_text())
        require(launch['condition'] == name and launch['case_index'] == index and launch['rounds'] == 4 and
                launch['graph_exec'] == (0 if index < 2 else state['graph_one_exec'] if index == 2 else state['graph_three_exec']), 'graph_launch_identity_mismatch')
        calls = launch['calls']; expected_calls = 3 if index == 1 else 1
        require(len(calls) == expected_calls, 'graph_api_count_mismatch')
        boundary = pre['end_ns']
        for call in calls:
            require(call['api'] == ('cuLaunchKernel' if index < 2 else 'cuGraphLaunch') and call['result'] == 0 and
                    boundary <= call['begin_ns'] <= call['end_ns'] <= post['begin_ns'], 'graph_api_boundary_mismatch')
            boundary = call['end_ns']
        if origin is None: origin = calls[0]['begin_ns']
        require(output['verified'] and finish[0]['end_ns'] <= output['query_begin_ns'] <= output['query_end_ns'] <=
                output['copy_begin_ns'] <= output['copy_end_ns'] <= after[3]['begin_ns'] and output['query_count'] > 0, 'graph_completion_outside_window_missing')
        slots = [0] if index in (0, 2) else [1, 2, 3]
        require([n['slot'] for n in output['nodes']] == slots, 'graph_node_result_missing')
        for node in output['nodes']:
            require(node['token'] == 0x71300001 + node['slot'] and node['value'] == node['token'] ^ 0x5a5a5a5a and
                    node['end_gpu_ns'] >= node['start_gpu_ns'] > node['previous_start_gpu_ns'] > 0 and
                    node['previous_start_gpu_ns'] == previous_gpu_starts.get(node['slot'], node['previous_start_gpu_ns']), 'graph_node_result_stale_or_incorrect')
            previous_gpu_starts[node['slot']] = node['start_gpu_ns']
        preparation = launch['preparation_launches']
        if launch['wrap_requested'] and index == 3:
            require(preparation == counts[active] - 1 - validate_frame(idle, counts)[active] and
                    validate_frame(pre, counts)[active] == counts[active] - 1, 'graph_wrap_preparation_mismatch')
        else:
            require(preparation == 0 and same_publication(idle, pre, counts), 'graph_idle_publication_changed')
        changes = compare_frames(pre, post, counts)
        deferred = compare_frames(post, settled, counts)
        moved = [r['channel_ordinal'] for r in changes if r['put_delta_mod']]
        stable = same_publication(post, settled, counts)
        exact = bool(moved) and all(r['exact_interval_coverage'] for r in changes) and stable
        gap = compare_frames(last, idle, counts) if last is not None else []
        details.append(dict(condition=name, channels=changes, deferred_channels=deferred,
                            validation_gap_channels=gap, bounded_no_api_stable=stable))
        public.append(dict(condition=name, api=calls[0]['api'], api_calls=len(calls), kernel_nodes=len(slots),
            active_channels='|'.join(map(str, moved)), total_put_delta=sum(r['put_delta_mod'] for r in changes),
            changed_entry_count=sum(len(r['changed_slots']) for r in changes),
            entry_set='|'.join(f"{r['channel_ordinal']}:{slot}" for r in changes for slot in r['interval_slots']),
            complete_exclusive_observed=exact, bounded_no_api_stable=stable,
            wrap_observed=any(r['wrap_observed'] for r in changes),
            preparation_launches=preparation, node_outputs_correct_and_fresh=True,
            api_begin_rel_ns=calls[0]['begin_ns'] - origin, api_end_rel_ns=calls[-1]['end_ns'] - origin,
            before_begin_rel_ns=pre['begin_ns'] - origin, before_end_rel_ns=pre['end_ns'] - origin,
            after_begin_rel_ns=post['begin_ns'] - origin, after_end_rel_ns=post['end_ns'] - origin,
            completion_query_begin_rel_ns=output['query_begin_ns'] - origin,
            completion_query_end_rel_ns=output['query_end_ns'] - origin,
            validation_gap_put_delta=sum(r['put_delta_mod'] for r in gap),
            rm_envelopes_before=before[4], rm_envelopes_after=after[4]))
        last = settled
    repeat_follows = all(r['put_delta_mod'] == 0 and not r['changed_slots'] for r in details[-1]['validation_gap_channels'])
    graphs = public[2:]
    summary = dict(classification='graph_epoch_entry_sets_observed' if all(r['complete_exclusive_observed'] for r in graphs) else 'graph_epoch_entry_sets_unresolved',
        source_commit=SOURCE, owned_compute_channels=len(common[0]['members']), measured_graph_launches=3,
        same_graph_exec_repeated=True, repeat_entry_count_stable=public[3]['total_put_delta'] == public[4]['total_put_delta'],
        repeat_immediately_follows_prior_entry_set=repeat_follows,
        graph_wrap_observed=any(r['wrap_observed'] for r in graphs), all_node_outputs_correct_and_fresh=True,
        same_owner_generation_mapping=True, project_rm_controls=0, extra_gpu_mappings=0,
        graph_nodes_compute_only=True, graph_setup_outside_measurements=True,
        no_other_cuda_api_in_launch_windows=True, conditions=public, details=details,
        limits=['Empirical binding of warmed fixed GraphExecs on this stack, not a public CUDA queue API',
                'Full ring double reads are sequential, not atomic snapshots; no hardware consumption inference',
                'Only the captured libc paths and validated owned compute TSG are covered',
                'Stream queries and result copies occur after each measured window and their gaps are reported'])
    return summary, public


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('evidence', type=Path)
    p.add_argument('--public-dir', type=Path, required=True); a = p.parse_args()
    summary, rows = analyze(a.evidence); a.public_dir.mkdir(parents=True, exist_ok=True)
    with (a.public_dir / 'summary.json').open('x') as f: json.dump(summary, f, indent=2); f.write('\n')
    with (a.public_dir / 'entries.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('conditions','details','limits')}, indent=2))


if __name__ == '__main__': main()
