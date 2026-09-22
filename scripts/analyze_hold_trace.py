#!/usr/bin/env python3
"""Correlate this process's CUDA kernels, RM ranges and GPU context switches.

Switch IDs are inferred from isolated reference/warmup kernel intervals, never
by equating Nsight IDs to RM TSG IDs. Residency is not SM instruction activity.
"""
import argparse
import csv
import json
import math
import pathlib
import sqlite3
import sys

from analyze_nsys import analyze, read_csv


def main(args):
    metadata = pathlib.Path(args.csv).with_suffix(".metadata.csv")
    meta = {r["key"]: r["value"] for r in read_csv(metadata)}
    rows = analyze(args.sqlite, args.csv, metadata)
    samples = read_csv(args.csv)
    controls = read_csv(args.csv + ".controls.csv")
    key = lambda r: (r["target_ms"], r["condition"], r["trial"], r["order"])
    sample_by_key = {key(r): r for r in samples}
    control_by_key = {key(r): r for r in controls}
    if (len(samples) != len(rows) or len(sample_by_key) != len(rows) or
            len(controls) != len(rows) or len(control_by_key) != len(rows) or
            any(key(r) not in control_by_key for r in rows)):
        raise ValueError("Controls CSV does not match this benchmark")
    for row in samples:
        control = control_by_key[key(row)]
        if (row["experiment"] != "preempt-hold" or
                any(control[k] != row[k] for k in ("valid", "status", "t_invalidate_ns"))):
            raise ValueError("Controls CSV validity/timestamps do not match samples")
    db = sqlite3.connect(pathlib.Path(args.sqlite).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    pid = int(meta["process_pid"])
    gpu_ids = {r["device_id"] for r in rows}
    if len(gpu_ids) != 1:
        raise ValueError("Expected one GPU")
    gpu = next(iter(gpu_ids))
    names = {r["id"]: r["value"] for r in db.execute("SELECT * FROM StringIds")}
    enums = {r["id"]: r["name"] for r in db.execute("SELECT * FROM ENUM_GPU_CTX_SWITCH")}
    switch = [dict(r) for r in db.execute("SELECT * FROM GPU_CONTEXT_SWITCH_EVENTS ORDER BY timestamp")
              if r["globalPid"] is not None and ((r["globalPid"] >> 24) & 0xFFFFFF) == pid and r["gpuId"] == gpu]
    if not switch:
        raise ValueError("No GPU context-switch events for this process/GPU")
    nvtx = []
    for event in db.execute("SELECT * FROM NVTX_EVENTS ORDER BY start"):
        if event["globalTid"] is None or ((event["globalTid"] >> 24) & 0xFFFFFF) != pid:
            continue
        name = event["text"] or names.get(event["textId"], "")
        if name.startswith("mc."):
            nvtx.append({"name": name, "start": event["start"], "end": event["end"]})
    kernels = [dict(r) for r in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
               if ((r["globalPid"] >> 24) & 0xFFFFFF) == pid and r["deviceId"] == gpu]
    trace_end = max([k["end"] for k in kernels] + [r["end"] or r["start"] for r in nvtx])

    # Pair the exported enum's native RESTORE_START/SAVE_END endpoints. A
    # leading unmatched SAVE may belong to initialization before collection;
    # a repeated RESTORE without SAVE during collection is ambiguous, so stop.
    starts, intervals = {}, []
    for event in switch:
        context = event["contextId"]
        tag = enums[event["tag"]]
        if tag == "RESTORE_START":
            if context in starts:
                raise ValueError("Repeated RESTORE without SAVE: incomplete/ambiguous trace")
            starts[context] = event["timestamp"]
        elif tag == "SAVE_END" and context in starts:
            intervals.append((context, starts.pop(context), event["timestamp"]))
    intervals.extend((context, start, trace_end) for context, start in starts.items())

    cuda_ids = {role: {r[f"{role}_context_id"] for r in rows} for role in ("a", "b")}
    if any(len(ids) != 1 for ids in cuda_ids.values()):
        raise ValueError("A/B CUDA contexts were not persistent")
    cuda_ids = {role: next(iter(ids)) for role, ids in cuda_ids.items()}
    a_launches = []
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in tables:
            a_launches += [dict(r) for r in db.execute(f"SELECT * FROM {table}")
                           if names[r["nameId"]] == "cuLaunchKernel" and ((r["globalTid"] >> 24) & 0xFFFFFF) == pid
                           and (r["globalTid"] & 0xFFFFFF) == int(meta["thread_a_tid"])]
    a_launches.sort(key=lambda r: (r["start"], r["correlationId"]))

    def associate(role):
        ranges = [r for r in nvtx if r["end"] is not None and
                  (r["name"].startswith("mc.reference/") if role == "a" else r["name"] == "mc.map.B")]
        candidates, evidence = set(), []
        for region in ranges:
            for kernel in kernels:
                if kernel["contextId"] != cuda_ids[role] or not region["start"] <= kernel["start"] < kernel["end"] <= region["end"]:
                    continue
                expected = "compute_kernel" if role == "a" else "tiny_kernel"
                if names[kernel["shortName"]] != expected:
                    continue
                contained = {context for context, begin, end in intervals
                             if begin <= kernel["start"] < kernel["end"] <= end}
                if len(contained) != 1:
                    raise ValueError(f"Cannot uniquely associate isolated {role} kernel with switch interval")
                context = next(iter(contained))
                candidates.add(context)
                evidence.append({"range": region["name"], "kernel_start": kernel["start"], "kernel_end": kernel["end"], "switch_context": context})
        if len(candidates) != 1 or not evidence:
            raise ValueError(f"Missing/inconsistent {role} context-switch association")
        return next(iter(candidates)), evidence

    a_id, a_evidence = associate("a")
    b_id, b_evidence = associate("b")
    if a_id == b_id:
        raise ValueError("A/B unexpectedly associated with one switch context")

    def region(name, required=True):
        matches = [r for r in nvtx if r["name"] == name]
        if not matches and not required:
            return None
        if len(matches) != 1:
            raise ValueError("Missing/duplicate NVTX range " + name)
        return matches[0]

    def residency(begin, end):
        return sum(max(0, min(end, finish) - max(begin, start))
                   for context, start, finish in intervals if context == a_id)

    output = []
    for row in rows:
        c = control_by_key[key(row)]
        index = c["case_index"]
        invalidate = region("mc.invalidate/" + index)
        preempt = region("mc.preempt/" + index, False)
        kind = "schedule_" if row["condition"] == "schedule-hold" else "fifo_"
        disable = region("mc." + kind + "disable/" + index, False)
        enable = region("mc." + kind + "enable/" + index, False)
        if bool(disable) != bool(enable):
            raise ValueError("Incomplete disable/enable pair")
        hold_begin, hold_end = (disable["end"], enable["start"]) if disable else (0, 0)
        if disable and not ((not preempt or preempt["end"] <= disable["start"]) and disable["start"] < hold_begin < hold_end < enable["end"]):
            raise ValueError("Unexpected control sequence")
        restore_in_hold = [e for e in switch if e["contextId"] == a_id and
                           enums[e["tag"]] == "RESTORE_START" and hold_begin < e["timestamp"] < hold_end]
        reuse_seq = int(c["a_reuse_launch_seq"])
        if not 1 <= reuse_seq <= len(a_launches):
            raise ValueError("Missing A reuse launch")
        reuse_api = a_launches[reuse_seq - 1]
        reuse = [k for k in kernels if k["correlationId"] == reuse_api["correlationId"]]
        if len(reuse) != 1 or reuse[0]["contextId"] != cuda_ids["a"] or names[reuse[0]["shortName"]] != "tiny_kernel":
            raise ValueError("Reuse is not a tiny kernel on the original A CUDA context")
        if reuse[0]["start"] <= row["profile_a_end_ns"]:
            raise ValueError("A reuse precedes old-work completion")
        row.update({
            "case_index": index,
            "a_tsg_id": meta["a_hardware_tsg_id"],
            "a_switch_context_id": a_id,
            "b_switch_context_id": b_id,
            "switch_association": "inferred_from_isolated_kernels",
            "profile_invalidate_mark_ns": invalidate["start"],
            "profile_preempt_range_begin_ns": preempt["start"] if preempt else 0,
            "profile_preempt_range_end_ns": preempt["end"] if preempt else 0,
            "profile_hold_begin_ns": hold_begin,
            "profile_hold_end_ns": hold_end,
            "profile_hold_duration_ms": (hold_end - hold_begin) / 1e6 if disable else math.nan,
            "a_restore_events_during_hold": len(restore_in_hold) if disable else -1,
            "a_context_residency_during_hold_ms": residency(hold_begin, hold_end) / 1e6 if disable else math.nan,
            "a_kernel_residency_envelope_during_hold_ms": residency(max(hold_begin, row["profile_a_start_ns"]), min(hold_end, row["profile_a_end_ns"])) / 1e6 if disable else math.nan,
            "a_kernel_incomplete_until_reenable": int(row["profile_a_end_ns"] > enable["start"]) if disable else -1,
            "a_context_residency_during_b_us": residency(row["profile_b_start_ns"], row["profile_b_end_ns"]) / 1000,
            "a_context_residency_after_preempt_ms": residency(preempt["end"], row["profile_a_end_ns"]) / 1e6 if preempt else math.nan,
            "a_context_residency_preempt_to_disable_return_us": residency(preempt["end"], hold_begin) / 1000 if disable and preempt else math.nan,
            "a_context_residency_after_reenable_ms": residency(enable["start"], row["profile_a_end_ns"]) / 1e6 if enable else math.nan,
            "a_pending_before_rearm": c["a_pending_before_rearm"],
            "a_output_matches_reference": c["a_output_matches_reference"],
            "a_reuse_ok": c["a_reuse_ok"],
            "profile_a_reuse_start_ns": reuse[0]["start"],
            "profile_a_reuse_end_ns": reuse[0]["end"],
            "profile_a_reuse_same_cuda_context": 1,
            "trace_drop_count": "unknown",
        })
        output.append(row)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(output[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(output)
    if args.evidence:
        evidence = {"pid": pid, "gpu_id": gpu, "a_cuda_context": cuda_ids["a"], "b_cuda_context": cuda_ids["b"],
                    "a_association": a_evidence, "b_association": b_evidence,
                    "switch_events": [{**e, "tag_name": enums[e["tag"]],
                                       "role": "A" if e["contextId"] == a_id else "B" if e["contextId"] == b_id else "initialization_or_unmapped"}
                                      for e in switch],
                    "nvtx": nvtx, "a_residency_intervals": [r for r in intervals if r[0] == a_id],
                    "interpretation": "inferred mapping; RESTORE_START to SAVE_END residency envelope, not SM active cycles; drop count unknown"}
        path = pathlib.Path(args.evidence)
        with path.open("x") as handle:
            json.dump(evidence, handle, indent=2)
            handle.write("\n")
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite")
    parser.add_argument("csv")
    parser.add_argument("--evidence", help="Create JSON containing only this experiment's process/GPU events")
    args = parser.parse_args()
    try:
        main(args)
    except (ValueError, KeyError, sqlite3.Error, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
