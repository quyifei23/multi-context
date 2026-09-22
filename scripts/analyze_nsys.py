#!/usr/bin/env python3
"""Join a live-handoff CSV to its own Nsight SQLite trace; no extra dependencies."""
import argparse
import csv
import math
import pathlib
import sqlite3
import sys


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def analyze(database, samples, metadata):
    meta = {r["key"]: r["value"] for r in read_csv(metadata)}
    if meta.get("run_status") != "completed":
        raise ValueError("The profiled benchmark did not complete successfully")
    pid = int(meta["process_pid"])
    db = sqlite3.connect(pathlib.Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        raise ValueError("No CUDA kernel activity table in this Nsight export")
    names = {r["id"]: r["value"] for r in db.execute("SELECT id,value FROM StringIds")}
    launches = {role: [] for role in ("a", "b")}
    tids = {role: int(meta[f"thread_{role}_tid"]) for role in launches}
    # Nsight 2024.6 stores Driver calls in the table named RUNTIME. Accept a
    # separate DRIVER table too; API function names, not table names, identify
    # the actual API layer. globalTid encoding follows NVIDIA's report scripts.
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table not in tables:
            continue
        for api in db.execute(f"SELECT * FROM {table}"):
            if names[api["nameId"]] != "cuLaunchKernel" or (api["globalTid"] >> 24) & 0xFFFFFF != pid:
                continue
            for role, tid in tids.items():
                if api["globalTid"] & 0xFFFFFF == tid:
                    launches[role].append(dict(api))
    for role, calls in launches.items():
        calls.sort(key=lambda r: (r["start"], r["correlationId"]))
        if not calls:
            raise ValueError(f"No launches for {role}: trace/CSV process or thread does not match")
        expected = meta.get(f"thread_{role}_launch_count")
        if expected is not None and len(calls) != int(expected):
            raise ValueError(f"Launch records for {role}: expected {expected}, found {len(calls)}")
        if len({r["correlationId"] for r in calls}) != len(calls):
            raise ValueError("Duplicate launch correlation IDs; unsupported/ambiguous trace")

    def kernel(role, sequence):
        if not 1 <= sequence <= len(launches[role]):
            raise ValueError(f"Missing launch {role}:{sequence}")
        api = launches[role][sequence - 1]
        records = db.execute(
            "SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=? AND globalPid=?",
            (api["correlationId"], api["globalTid"] & 0xFFFFFFFFFF000000),
        ).fetchall()
        if len(records) != 1:
            raise ValueError(f"Expected one kernel for {role}:{sequence}, found {len(records)}")
        k = records[0]
        expected = "compute_kernel" if role == "a" else "tiny_kernel"
        if names[k["shortName"]] != expected or not 0 < k["start"] < k["end"]:
            raise ValueError(f"Kernel name/timestamps mismatch for {role}:{sequence}")
        return api, k

    output = []
    for s in read_csv(samples):
        if s["experiment"] != "live-handoff":
            continue
        b_api, b = kernel("b", int(s["b_launch_seq"]))
        a = kernel("a", int(s["a_launch_seq"]))[1] if int(s["a_launch_seq"]) else None
        if a and (a["deviceId"] != b["deviceId"] or a["contextId"] == b["contextId"]):
            raise ValueError("Expected different contexts on the same device")
        # Nsight's API entry/exit lie inside the benchmark's host call bracket.
        # Translate a nearby GPU timestamp via that API bracket. This assumes
        # locally equal clock rates; these are conditional bounds, not a claim
        # of nanosecond accuracy. GPU A/B ordering needs no clock translation.
        invalidate = int(s["t_invalidate_ns"])
        lower = b["start"] - b_api["start"] + int(s["t_B_launch_ns"]) - invalidate
        upper = b["start"] - b_api["end"] + int(s["t_B_launch_return_ns"]) - invalidate
        alignment_valid = 0 <= lower <= upper <= int(s["t_B_complete_ns"]) - invalidate
        row = {k: s[k] for k in ("target_ms", "condition", "trial", "order", "valid", "a_launch_seq", "b_launch_seq")}
        row.update({
            "profiled": 1,
            "device_id": b["deviceId"],
            "a_context_id": a["contextId"] if a else -1,
            "b_context_id": b["contextId"],
            "profile_a_start_ns": a["start"] if a else 0,
            "profile_a_end_ns": a["end"] if a else 0,
            "profile_b_start_ns": b["start"],
            "profile_b_end_ns": b["end"],
            "a_unfinished_at_b_start": int(a["start"] <= b["start"] < a["end"]) if a else -1,
            "a_unfinished_at_b_end": int(a["start"] <= b["end"] < a["end"]) if a else -1,
            "b_finishes_before_a_by_us": (a["end"] - b["end"]) / 1000 if a else math.nan,
            "profile_clock_alignment_valid": int(alignment_valid),
            "profile_invalidate_to_b_gpu_start_est_us": (lower + upper) / 2000 if alignment_valid else math.nan,
            "profile_invalidate_to_b_gpu_start_lower_us": lower / 1000 if alignment_valid else math.nan,
            "profile_invalidate_to_b_gpu_start_upper_us": upper / 1000 if alignment_valid else math.nan,
            "profile_api_alignment_width_ns": upper - lower,
            "host_invalidate_to_b_launch_us": s["invalidate_to_b_launch_us"],
            "host_invalidate_to_b_complete_us": s["b_takeover_us"],
        })
        output.append(row)
    db.close()
    if not output:
        raise ValueError("No live-handoff experiment rows in the CSV")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", help="Nsight --export=sqlite output")
    parser.add_argument("csv", help="CSV from this exact profiled process")
    parser.add_argument("--metadata", help="Default: CSV path with .metadata.csv suffix")
    args = parser.parse_args()
    try:
        records = analyze(args.sqlite, args.csv, args.metadata or pathlib.Path(args.csv).with_suffix(".metadata.csv"))
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    except (ValueError, KeyError, sqlite3.Error, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
