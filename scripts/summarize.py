#!/usr/bin/env python3
"""Summarize observed latencies; invalid trials never contribute to the minimum."""
import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict


def p95(values):
    values = sorted(values)
    return values[math.ceil(0.95 * len(values)) - 1]


def summarize(paths, output):
    writer = csv.writer(output)
    writer.writerow(["file", "target_ms", "condition", "metric", "n", "invalid_trials",
                     "min", "median", "p95"])
    for path in paths:
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        groups = defaultdict(list)
        pairs = defaultdict(dict)
        for row in rows:
            if row["experiment"] == "calibration":
                continue
            groups[(row["target_ms"], row["condition"])].append(row)
            if row["experiment"] == "standby" and row["valid"] == "1":
                pairs[(row["target_ms"], row["trial"])][row["condition"]] = float(row["a_event_ms"])
        for (target, condition), trials in groups.items():
            valid = [r for r in trials if r["valid"] == "1"]
            invalid = len(trials) - len(valid)
            metrics = (["invalidate_to_b_launch_us", "invalidate_to_b_gpu_start_est_us", "b_takeover_us",
                        "a_event_pending_after_b_complete", "a_event_ms", "destroy_latency_us"]
                       if trials[0]["experiment"] == "live-handoff"
                       else ["a_event_ms", "a_effective_gflops"] if trials[0]["experiment"] == "standby"
                       else ["b_request_us"] if condition == "b_with_idle_a"
                       else ["destroy_latency_us", "b_takeover_us", "handoff_us"])
            for metric in metrics:
                values = [float(r[metric]) for r in valid if math.isfinite(float(r[metric]))
                          and not (metric == "a_event_pending_after_b_complete" and float(r[metric]) < 0)]
                stats = [min(values), statistics.median(values), p95(values)] if values else ["NA"] * 3
                writer.writerow([path, target, condition, metric, len(values), invalid, *stats])
        ratios = defaultdict(list)
        for (target, _), pair in pairs.items():
            if "a_only" in pair and "a_with_idle_b" in pair:
                ratios[target].append(100 * (pair["a_with_idle_b"] / pair["a_only"] - 1))
        for target, values in ratios.items():
            writer.writerow([path, target, "paired_standby", "a_time_overhead_percent", len(values), 0,
                             min(values), statistics.median(values), p95(values)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", nargs="+")
    args = parser.parse_args()
    summarize(args.csv_files, sys.stdout)
