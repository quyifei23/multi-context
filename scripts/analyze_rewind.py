#!/usr/bin/env python3
"""Validate rewind evidence and export only relative, de-identified observations."""
import argparse
import csv
import json
import pathlib
import struct
import sys

CLASSES = {"old_queue_preserved", "queued_sentinel_removed_context_reusable",
           "context_or_submission_poisoned", "ambiguous"}


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def analyze(path):
    path = pathlib.Path(path)
    samples = read(path)
    epochs = read(str(path) + ".epochs.csv")
    calls = read(str(path) + ".epoch_calls.csv")
    meta = {r["key"]: r["value"] for r in read(path.with_suffix(".metadata.csv"))}
    directory = pathlib.Path(str(path) + ".rm")
    identity = json.loads((directory / "bound.identity.json").read_text())
    events = [json.loads(line) for line in (directory / "A_control_events.jsonl").read_text().splitlines()]
    by_seq = {str(e["operation_seq"]): e for e in events if e["attempted"]}
    if any(e["command"] == 0xa06c0105 for e in events):
        raise ValueError("Unexpected Group PREEMPT in the rewind experiment")
    if len(samples) != len(epochs) or not epochs:
        raise ValueError("Missing or mismatched epoch result; inspect the last API begin and process exit")
    members = [m["channel"]["handle"] for m in identity["members"]]
    token_set, normal_payloads, output = set(), set(), []
    for s, r in zip(samples, epochs):
        if (r["classification"] not in CLASSES or s["status"] != r["classification"] or
                any(s[k] != r[k] for k in ("trial", "order", "iterations", "t_invalidate_ns"))):
            raise ValueError("Main/epoch CSV mismatch")
        expected = (int(r["old_expected"]), int(r["new_expected"]))
        if expected[0] == expected[1] or any(t in token_set or not t for t in expected):
            raise ValueError("Old/new epoch tokens are not unique")
        token_set.update(expected)
        selected = [c for c in calls if c["case_index"] == r["case_index"]]
        errors = sorted({int(c["cuda_result"]) for c in selected if c["edge"] == "end" and int(c["cuda_result"]) not in (0, 600)})
        observed = [c for c in selected if int(c["old_token"]) == expected[0]]
        for kind in ("disable", "enable"):
            if r[kind + "_attempted"] != "1":
                continue
            e = by_seq[r[kind + "_operation_seq"]]
            if (e["pid"] != int(meta["process_pid"]) or e["pid"] != identity["owner_pid"] or
                    e["command"] != 0x2080110b or e["target"]["hObject"] != identity["subdevice"]["handle"]):
                raise ValueError("Wrong control target/process")
            payload = bytes.fromhex(e["params_hex"])
            if (len(payload) != 536 or payload[0] != (kind == "disable") or payload[8] != 0 or
                    payload[9] != (int(r["rewind"]) if kind == "disable" else 0) or
                    struct.unpack_from("<Q", payload, 16)[0] != 0 or
                    struct.unpack_from("<I", payload, 4)[0] != len(members) or
                    list(struct.unpack_from(f"<{len(members)}I", payload, 24)) != [identity["hClient"]] * len(members) or
                    list(struct.unpack_from(f"<{len(members)}I", payload, 280)) != members):
                raise ValueError("FIFO flags or full channel membership mismatch")
            if int(r[kind + "_ioctl_begin_ns"]) != e["call_begin_ns"] or int(r[kind + "_ioctl_end_ns"]) != e["call_end_ns"]:
                raise ValueError("Control timestamp/journal mismatch")
            if kind == "disable":
                normal_payloads.add(payload[:9] + b"\x00" + payload[10:])
        preserved = r["classification"] == "old_queue_preserved"
        if preserved:
            required = ("old_value_ok", "new_value_ok", "hold_all_pending", "context_same", "buffer_same")
            if (errors or any(r[k] != "1" for k in required) or
                    any(r[k] != "0" for k in ("a_event_result", "old_event_result", "new_event_result", "disable_rm_status", "enable_rm_status")) or
                    r["old_observed"] != r["old_expected"] or r["new_observed"] != r["new_expected"] or
                    r["old_stream"] == r["new_stream"]):
                raise ValueError("Preserved classification lacks matching execution/identity evidence")
        delta = lambda end, begin: (int(r[end]) - int(r[begin])) / 1000 if int(r[end]) and int(r[begin]) else float("nan")
        old_seen = min((int(c["host_ns"]) for c in observed), default=0)
        row = {"run": path.stem, **{k: r[k] for k in (
            "trial", "order", "rewind", "classification", "iterations", "reference_a_ms", "hold_ms", "query_timeout_ms",
            "old_value_ok", "new_value_ok", "a_event_result", "old_event_result", "new_event_result", "hold_observations",
            "hold_all_pending", "context_same", "buffer_same", "a_event_ms", "disable_attempted", "disable_api_result", "disable_rm_status",
            "enable_attempted", "enable_api_result", "enable_rm_status")}}
        row.update({"cuda_async_error_codes": ";".join(map(str, errors)),
                    "invalidate_to_disable_return_us": delta("disable_end_ns", "t_invalidate_ns"),
                    "disable_call_us": delta("disable_end_ns", "disable_begin_ns"),
                    "disable_ioctl_us": delta("disable_ioctl_end_ns", "disable_ioctl_begin_ns"),
                    "observed_hold_ms": delta("enable_begin_ns", "disable_end_ns") / 1000,
                    "enable_to_old_token_observed_us": (old_seen - int(r["enable_end_ns"])) / 1000 if old_seen else float("nan"),
                    "old_token_seen_before_new_launch": int(0 < old_seen < int(r["t_new_launch_begin_ns"])),
                    "observation_after_enable_ms": delta("t_observation_end_ns", "enable_end_ns") / 1000})
        output.append(row)
    if len(normal_payloads) > 1:
        raise ValueError("Within-run FIFO payload changed in more than the rewind byte")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+")
    args = parser.parse_args()
    try:
        rows = [r for p in args.csv for r in analyze(p)]
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    except (ValueError, KeyError, OSError, struct.error) as error:
        parser.exit(1, f"Error: {error}\n")
