"""Measure portable archive sizes and Linux working memory at five events."""

import argparse
import ctypes
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from campaign import read
from model import Box
from redraw_audit import labels
from redraw_checkpoint import PARTICLE, draw_box, fit_box, load, serialize
from redraw_design import WINDOWS
from redraw_scaling import RECIPES, scale_cases
from stage_two_design import initialize


def rss():
    """Return resident and peak resident bytes from Linux procfs."""
    status = Path("/proc/self/status").read_text().splitlines()
    return {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in status
        if line.startswith(("VmRSS:", "VmHWM:"))
    }


def measured(action):
    """Reset peak RSS with inputs resident and time the operation."""
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    Path("/proc/self/clear_refs").write_text("5\n")
    baseline = rss()["VmRSS"]
    begin = perf_counter()
    result = action()
    elapsed = perf_counter() - begin
    peak = rss()["VmHWM"]
    return result, (baseline, peak, elapsed)


def budgets(box):
    """Group integer counts and nine extensive coordinates by physical class."""
    keys, inv, count = np.unique(
        labels(box, 10), axis=0, return_inverse=True, return_counts=True
    )
    inventory = [np.bincount(inv, weights=v) for v in (*box.dry.T, box.water)]
    return keys, count, np.array(inventory)


def record(root, case, forcing, start, spec):
    """Measure both directions and verify recovered physical-class budgets."""
    boxes, rng, time = read(root / case.key / f"{start}.npz")
    np.testing.assert_equal(time, start)
    reference = [budgets(b) for b in boxes]
    slots = sum(len(b.dry) for b in boxes)
    raw = sum(getattr(b, k).nbytes for b in boxes for k in PARTICLE)

    def encode():
        if spec:
            fields = [fit_box(b, *spec) for b in boxes]
            particle_keys = ("mean", "count", "label", "delta")
            rows = sum(len(b["count"]) for b in fields)
        else:
            fields = [vars(b) for b in boxes]
            particle_keys = PARTICLE
            rows = slots
        data_bytes = sum(b[k].nbytes for b in fields for k in particle_keys)
        payload = serialize(fields, case, forcing, start, rng, spec)
        return payload, rows, data_bytes

    (payload, rows, data_bytes), enc = measured(encode)
    boxes.clear()  # The decoder's input is the archive alone.
    sample_rng = np.random.default_rng([case.seed, case.n, start, 197])

    def decode():
        fields, header = load(payload)
        if spec:
            restored = [draw_box(f, sample_rng) for f in fields]
        else:
            restored = [Box(**f) for f in fields]
        return restored, header

    (restored, meta), dec = measured(decode)
    np.testing.assert_equal(
        (meta["seconds"], meta["case"]["n"]), (start, case.n)
    )
    for b, orig in zip(restored, reference, strict=True):
        keys, counts, sums = budgets(b)
        np.testing.assert_array_equal(keys, orig[0])
        np.testing.assert_array_equal(counts, orig[1])
        np.testing.assert_allclose(sums, orig[2], rtol=1e-10, atol=1e-12)
    row = dict(
        seed=case.seed,
        n=case.n,
        start=start,
        recipe=spec[1] if spec else 0,
        slots=slots,
        rows=rows,
        resolved_array_bytes=raw,
        stored_array_bytes=data_bytes,
        archive_bytes=len(payload),
    )
    for stage, (base, peak, elapsed) in (("encode", enc), ("decode", dec)):
        row[stage + "_base_rss_bytes"] = base
        row[stage + "_peak_rss_bytes"] = peak
        row[stage + "_seconds"] = elapsed
    return row


def main():
    """Run the command-line study stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for case in scale_cases():
        initial, rng, forcing = initialize(case)
        del initial, rng
        for start in WINDOWS:
            for spec in (None, *RECIPES):
                rows.append(record(args.root, case, forcing, start, spec))
        print(f"Measured {case.n:,} starting particles per parcel", flush=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.8g")


if __name__ == "__main__":
    main()
