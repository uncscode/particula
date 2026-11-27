"""Compare water summaries at equal representative count in cloudy restarts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import read
from metrics import compress, make_tree, prepare
from model import KAPPA, SEEDS, init, step

TREATMENTS = (
    "bin mean",
    "dry representatives",
    "two class means",
    "class representatives",
)


def split_water(box, dry_bin, prep, pool):
    """Split the droplet-growth branches before grouping dry composition."""
    state, features, flags, ends, _ = prep
    # Same existing-droplet growth branch as model.water_step.
    growing = box.water > 1.05 * 49 * (box.dry @ KAPPA) + 100
    groups = 2 * dry_bin + growing
    counts = np.bincount(groups, minlength=2 * (len(ends) - 1))
    offsets = np.r_[0, counts.cumsum()].astype(np.int32)
    ordered = np.argsort(groups, kind="stable").astype(np.int32)
    means, meta, order = make_tree(
        (state, features, flags, offsets, ordered), 16, 2
    )
    pairs = counts.reshape(-1, 2)
    total = pairs.sum(1)
    budget = np.minimum(total, 16)
    # Round proportional allocation, keeping each occupied class represented.
    wet_rows = np.rint(budget * pairs[:, 1] / np.maximum(total, 1)).astype(int)
    wet_rows = np.clip(
        wet_rows,
        np.maximum(pairs[:, 1] > 0, budget - pairs[:, 0]),
        np.minimum(pairs[:, 1], budget - (pairs[:, 0] > 0)),
    )
    caps = np.column_stack((budget - wet_rows, wet_rows)).ravel()
    limit = caps[meta[:, 0]]
    keep = (meta[:, 1] <= limit) & (meta[:, 2] > limit)
    group_water = np.bincount(
        groups, weights=box.water, minlength=len(counts)
    ) / np.maximum(counts, 1)
    for center, (group, _, _, start, n) in zip(
        means[keep], meta[keep], strict=True
    ):
        idx = order[start : start + n]
        box.dry[idx] = center[:8]
        box.water[idx] = group_water[group] if pool else center[8]
    np.testing.assert_equal(keep.sum(), budget.sum())
    return int(keep.sum())


def reduce_box(box, treatment):
    """Check dry species and water inventories in every original dry bin."""
    prep = prepare(box, 10)
    _, _, _, ends, order = prep
    bins = np.empty(len(order), int)
    bins[order] = np.repeat(np.arange(len(ends) - 1), np.diff(ends))

    def inventories():
        return np.stack(
            [
                np.bincount(bins, weights=v, minlength=len(ends) - 1)
                for v in (*box.dry.T, box.water)
            ]
        )

    initial = inventories()
    if treatment in ("bin mean", "dry representatives"):
        rows = compress(box, 10, 16, 2)
        if treatment == "bin mean":
            box.water = (initial[-1] / np.maximum(np.diff(ends), 1))[bins]
    else:
        rows = split_water(box, bins, prep, treatment == "two class means")
    np.testing.assert_allclose(inventories(), initial, rtol=1e-10)
    return rows


def run(job):
    """Compare the remaining cloud peaks with the paired full trajectory."""
    base, seed, reg, treatments = job
    folder = base / str(seed) / reg
    n, dt, _ = map(int, (folder / "done").read_text().split())
    _, _, params = init(seed, n)
    ref = pd.read_csv(folder / "trajectory.csv").query("seconds>10800")
    expected = ref[["drops_cm3_A", "water_g_m3_A"]].max().to_numpy()
    rows = []
    for treatment in treatments:
        boxes, rng, t = read(folder / "10800.npz")
        count = sum(reduce_box(box, treatment) for box in boxes)
        peak = np.zeros(2)
        for s in range(t, 21600, dt):
            step(boxes, rng, params, reg, s, dt)
            peak = np.maximum(peak, [boxes[0].droplets, boxes[0].liquid])
        rows.append([seed, reg, dt, treatment, count, *expected, *peak])
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS[8:])
    parser.add_argument(
        "--treatments", choices=TREATMENTS, nargs="+", default=TREATMENTS
    )
    args = parser.parse_args()
    jobs = [
        (args.base, s, r, args.treatments)
        for s in args.seeds
        for r in ("c1b1n0", "c1b1n1")
    ]
    with ProcessPoolExecutor(min(8, len(jobs))) as pool:
        results = [r for block in pool.map(run, jobs) for r in block]
    pd.DataFrame(
        results,
        columns=[
            "seed",
            "regime",
            "dt_s",
            "treatment",
            "rows_both_parcels",
            "ref_drops_cm3",
            "ref_lwc_g_m3",
            "drops_cm3",
            "lwc_g_m3",
        ],
    ).to_csv(args.output, index=False, float_format="%.10g")
