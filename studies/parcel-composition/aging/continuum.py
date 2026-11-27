"""Check the largest activation error anywhere between instrument settings."""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from analyze import HERE, LIMITS
from compression import tree
from physics import EDGES, KAPPA, KELVIN, SS, evolve
from run import SEEDS


def critical(vol):
    """Critical supersaturation (%) for the wettable study particles."""
    return 100 * np.expm1(np.sqrt(4 * KELVIN**3 / (27 * (vol @ KAPPA))))


def all_caps(vol, cap, protect):
    """Check both sides of each CDF jump and both interval endpoints."""
    diam = np.cbrt(vol.sum(axis=1))
    groups = np.searchsorted(EDGES[1:-1], diam)
    feat = np.column_stack(
        [
            np.log(diam) / np.log(EDGES[1] / EDGES[0]),
            vol / vol.sum(axis=1, keepdims=True),
        ]
    )
    exact = np.sort(critical(vol))
    thresholds, counts, births, deaths = [], [], [], []
    for b in np.unique(groups):
        take = groups == b
        w, m, splits = tree(vol[take], feat[take], cap, protect)
        born = np.ones(len(w), dtype=int)
        dead = np.full(len(w), cap + 1)
        for k, (parent, left, right) in enumerate(splits, start=2):
            born[[left, right]] = k
            dead[parent] = k
        thresholds.extend(critical(m))
        counts.extend(w)
        births.extend(born)
        deaths.extend(dead)
    # Include interval endpoints as zero-weight representatives at every cap.
    threshold = np.r_[thresholds, SS[[0, -1]]]
    order = np.argsort(threshold, kind="stable")
    threshold = threshold[order]
    weights = np.r_[counts, 0, 0][order]
    start = np.r_[births, 1, 1][order]
    end = np.r_[deaths, cap + 1, cap + 1][order]
    observed_left = np.searchsorted(exact, threshold, side="left")
    observed_right = np.searchsorted(exact, threshold, side="right")
    errors = []
    for k in range(1, cap + 1):
        live = (start <= k) & (k < end)
        x = threshold[live]
        cumulative = np.cumsum(weights[live])
        # Group identical thresholds so simultaneous jumps stay simultaneous.
        first = np.r_[True, np.diff(x) > 0]
        last = np.r_[np.diff(x) > 0, True]
        lower = first & (x > SS[0]) & (x <= SS[-1])
        upper = last & (x >= SS[0]) & (x <= SS[-1])
        left = (cumulative - weights[live] - observed_left[live])[lower]
        right = (cumulative - observed_right[live])[upper]
        errors.append(100 * max(abs(left).max(), abs(right).max()) / len(vol))
    return errors


def history(job):
    """Return a complete paired history for a compression rule."""
    seed, cap, protect = job
    result = []
    for hour, state, *_ in evolve(seed, 100000):
        for parcel, vol in zip("AB", state, strict=True):
            for k, error in enumerate(all_caps(vol, cap, protect), start=1):
                result.append((seed, hour, parcel, k, error))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic", action="store_true")
    parser.add_argument("--limit", type=int, default=96)
    args = parser.parse_args()
    jobs = [(s, args.limit, not args.generic) for s in SEEDS]
    with ProcessPoolExecutor(3) as pool:
        records = [row for group in pool.map(history, jobs) for row in group]
    table = pd.DataFrame(
        records, columns=["seed", "hour", "parcel", "cap", "continuous_max_pp"]
    )
    table["split"] = np.where(
        table.seed.isin(SEEDS[:8]), "calibration", "held-out"
    )
    maxima = (
        table.groupby(["split", "cap"]).continuous_max_pp.max().reset_index()
    )
    method = "generic" if args.generic else "carrier"
    maxima.to_csv(
        HERE / f"continuum_{method}.csv", index=False, float_format="%.8g"
    )
    print(
        method,
        maxima[maxima.continuous_max_pp <= LIMITS[1]].groupby("split").first(),
    )
