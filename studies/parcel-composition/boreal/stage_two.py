"""Test cohort-conditioned sizes in the coupled boreal challenge."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import audit, inventory, read
from cloud_classes import reduce_box
from metrics import make_tree, prepare, profile
from model import KAPPA, step
from restart import activation, relative
from scipy.stats import wasserstein_distance
from shape import basis, criteria, fit, positions
from stage_two_design import REGIME, TIMES, design, initialize

from optics import Optics

METHODS = ("dry", "water", "cohort", "rank")
CUTS = np.array([2.0, 3.0, 5.0, 10.0])


def quotas(counts):
    """Allocate 16 rows within each bin, preserving all occupied classes."""
    result = (counts > 0).astype(int)
    for size, rows in zip(
        counts.reshape(-1, 4), result.reshape(-1, 4), strict=True
    ):
        for _ in range(min(16, size.sum()) - rows.sum()):
            priority = np.where(rows < size, size / (rows + 1), -1.0)
            rows[np.argmax(priority)] += 1
    return result


def rank_sections(idx, diameter, count):
    """Place boundaries near equal-number quantiles, keeping ties together."""
    idx = idx[np.argsort(diameter[idx], kind="stable")]
    # Arithmetic on equal birth batches can differ around machine precision.
    boundaries = 1 + np.flatnonzero(np.diff(np.round(diameter[idx], 8)))
    if len(boundaries) < count - 1:
        # Once all distinct sizes are isolated, extra rows may divide a tie.
        extra = np.setdiff1d(np.arange(1, len(idx)), boundaries)
        required = count - 1 - len(boundaries)
        if required:
            pick = ((np.arange(required) + 0.5) * len(extra) / required).astype(
                int
            )
            boundaries = np.sort(np.r_[boundaries, extra[pick]])
    else:
        picks = []
        first = 0
        for k in range(1, count):
            available = boundaries[first : len(boundaries) - (count - 1 - k)]
            nearest = np.argmin(abs(available - k * len(idx) / count))
            picks.append(available[nearest])
            first += nearest + 1
        boundaries = picks
    return np.split(idx, boundaries)


def reduce(b, method, refine=None):
    """Compress composition and water, retaining exact lineage."""
    if method in ("dry", "water"):
        return reduce_box(
            b,
            {"dry": "dry representatives", "water": "class representatives"}[
                method
            ],
        )
    prep = prepare(b, 10)
    state, features, flags, ends, order = prep
    bins = np.empty(len(order), int)
    bins[order] = np.repeat(np.arange(len(ends) - 1), np.diff(ends))
    growing = b.water > 1.05 * 49 * (b.dry @ KAPPA) + 100
    group = bins * 4 + growing * 2 + (~b.old)
    counts = np.bincount(group, minlength=4 * (len(ends) - 1))
    offsets = np.r_[0, counts.cumsum()].astype(np.int32)
    slots = np.argsort(group, kind="stable").astype(np.int32)
    caps = quotas(counts)
    _, meta, ordered = make_tree(
        (state, features, flags, offsets, slots), 16, 2
    )
    chosen = (meta[:, 1] <= caps[meta[:, 0]]) & (caps[meta[:, 0]] < meta[:, 2])
    blocks = [
        ordered[start : start + n]
        for g, _, _, start, n in meta[chosen]
        if method != "rank" or g % 2 == 0
    ]
    if method == "rank":
        diameter = np.cbrt(b.dry.sum(1))
        for g in np.flatnonzero(
            (counts > 0) & (np.arange(len(counts)) % 2 == 1)
        ):
            blocks.extend(
                rank_sections(
                    slots[offsets[g] : offsets[g + 1]], diameter, caps[g]
                )
            )
    np.testing.assert_equal(len(blocks), caps.sum())
    if refine is not None:
        blocks = refine(state, blocks)
    for idx in blocks:
        anchor = state[idx[0]]
        center = anchor + (state[idx] - anchor).mean(0)
        b.dry[idx] = center[:8]
        b.water[idx] = center[8]
    for before, after in zip(state.T, (*b.dry.T, b.water), strict=True):
        np.testing.assert_allclose(
            np.bincount(group, weights=before),
            np.bincount(group, weights=after),
            rtol=1e-10,
            atol=1e-13,
        )
    return len(blocks)


def cohort_sizes(b):
    """Dry diameter of particles containing exclusively new lineages, in nm."""
    return np.sort(np.cbrt(b.dry[~b.old].sum(1)))


def cdf(x, grid):
    """Evaluate a sorted sample’s number-weighted cumulative fraction."""
    return np.searchsorted(x, grid, "right") / len(x)


def ks(x, y):
    """Measure the largest difference between empirical CDFs in points."""
    grid = np.unique(np.r_[x, y])
    return 100 * np.max(abs(cdf(x, grid) - cdf(y, grid)))


def spectrum(x, y):
    """Compare size distributions of surviving new particles."""
    values = dict(ref_new=len(x), new=len(y))
    if len(x) and len(y):
        errors = 100 * (cdf(y, CUTS) - cdf(x, CUTS))
        values.update(
            ks_raw=ks(x, y),
            ks_pp=ks(np.round(x, 8), np.round(y, 8)),
            w1_nm=wasserstein_distance(x, y),
            cut_pp=np.max(abs(errors)),
            ref_median_nm=np.median(x),
            median_nm=np.median(y),
        )
        for d, e in zip(CUTS, errors, strict=True):
            values[f"cdf_{int(d)}_pp"] = e
    return values


def chi_data(b):
    """Return dry χ and counts with defined χ on the common grid."""
    chi, count, _ = profile(b, prepare(b, 16), np.zeros((len(b.old), 1)))
    y = chi[1:-1]
    return np.nan_to_num(y), np.where(np.isfinite(y), count[1:-1], 0)


@lru_cache(maxsize=2)
def optical(path):
    """Reuse the qualified optical table inside each worker."""
    return Optics(path)


def score(b, r, opt):
    """Compare each parcel with its own full-resolution reference."""
    x, y = cohort_sizes(r), cohort_sizes(b)
    out = spectrum(x, y)
    rmse, sup = activation(r.dry, b.dry)
    a = opt.areas(b.dry, b.water).sum(0)
    ref = opt.areas(r.dry, r.water).sum(0)
    out.update(
        act_rmse=rmse,
        act_max=sup,
        dry_abs=relative(a[0], ref[0]),
        wet_abs=relative(a[1], ref[1]),
    )
    yy, w = chi_data(r)
    model, v = chi_data(b)
    common = np.where(v > 0, w, 0)
    out["chi_coverage"] = common.sum() / max(1, w.sum())
    for key, p, ww in [
        ("rep", model, common),
        ("nine", fit(basis(positions((1, 6, 1))), yy, w), w),
        (
            "six",
            fit(basis(tuple(np.linspace(0.5 / 16, 3 - 0.5 / 16, 6))), yy, w),
            w,
        ),
    ]:
        for name, value in zip(
            ("all", "small", "middle", "large"),
            criteria(yy, p, ww),
            strict=True,
        ):
            out[f"chi_{key}_{name}"] = value
    return out


def run(job):
    """Score all snapshots and the two paired restarts of one history."""
    root, case, method, optpath = job
    folder = root / case.key
    dest = folder / f"stage2-{method}.csv"
    if dest.exists():
        return
    trace = pd.read_csv(folder / "trace.csv").set_index("seconds")
    opt = optical(optpath)
    _, _, params = initialize(case)
    result = []

    def record(boxes, full, start, end, rows, extra):
        for parcel, (b, r) in enumerate(zip(boxes, full, strict=True)):
            result.append(
                dict(
                    **asdict(case),
                    method=method,
                    start=start,
                    end=end,
                    parcel=parcel,
                    rows=rows[parcel],
                    **extra,
                    **score(b, r, opt),
                )
            )

    for t in TIMES:
        boxes, rng, _ = read(folder / f"{t}.npz")
        full, _, _ = read(folder / f"{t}.npz")
        initial = inventory(boxes)
        rows = [reduce(b, method) for b in boxes]
        np.testing.assert_allclose(inventory(boxes), initial, rtol=1e-10)
        record(boxes, full, t, t, rows, {})
        if t not in (3600, 10800):
            continue
        source = np.zeros(8)
        baseborn = sum(b.born for b in boxes)
        peaks = np.zeros(2)
        for s in range(t, 21600, case.dt):
            source += step(boxes, rng, params, REGIME, s, case.dt)
            peaks = np.maximum(peaks, [boxes[0].droplets, boxes[0].liquid])
            end = s + case.dt
            if end not in TIMES:
                continue
            audit(boxes, initial, case.n, source)
            full, _, _ = read(folder / f"{end}.npz")
            window = trace.loc[t + case.dt : end]
            expect = window[["drops_cm3", "water_g_m3"]].max().to_numpy()
            born = sum(b.born for b in boxes)
            born_ref = int(trace.loc[end, "born"])
            success = sum(b.reached for b in boxes) / born if born else np.nan
            success_ref = (
                trace.loc[end, "reached"] / born_ref if born_ref else np.nan
            )
            extra = dict(
                ref_drops=expect[0],
                drops=peaks[0],
                ref_lwc=expect[1],
                lwc=peaks[1],
                ref_births=born_ref - baseborn,
                births=born - baseborn,
                ref_survival=100 * success_ref,
                survival=100 * success,
                survival_pp=100 * abs(success - success_ref),
            )
            record(boxes, full, t, end, rows, extra)
    tmp = dest.with_suffix(".tmp")
    pd.DataFrame(result).to_csv(tmp, index=False, float_format="%.10g")
    tmp.replace(dest)
    print(case.key, method, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    jobs = [
        (args.root, c, m, args.optics)
        for c in design()
        for m in (METHODS if c.primary else ("water", "rank"))
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run, [jobs[3]] if args.smoke else jobs))
