"""Allocate snapshot rows per dry decade using calibration population only."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import read
from metrics import critical, make_tree, prepare, reference
from model import REGIMES, SEEDS, SS

from optics import Optics

GRID = 10
LIMIT = 32
DEC = np.clip((np.arange(32) - 1) // 10, 0, 2)


def extract(job):
    """Cache nested tree splits and resolved CDF counts for one history."""
    folder, cache, optic = job
    dest = cache / f"{folder.parent.name}-{folder.name}.npz"
    opt = Optics(optic)
    rows = {}
    seen = set()
    for path in sorted(folder.glob("*.npz")):
        boxes, _, t = read(path)
        if t in seen:
            continue
        seen.add(t)
        for i, b in enumerate(boxes):
            c, areas = reference(b, opt)
            prep = prepare(b, GRID)
            means, meta, _ = make_tree(prep, LIMIT, 2)
            x = critical(means[:, :8])
            order = np.argsort(x)
            x = x[order]
            meta = meta[order]
            means = means[order]
            area = opt.areas(means[:, :8], means[:, 8])[:, 0]
            tag = f"{t}_{i}"
            rows[tag] = np.column_stack(
                [
                    meta[:, :3],
                    meta[:, 4],
                    x,
                    area * meta[:, 4] / areas[:, 0].sum(),
                    np.searchsorted(c, x, "left"),
                    np.searchsorted(c, x, "right"),
                    np.searchsorted(SS, x, "left"),
                ]
            )
            rows[tag + "_ref"] = np.searchsorted(c, SS, "right")
            edges = prep[3]
            length = np.diff(edges)
            order = prep[4]
            soot = np.array(
                [
                    b.dry[order[a:z], 3].sum()
                    for a, z in zip(edges[:-1], edges[1:], strict=True)
                ]
            )
            rows[tag + "_bin"] = np.stack([length, soot])
    np.savez_compressed(dest, **rows)


class Snapshot:
    """Cached sorted jump locations for one parcel at one checkpoint."""

    def __init__(self, key, data, ref, bins):
        """Unpack the cached tree, exact CDF and dry-size inventory."""
        self.key = key
        self.b, self.born, self.dead, self.n = data[:, :4].astype(int).T
        self.x, self.area, self.left, self.right, self.channel = data[:, 4:].T
        self.channel = self.channel.astype(int)
        self.ref = ref
        self.count = bins[0].astype(int)
        self.soot = bins[1]
        self.N = self.count.sum()

    def score(self, caps):
        """Return the activation and dry coated-absorption errors."""
        a = caps[self.b]
        active = (self.born <= a) & (a < self.dead)
        count = self.n[active]
        x = self.x[active]
        cum = count.cumsum()
        lo = (x > SS[0]) & (x <= SS[-1]) & np.r_[True, x[1:] != x[:-1]]
        hi = (x >= SS[0]) & (x <= SS[-1]) & np.r_[x[1:] != x[:-1], True]
        chan = (
            (
                np.bincount(self.channel[active], weights=count, minlength=32)[
                    :31
                ].cumsum()
                - self.ref
            )
            * 100
            / self.N
        )
        sup = max(
            abs(chan[[0, -1]]).max(),
            abs(cum[lo] - count[lo] - self.left[active][lo]).max(initial=0)
            * 100
            / self.N,
            abs(cum[hi] - self.right[active][hi]).max(initial=0) * 100 / self.N,
        )
        return (
            np.sqrt(np.mean(chan**2)),
            sup,
            100 * abs(self.area[active].sum() - 1),
        )


def uniform(count, rows):
    """Spread exactly the available rows round-robin over occupied bins."""
    caps = np.zeros(len(count), int)
    while rows:
        ix = np.flatnonzero(caps < count)[:rows]
        caps[ix] += 1
        rows -= len(ix)
    return caps


def cases(root, seeds):
    """Load the frozen population split into memory."""
    samples = []
    for seed in seeds:
        for reg in REGIMES:
            with np.load(root / f"{seed}-{reg}.npz") as z:
                for k in z.files:
                    if len(k.split("_")) != 2:
                        continue
                    t, i = map(int, k.split("_"))
                    samples.append(
                        Snapshot(
                            (seed, reg, t, i),
                            z[k],
                            z[k + "_ref"],
                            z[k + "_bin"],
                        )
                    )
    return samples


def evaluate(samples, alloc):
    """Return worst normalized error and mean record bytes."""
    caps = np.asarray(alloc)[DEC]
    worst = cost = 0
    for sample in samples:
        worst = max(worst, max(np.array(sample.score(caps)) / [2, 5, 5]))
        cost += 80 * np.minimum(sample.count, caps).sum()
    return worst, cost / len(samples)


def run(cache, out):
    """Train decade budgets and test their equal-byte control."""
    cal = cases(cache, SEEDS[:8])
    levels = (4, 8, 12, 16, 20, 24, 28, 32)
    alloc = (4, 4, 4)
    val, cost = evaluate(cal, alloc)
    route = [(*alloc, val, cost)]
    while val > 1 and min(alloc) < LIMIT:
        tries = []
        for i, k in enumerate(alloc):
            if k == LIMIT:
                continue
            candidate = list(alloc)
            candidate[i] = levels[levels.index(k) + 1]
            error, storage = evaluate(cal, candidate)
            if storage > cost:
                tries.append(
                    (
                        (val - error) / (storage - cost),
                        tuple(candidate),
                        error,
                        storage,
                    )
                )
        if not tries:
            break
        _, alloc, val, cost = max(tries)
        route.append((*alloc, val, cost))
    pd.DataFrame(
        route, columns="small middle large worst mean_bytes".split()
    ).to_csv(out / "particle-allocation.csv", index=False)
    del cal
    results = []
    for s in cases(cache, SEEDS[8:]):
        caps = np.asarray(alloc)[DEC]
        rows = np.minimum(caps, s.count).sum()
        for label, cap in [
            ("adaptive", caps),
            ("uniform", uniform(s.count, rows)),
            ("fixed16", np.full(32, 16)),
        ]:
            stored = np.minimum(cap, s.count)
            err = s.score(cap)
            for dec in (-1, 0, 1, 2, 3):
                ix = np.arange(32)
                mask = (
                    (ix == 0)
                    if dec == -1
                    else (
                        (ix == 31)
                        if dec == 3
                        else ((ix >= dec * 10 + 1) & (ix <= dec * 10 + 10))
                    )
                )
                results.append(
                    [
                        *s.key,
                        label,
                        dec,
                        s.count[mask].sum(),
                        s.soot[mask].sum() / s.soot.sum(),
                        stored[mask].sum(),
                        80 * stored[mask].sum(),
                        *err,
                    ]
                )
    pd.DataFrame(
        results,
        columns=(
            "seed regime time parcel method decade count "
            "bc_mass_share rows bytes rmse sup absorption"
        ).split(),
    ).to_csv(
        out / "storage.csv.gz",
        index=False,
        float_format="%.7g",
        compression={"method": "gzip", "mtime": 0},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    a = p.parse_args()
    a.cache.mkdir(exist_ok=True, parents=True)
    jobs = [
        (a.base / str(s) / r, a.cache, a.optics)
        for s in SEEDS
        for r in REGIMES
        if not (a.cache / f"{s}-{r}.npz").exists()
    ]
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(extract, jobs))
    run(a.cache, a.out)
