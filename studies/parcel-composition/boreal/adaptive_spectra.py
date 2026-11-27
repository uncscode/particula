"""Score expected redraw counts in independent dry-size channels."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from adaptive_grid import choose_edges, fixed_pinned_edges
from adaptive_trials import ARMS
from campaign import read
from redraw_checkpoint import fit_box
from response_course import ASCENT, CASES, SHORT

# Same fractional width as the motivating counter. Keep disjoint test channels.
CUTS = 1.72 * (160 / 150) ** np.arange(107)


def expected_below(rows, cuts):
    """Expected counts below each cutoff, including odd central particles."""
    center = rows["mean"][:, :8].sum(1)
    width = np.abs(rows["delta"][:, :8].sum(1))
    count = rows["count"]
    # Positive/negative pairs both have uniform marginal dry volume.
    offset = cuts[:, None] ** 3 - center
    moving = width > 0
    prob = (offset > 0).astype(float)
    prob[:, moving] = np.clip(
        0.5 + 0.5 * offset[:, moving] / width[moving], 0.0, 1.0
    )
    return prob @ (count - count % 2) + (offset > 0) @ (count % 2)


def sweep(job):
    """Read the identical population for each grid and measure expected bias."""
    root, c, start = job
    boxes, _, _ = read(root / "response-course" / c.key / f"{start}.npz")
    tables = []
    for i, b in enumerate(boxes):
        v = b.dry.sum(1)
        for arm in ("fixed", *ARMS):
            edges = None
            if arm == "fixed-pinned":
                edges = fixed_pinned_edges()
            elif arm != "fixed":
                edges = choose_edges(b, arm)
            rows = fit_box(b, size_edges=edges)
            target = expected_below(rows, np.array([150.0, 160.0]))
            true = np.count_nonzero((v >= 150**3) & (v < 160**3))
            if arm == "fixed-pinned" or (
                edges is not None and all(p in edges for p in (150.0, 160.0))
            ):
                np.testing.assert_allclose(np.diff(target), true, atol=1e-7)
            full = np.diff(np.searchsorted(np.sort(v), CUTS**3, side="left"))
            pred = np.diff(expected_below(rows, CUTS))
            tables.append(
                pd.DataFrame(
                    dict(
                        seed=c.seed,
                        start=start,
                        parcel=i,
                        method=arm,
                        lo_nm=CUTS[:-1],
                        hi_nm=CUTS[1:],
                        full_count=full,
                        expected_count=pred,
                    )
                )
            )
    return pd.concat(tables)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    jobs = [(a.root, c, t) for c in CASES for t in (*SHORT, ASCENT)]
    with ProcessPoolExecutor(a.workers) as pool:
        result = pd.concat(pool.map(sweep, jobs), ignore_index=True)
    result.to_csv(a.out, index=False, float_format="%.12g")
