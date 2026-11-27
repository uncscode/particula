"""Audit physical classes, payload costs and redraw variation by dry decade."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import read
from metrics import EDGES
from model import KAPPA
from redraw_checkpoint import draw_box, fit_box
from redraw_design import DRAWS, TEST, WINDOWS, cases


def labels(b, grid):
    """Infer the stored physical class from the reconstructed particle state."""
    return np.column_stack(
        (
            np.searchsorted(EDGES[grid], np.cbrt(b.dry.sum(1)), "right"),
            b.water > 1.05 * 49 * (b.dry @ KAPPA) + 100,
            b.old,
            b.pending,
            b.dry[:, 3] > 0,
        )
    )


def audit(job):
    """Check all ten redraws against original class counts and inventories."""
    root, c, start, recipe = job
    original, _, _ = read(root / c.key / f"{start}.npz")
    grid, _, _ = recipe
    out = []
    for i, b in enumerate(original):
        p = fit_box(b, *recipe)
        key, inv, cnt = np.unique(
            labels(b, grid), axis=0, return_inverse=True, return_counts=True
        )
        inventory = np.array(
            [np.bincount(inv, weights=v) for v in (*b.dry.T, b.water)]
        )
        # Use a different random sequence for these additional checks.
        for draw in range(DRAWS):
            rng = np.random.default_rng(
                np.random.SeedSequence([c.seed, c.n, start, draw, i, 123])
            )
            r = draw_box(p, rng)
            actual, ix, n = np.unique(
                labels(r, grid), axis=0, return_inverse=True, return_counts=True
            )
            np.testing.assert_equal(actual, key)
            np.testing.assert_equal(n, cnt)
            np.testing.assert_allclose(
                [np.bincount(ix, weights=v) for v in (*r.dry.T, r.water)],
                inventory,
                rtol=1e-10,
                atol=1e-12,
            )
        size = p["label"][:, 0]
        movable = np.any(p["delta"] != 0, axis=1) & (p["count"] > 1)
        # Five class labels, a count, nine means and nine slopes.
        per_row = sum(
            p[k].nbytes for k in ("count", "label", "mean", "delta")
        ) // len(size)
        for decade, (lo, hi) in enumerate(
            (
                (0, 1),
                (1, grid + 1),
                (grid + 1, 2 * grid + 1),
                (2 * grid + 1, 3 * grid + 1),
                (3 * grid + 1, 3 * grid + 2),
            )
        ):
            mask = (size >= lo) & (size < hi)
            out.append(
                dict(
                    seed=c.seed,
                    n=c.n,
                    start=start,
                    parcel=i,
                    decade=decade - 1,
                    rows=int(mask.sum()),
                    particles=p["count"][mask].sum(),
                    raw_row_bytes=per_row * mask.sum(),
                    rows_with_spread=(mask & movable).sum(),
                    max_rows_in_bin=np.bincount(size[mask]).max(initial=0),
                )
            )
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("selection", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    recipe = json.loads(a.selection.read_text())["recipe"]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = [
            r
            for group in pool.map(
                audit,
                [(a.root, c, t, recipe) for c in cases(TEST) for t in WINDOWS],
            )
            for r in group
        ]
    pd.DataFrame(rows).to_csv(a.out, index=False)
