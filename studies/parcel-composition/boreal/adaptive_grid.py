"""Relocate dry-size edges at a fixed bin and representative-row budget."""

import numpy as np
from metrics import EDGES
from redraw_checkpoint import DEFAULT_RECIPE

PINS = (150.0, 160.0)
MODES = ("density", "pinned")


def redistribute(values, lo, hi, bins, cuts):
    """Split a dense block by equal shares of number and log-size coverage."""
    # The histogram smooths ties from identical freshly nucleated particles.
    mesh = np.geomspace(lo, hi, 32 * bins + 1)
    counts, _ = np.histogram(values, bins=mesh)
    weight = 0.5 * (
        counts / counts.sum() + np.full(len(counts), 1 / len(counts))
    )
    cdf = np.r_[0.0, weight.cumsum()]
    breaks = np.array([lo, *cuts, hi])
    levels = np.interp(np.log(breaks), np.log(mesh), cdf)
    # Reserve one interval between every pair of instrument cutoffs.
    quotas = np.ones(len(breaks) - 1, dtype=int)
    for _ in range(bins - quotas.sum()):
        quotas[np.argmax(np.diff(levels) / (quotas + 1))] += 1
    result = [lo]
    for high, a, b, n in zip(
        breaks[1:], levels[:-1], levels[1:], quotas, strict=True
    ):
        targets = np.linspace(a, b, n + 1)[1:-1]
        result.extend(np.exp(np.interp(targets, cdf, np.log(mesh))))
        result.append(high)
    return result


def choose_edges(box, mode):
    """Keep sparse bins and decade endpoints fixed, and encode parcel edges."""
    grid, cap, _ = DEFAULT_RECIPE
    if mode not in MODES:
        raise ValueError("Unknown adaptive grid")
    original = EDGES[grid]
    edges = original.copy()
    d = np.cbrt(box.dry.sum(1))
    count = np.bincount(
        np.searchsorted(original, d, "right"), minlength=len(edges) + 1
    )[1:-1]
    for decade in range(0, len(count), grid):
        dense = np.r_[False, count[decade : decade + grid] >= cap, False]
        starts = np.flatnonzero(~dense[:-1] & dense[1:]) + decade
        ends = np.flatnonzero(dense[:-1] & ~dense[1:]) + decade
        for i, j in zip(starts, ends, strict=True):
            lo, hi = original[[i, j]]
            cuts = [p for p in PINS if lo < p < hi] if mode == "pinned" else []
            if j - i < max(2, len(cuts) + 1):
                continue
            x = d[(d >= lo) & (d < hi)]
            trial = redistribute(x, lo, hi, j - i, cuts)
            # A moved edge cannot turn a previously dense bin into a sparse bin.
            if np.min(np.histogram(x, bins=trial)[0]) >= cap:
                edges[i : j + 1] = trial
    return edges


def fixed_pinned_edges():
    """Move the last two interior boundaries below 170 nm to the cutoffs."""
    result = EDGES[DEFAULT_RECIPE[0]].copy()
    upper = np.searchsorted(result, np.mean(PINS))
    result[upper - 2 : upper] = PINS
    return result
