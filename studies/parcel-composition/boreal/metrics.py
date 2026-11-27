"""Exact activation & coated absorption errors; dry-size entropy diagnostics."""

import ctypes
import os

import numpy as np
from model import KAPPA, RHO, SS
from particula.particles import get_mixing_state_index

EDGES = {
    0: np.geomspace(30, 450, 13),
    **{b: 1.7 * 10 ** (np.arange(3 * b + 1) / b) for b in (4, 8, 10, 12, 16)},
}
LIB = ctypes.CDLL(os.environ.get("BOREAL_TREE", ".artifacts/boreal/tree.so"))
PTR = ctypes.c_void_p
LIB.build.argtypes = [PTR] * 4 + [ctypes.c_int] * 3 + [PTR] * 3


def critical(v):
    """Compute equilibrium critical supersaturation in percent."""
    return 100 * np.expm1(np.sqrt(4 * 2.09915**3 / (27 * (v @ KAPPA))))


def prepare(box, grid):
    """Group original particle slots by dry diameter only."""
    edges = EDGES[grid]
    dry = box.dry
    vol = dry.sum(1)
    d = np.cbrt(vol)
    bin_id = np.searchsorted(edges, d, side="right")
    if not grid:
        bin_id = np.clip(bin_id - 1, 0, 11)
    bins = len(edges) + 1 if grid else 12
    sort = np.argsort(bin_id, kind="stable").astype(np.int32)
    ends = np.r_[0, np.bincount(bin_id, minlength=bins).cumsum()].astype(
        np.int32
    )
    state = np.column_stack([dry, box.water])
    features = np.column_stack(
        [np.log(d) / np.log(edges[1] / edges[0]), dry / vol[:, None]]
    )
    flags = ((dry[:, 3] > 0).astype(int) + 2 * (~box.old)).astype(np.int32)
    return state, features, flags, ends, sort


def make_tree(prep, cap, rule):
    """Build the nested median splits with optional carrier priorities."""
    state, feat, flags, ends, order = prep
    order = order.copy()
    bins = len(ends) - 1
    out = np.zeros((bins * (2 * cap - 1), 9))
    meta = np.zeros((len(out), 5), np.int32)
    args = [x.ctypes.data for x in (state, feat, flags, ends)]
    n = LIB.build(
        *args,
        bins,
        cap,
        rule,
        order.ctypes.data,
        out.ctypes.data,
        meta.ctypes.data,
    )
    return out[:n], meta[:n], order


def reference(box, opt):
    """Evaluate the original particles at full resolution."""
    c = np.sort(critical(box.dry))
    areas = opt.areas(box.dry, box.water)
    return c, areas


def profile(box, prep, areas):
    """Evaluate dry-mass entropy and absorption in each occupied bin."""
    state, _, _, ends, order = prep
    chi = np.full(len(ends) - 1, np.nan)
    soot = np.zeros(len(chi))
    for b, (start, end) in enumerate(zip(ends[:-1], ends[1:], strict=True)):
        if end > start:
            take = order[start:end]
            chi[b] = get_mixing_state_index(state[take, :8] * RHO)
            soot[b] = areas[take, 0].sum()
    return chi, np.diff(ends), soot


def compress(box, grid, cap, rule):
    """Replace each member by its group mean in its original slot."""
    prep = prepare(box, grid)
    limit = int(np.max(cap))
    means, meta, order = make_tree(prep, limit, rule)
    local = np.broadcast_to(cap, (len(prep[3]) - 1,))[meta[:, 0]]
    chosen = (meta[:, 1] <= local) & (meta[:, 2] > local)
    # Put each expanded row into its original slot for matched random-event
    # coupling. The sidecar (slot labels) is counted separately in restart
    # bytes.
    for center, (_, _, _, s, n) in zip(
        means[chosen], meta[chosen], strict=True
    ):
        ix = order[s : s + n]
        box.dry[ix] = center[:8]
        box.water[ix] = center[8]
    return int(chosen.sum())


LIB.scores.argtypes = (
    [ctypes.c_int] * 3 + [PTR] * 8 + [ctypes.c_double] * 2 + [PTR]
)


def budget(ref, means, meta, opt, cap):
    """Score every integer cap from the same nested tree."""
    c, areas = ref
    raw = critical(means[:, :8])
    order = np.argsort(raw)
    x = raw[order]
    a = opt.areas(means[:, :8], means[:, 8])[order]
    m = np.ascontiguousarray(meta[order])
    ptrs = [
        m,
        x,
        np.searchsorted(c, x, "left").astype(np.int32),
        np.searchsorted(c, x, "right").astype(np.int32),
        np.searchsorted(SS, x, "left").astype(np.int32),
        a,
        areas.sum(0),
        np.searchsorted(c, SS, "right").astype(np.int32),
    ]
    out = np.empty((cap, 6))
    LIB.scores(
        len(x),
        cap,
        len(c),
        *[v.ctypes.data for v in ptrs],
        SS[0],
        SS[-1],
        out.ctypes.data,
    )
    return out
