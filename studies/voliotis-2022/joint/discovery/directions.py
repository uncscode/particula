"""Locally refit shared chemistry when proposing a new candidate reaction."""

import numpy as np
from reactions import catalog
from scipy.optimize import linprog

UPPER = np.array([r["upper"] for r in catalog()])


def directions(model, obj, state, ids, remaining):
    """Linearize the run errors and solve a bounded minimax problem."""
    p = np.array(state["p"])
    k = np.array(state["k"])
    base = list(range(9)) + ([9] if obj == "joint" else []) + [12]
    active = ids + remaining
    full = np.r_[p, k]
    idx = np.array(base + [16 + h for h in active])
    low = np.r_[
        [1 / 12] * 3,
        [0.0] * 6,
        2.818,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        np.zeros(len(k)),
    ]
    high = np.r_[
        [4.0] * 3, [2.0] * 6, 16.5515, 1.0, 300.0, 100.0, 1.0, 0.1, 5.0, UPPER
    ]
    scale = np.minimum(
        (high - low) * 0.25, np.abs(full) * 0.35 + 0.015 * (high - low)
    )
    scale[16:] = np.minimum(UPPER * 0.45, k * 0.4 + [0.16 * v for v in UPPER])
    step = scale[idx]
    z = full[idx]
    delta = np.where(z + step * 0.012 <= high[idx], 1.0, -1.0) * step * 0.012
    probes = np.tile(full, (len(idx) + 1, 1))
    probes[1:, idx] += np.diag(delta)
    out = model.run(
        probes[:, :16],
        probes[:, 16:],
        active,
        np.array(state["dose"])[:, None],
        obj,
        sub=2,
    )
    error = out[1][:, :, :2] if obj == "joint" else out[1][:, :, :1]
    y = error.reshape(len(probes), -1)
    jac = (y[1:] - y[0]).T / (delta / step)
    bounds = list(
        zip(
            np.maximum(-1, (low[idx] - z) / step),
            np.minimum(1, (high[idx] - z) / step),
            strict=True,
        )
    )
    proposals = []
    for h in remaining or [None]:
        n = len(base) + len(ids)
        use = list(range(n)) + (
            [] if h is None else [len(base) + active.index(h)]
        )
        a = np.column_stack((jac[:, use], -np.ones(y.shape[1])))
        res = linprog(
            np.r_[np.zeros(len(use)), 1.0],
            A_ub=a,
            b_ub=-y[0],
            bounds=[bounds[j] for j in use] + [(0.0, None)],
            method="highs",
        )
        if not res.success:
            raise ArithmeticError(res.message)
        group = [full.copy()]
        for fraction in (0.2, 0.55, 1.0):
            pt = full.copy()
            pt[idx[use]] = np.clip(
                z[use] + fraction * step[use] * res.x[:-1],
                low[idx[use]],
                high[idx[use]],
            )
            group.append(pt)
        if h is not None:
            for fraction in (0.001, 0.03, 0.3):
                pt = full.copy()
                pt[16 + h] = fraction * UPPER[h]
                group.append(pt)
        proposals.append(group)
    return np.asarray(proposals)


def compete(model, obj, state, ids, remaining, doses):
    """Rank actual full-grid fits after testing the linearized suggestions."""
    grouped = directions(model, obj, state, ids, remaining)
    points = grouped.reshape(-1, grouped.shape[-1])
    out = model.run(
        points[:, :16], points[:, 16:], ids + remaining, doses, obj, sub=2
    )
    winners = []
    for j in range(len(grouped)):
        start = j * grouped.shape[1]
        stop = start + grouped.shape[1]
        # Break exact minimax ties using mean mass error.
        pick = (
            start
            + np.lexsort(
                (out[1][start:stop, :, 0].mean(1), out[0][start:stop])
            )[0]
        )
        winners.append((points[pick, :16], points[pick, 16:], out, pick))
    return winners
