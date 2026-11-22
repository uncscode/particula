"""Refit neutral laws with the binary share imposed at the held-out 248 K anchor."""

from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from formation import (
    DRAWS,
    HERE,
    Q,
    J,
    T,
    WINDOW,
    SPLITS,
    BASE_BOUNDS,
    BASE_EXTRA,
    inputs,
    weight,
    kernel,
    anchors,
    dump_csv,
)
from scipy.optimize import least_squares


def run(spec):
    idx, share = spec
    row = DRAWS[idx]
    acid, base = inputs(row)
    data = np.array([(T - 248) / 50, np.log(acid), np.log(base)])
    anchor = data[:, 334:335]  # CSV source line 336, includes header
    mask = WINDOW & ~SPLITS["248_low"] & (Q == 0)
    test = WINDOW & SPLITS["248_low"] & (Q == 0)
    w = weight(row, mask)
    y = np.log(J[mask] * row["jscale"])
    bounds = np.array(BASE_BOUNDS * 2 + BASE_EXTRA).T
    free = np.array([1, 3, 4, 5, 7, 8, 9])

    def ev(v, data0):
        p = np.zeros(10)
        p[free] = v
        t, jt, _ = kernel(p[4:], anchor, "assisted", "linear")
        x, a, _ = anchor[:, 0]
        p[0] = t[0] + np.log(share / (1 - share)) + np.exp(p[1]) * x - p[3] * a
        di = np.r_[0, np.exp(p[1]) * x, 0, -a, jt[0]][free]
        pred, jac, _ = kernel(p, data0, "full", "linear")
        return pred, jac[:, free] + jac[:, 0, None] * di, p[0], di

    sols = []
    for start in (
        anchors("248_low", "full", "linear", False)[0],
        anchors("248_low", "full", "linear", False)[
            1 if row["base0"] < 1 else 2
        ],
    ):

        def fun(v):
            r, d, b, di = ev(v, data[:, mask])
            off = b - np.clip(b, -100, 35)
            return np.r_[(r - y) * w, off * 1e4], np.vstack(
                (d * w[:, None], di * (1e4 if off else 0.0))
            )

        cache = {}

        def evaluate(v):
            if "x" not in cache or not np.array_equal(v, cache["x"]):
                cache.update(x=v.copy(), val=fun(v))
            return cache["val"]

        res = least_squares(
            lambda v: evaluate(v)[0],
            start[free],
            jac=lambda v: evaluate(v)[1],
            bounds=bounds[:, free],
            max_nfev=600,
        )
        if res.success and -100.00001 <= ev(res.x, anchor)[2] <= 35.00001:
            sols.append(res)
    if not sols:
        raise RuntimeError(spec)
    best = min(sols, key=lambda s: s.cost)
    pred, _, _, _ = ev(best.x, data[:, test])
    return dict(
        id=idx,
        block=row["block"],
        share=share,
        train_factor=np.exp(np.sqrt(2 * best.cost / mask.sum())),
        heldout_factor=np.exp(
            np.median(abs(pred - np.log(J[test] * row["jscale"])))
        ),
        status=best.status,
    )


if __name__ == "__main__":
    with ProcessPoolExecutor(4) as pool:
        result = list(
            pool.map(
                run, product(range(128), (0.0001, 0.5, 0.9999)), chunksize=2
            )
        )
    dump_csv(HERE / "profile.csv", result)
