"""Fit number-weighted chi on the 48-bin dry grid & score each decade."""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from model import REGIMES, SEEDS

X = (np.arange(48) + 0.5) / 16
D = 1.7 * 10**X
HOLD = set(SEEDS[8:])


@lru_cache(maxsize=1024)
def basis(knots):
    """Return linear-spline basis heights on the reference grid."""
    eye = np.eye(len(knots))
    return np.column_stack([np.interp(X, knots, row) for row in eye])


def weights_error(y, p, w):
    """Calculate number-weighted RMS error on occupied bins."""
    return np.sqrt(np.sum(w * (p - y) ** 2) / w.sum()) if w.sum() else np.nan


def _heights(design, y, w):
    active = w > 0
    root = np.sqrt(w[active])
    return np.linalg.lstsq(
        root[:, None] * design[active], root * y[active], rcond=1e-10
    )[0]


def fit(design, y, w):
    """Fit and bound the profile using number-weighted least squares."""
    return np.clip(design @ _heights(design, y, w), 0, 1)


def positions(alloc):
    """Share the boundary knots of neighboring decades."""
    return tuple(
        np.unique(
            np.concatenate(
                [np.linspace(i, i + 1, n + 1) for i, n in enumerate(alloc)]
            )
        )
    )


def load_profiles(root, seeds=SEEDS):
    """Load the 48 interior dry-size bins, retaining undefined weights."""
    data = []
    for seed in seeds:
        for reg in REGIMES:
            with np.load(root / str(seed) / reg / "snapshot.npz") as z:
                a = z["profiles"]
            a = a[(a[:, 2] == 16)].reshape(-1, 50, 7)
            for row in a:
                y = row[1:-1, 5]
                w = np.where(np.isfinite(y), row[1:-1, 4], 0)
                data.append(
                    ((seed, reg, *map(int, row[0, :2])), np.nan_to_num(y), w)
                )
    return data


def criteria(y, p, w):
    """Score overall RMSE and each decade with 200 resolved particles."""
    whole = weights_error(y, p, w)
    dec = [
        weights_error(y[s : s + 16], p[s : s + 16], w[s : s + 16])
        if w[s : s + 16].sum() >= 200
        else np.nan
        for s in (0, 16, 32)
    ]
    return whole, *dec


def norm(err):
    """Normalize each applicable error by its declared tolerance."""
    return np.nanmax(np.array(err) / [0.02, 0.05, 0.05, 0.05])


def train(data):
    """Learn the common shape and allocate added knot heights."""
    templates = []
    for _, y, w in data:
        mean = np.average(y, weights=w)
        sd = weights_error(y, mean, w)
        templates.append(np.where(w > 0, (y - mean) / max(1e-12, sd), np.nan))
    t = np.stack(templates)
    template = np.divide(
        np.nansum(t, axis=0),
        np.isfinite(t).sum(0),
        out=np.zeros(48),
        where=np.isfinite(t).any(0),
    )
    alloc = (1, 1, 1)
    # Each extra interval adds one shared-position float64 height.
    route = []
    cache = {}
    for _ in range(22):

        def score(trial):
            if trial not in cache:
                design = basis(positions(trial))
                cache[trial] = max(
                    norm(criteria(y, fit(design, y, w), w)) for _, y, w in data
                )
            return cache[trial]

        route.append((alloc, score(alloc)))
        if score(alloc) <= 1:
            break
        options = [
            tuple(k + (i == j) for i, k in enumerate(alloc))
            for j in range(3)
            if alloc[j] < 16
        ]
        alloc = min(options, key=score)
    return template, route


def calibrate(root, out):
    """Persist both learned quantities before held-out scoring."""
    template, route = train(load_profiles(root, SEEDS[:8]))
    np.savetxt(
        out / "shared-chi-shape.csv",
        np.c_[D, template],
        delimiter=",",
        header="dry_nm,standardized_shape",
        comments="",
    )
    np.savetxt(
        out / "knot-allocation.csv",
        [(*a, e) for a, e in route],
        delimiter=",",
        header="small,middle,large,worst_normalized_calibration",
        comments="",
    )
    return template, route[-1][0]


def movable(y, w):
    """Fit six heights and search over the four interior knot positions."""
    knots = np.linspace(X[0], X[-1], 6)
    prediction = fit(basis(tuple(knots)), y, w)
    loss = norm(criteria(y, prediction, w))
    # Two coordinate sweeps, same search rule for every snapshot. Positions
    # may move across decades; the endpoints are common to all encodings.
    for order in ((1, 2, 3, 4), (4, 3, 2, 1)):
        for j in order:
            for point in X[(X > knots[j - 1]) & (X < knots[j + 1])]:
                trial = knots.copy()
                trial[j] = point
                curve = fit(basis(tuple(trial)), y, w)
                error = norm(criteria(y, curve, w))
                if error < loss:
                    knots, prediction, loss = trial, curve, error
    return knots, prediction


def save_heights(root, out):
    """Save replayable float64 heights, coordinates and undefined-bin flags."""
    alloc = np.loadtxt(out / "knot-allocation.csv", delimiter=",", skiprows=1)[
        -1, :3
    ]
    knots = positions(alloc.astype(int))
    design = basis(knots)
    heights = []
    flags = []
    keys = []
    for (seed, reg, t, parcel), y, w in load_profiles(root, SEEDS[8:]):
        heights.append(_heights(design, y, w))
        flags.append(np.packbits(w > 0))
        keys.append((seed, list(REGIMES).index(reg), t, parcel))
    np.savez_compressed(
        out / "held-out-chi.npz",
        keys=np.array(keys, np.int32),
        regimes=list(REGIMES),
        log10_d_over_1p7_nm=knots,
        heights=heights,
        defined_bits=flags,
    )


def evaluate(root, out):
    """Compare all encodings and record the largest missed feature."""
    template, alloc = calibrate(root, out)
    data = load_profiles(root)
    methods = {
        "quadratic": np.column_stack((np.ones(48), X, X**2)),
        "six legacy": basis(
            tuple(
                np.linspace(
                    np.log10(30 / 1.7) + np.log10(15) / 24,
                    np.log10(450 / 1.7) - np.log10(15) / 24,
                    6,
                )
            )
        ),
        "six full": basis(tuple(np.linspace(X[0], X[-1], 6))),
        **{f"{i}/dec": basis(positions((i,) * 3)) for i in (2, 4, 6, 8)},
        "adaptive": basis(positions(alloc)),
        "equal bytes": basis(tuple(np.linspace(0, 3, sum(alloc) + 1))),
    }
    rows = []
    worst = {}
    locations = []
    for key, y, w in data:
        mean = np.average(y, weights=w)
        sd = weights_error(y, mean, w)
        centered_template = template - np.average(template, weights=w)
        fitters = {
            "mean": np.full(48, mean),
            "mean+sd": np.clip(
                mean
                + sd
                * centered_template
                / max(weights_error(centered_template, 0, w), 1e-12),
                0,
                1,
            ),
        }
        fitters.update({k: fit(design, y, w) for k, design in methods.items()})
        knots = [0.0, 3.0]
        for _ in range(4):
            prediction = fit(basis(tuple(knots)), y, w)
            residual = w * (prediction - y) ** 2
            residual[np.isin(X, knots)] = -1
            knots = sorted([*knots, X[np.argmax(residual)]])
        fitters["six movable"] = fit(basis(tuple(knots)), y, w)
        locations.append([*key, "six movable", *knots[1:-1]])
        knots, fitters["six movable refined"] = movable(y, w)
        locations.append([*key, "six movable refined", *knots[1:-1]])
        for method, pred in fitters.items():
            errs = criteria(y, pred, w)
            k = np.argmax(np.where(w > 0, abs(pred - y), -1))
            rows.append(
                [
                    *key,
                    method,
                    *errs,
                    D[k],
                    pred[k] - y[k],
                    *w.reshape(3, 16).sum(1),
                ]
            )
            if key[0] in HOLD and method == "six full":
                score = norm(errs)
                if score > worst.get("score", -1):
                    worst = dict(score=score, key=key)
    pd.DataFrame(
        locations, columns="seed regime time parcel method x1 x2 x3 x4".split()
    ).to_csv(
        out / "movable-knots.csv.gz",
        index=False,
        float_format="%.8g",
        compression={"method": "gzip", "mtime": 0},
    )
    frame = pd.DataFrame(
        rows,
        columns=(
            "seed regime time parcel method overall small middle "
            "large feature_nm signed_feature small_n middle_n "
            "large_n"
        ).split(),
    )
    frame.to_csv(
        out / "shape-scores.csv.gz",
        index=False,
        float_format="%.7g",
        compression={"method": "gzip", "mtime": 0},
    )
    print("frozen", alloc, "hardest six", worst, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.scores, args.output)
    save_heights(args.scores, args.output)
