"""Fit common and delayed production before scoring held-out chamber data."""

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scripts.pinic_isomers.campaign import first_crossing
from scripts.pinic_isomers.timing_gpu import Exchange


def fitted(model, y, delayed):
    """Profile channel gains and differentiate the training projection."""
    t = model.time
    train = t <= 0.75 * t[-1]
    active = [0, 1, 2, 3] if delayed else [0, 1]

    def evaluate(x):
        p = np.array([3.0, 0.0, 3.0, 0.0])
        p[active] = x
        h, j = model.predict(p, delayed)
        htr = h[:, train]
        jtr = j[:, train, :]
        den = (htr * htr).sum(axis=1)
        gains = (y[:, train] * htr).sum(axis=1) / den
        dg = (
            np.einsum("st,stp->sp", y[:, train] - 2 * gains[:, None] * htr, jtr)
            / den[:, None]
        )
        pred = h * gains[:, None]
        jac = gains[:, None, None] * j + h[:, :, None] * dg[:, None, :]
        return pred, jac[:, :, active], p, gains

    lo = np.array([np.log(1), np.log(0.01), np.log(1), 0.0])[active]
    hi = np.array([np.log(500), np.log(100), np.log(300), 1.0])[active]

    def fun(x):
        return (evaluate(x)[0] - y)[:, train].ravel()

    def jac(x):
        return evaluate(x)[1][:, train, :].reshape(-1, len(active))

    solutions = []
    for a, b in [(12.0, 0.3), (80.0, 30.0)]:
        guess = np.array([np.log(a), np.log(b), np.log(25.0), 0.6])[active]
        fit = least_squares(
            fun, guess, jac=jac, bounds=(lo, hi), ftol=1e-10, gtol=1e-9
        )
        if not fit.success:
            raise ValueError(fit.message)
        solutions.append(fit)
    best = min(solutions, key=lambda s: s.cost)
    pred, der, p, gains = evaluate(best.x)
    # Assess nonzero derivatives at an interior point, away from active bounds.
    x = (lo + hi) / 2
    ref = jac(x)
    for k in range(len(x)):
        diff = np.zeros_like(x)
        diff[k] = 1e-5
        fd = (fun(x + diff) - fun(x - diff)) / (2e-5)
        np.testing.assert_allclose(fd, ref[:, k], rtol=2e-5, atol=2e-8)
    return dict(
        params=p.tolist(),
        gains=gains.tolist(),
        pred=pred.tolist(),
        train_rmse=np.sqrt(
            np.mean((pred[:, train] - y[:, train]) ** 2, axis=1)
        ).tolist(),
        holdout_rmse=np.sqrt(
            np.mean((pred[:, ~train] - y[:, ~train]) ** 2, axis=1)
        ).tolist(),
        half_times=[
            first_crossing(t, curve / curve[-5:].mean(), 3) for curve in pred
        ],
        cost=float(best.cost),
    )


def simulate():
    """Bracket area, refit both explanations and predict the final quarter."""
    folder = Path(__file__).resolve().parents[2] / "docs/data/pinic-isomers"
    data = np.loadtxt(folder / "traces.csv", delimiter=",", skiprows=1)
    data = data[data[:, 0] >= 0]
    t = data[:, 0]
    raw = data[:, 2:].T
    scale = raw[:, t <= 0.75 * t[-1]].max(axis=1)
    y = raw / scale[:, None]
    smps = np.loadtxt(folder / "smps.csv", delimiter=",", skiprows=1)
    rows = []
    for radius, dt, early in [
        (40.0, 0.25, "ramp"),
        (80.0, 0.25, "ramp"),
        (160.0, 0.25, "ramp"),
        (80.0, 0.125, "ramp"),
        (80.0, 0.25, "flat"),
    ]:
        model = Exchange(smps, t, radius, dt=dt, early=early)
        for delayed in (False, True):
            f = fitted(model, y, delayed)
            f.update(radius=radius, dt=dt, early=early, delayed=delayed)
            f["late_scale"] = scale.tolist()
            rows.append(f)
    # Forecast separation under repeats with the trained gains and parameters.
    forecasts = []
    central = [
        r
        for r in rows
        if r["radius"] == 80 and r["dt"] == 0.25 and r["early"] == "ramp"
    ]
    for load in (0.25, 0.5, 1.0, 2.0, 4.0):
        model = Exchange(smps, t, 80.0, loading=load)
        pred = [
            model.predict(c["params"], c["delayed"])[0]
            * np.array(c["gains"])[:, None]
            for c in central
        ]
        forecasts.append(
            [load, float(np.sqrt(np.mean((pred[0] - pred[1]) ** 2)))]
        )
    return dict(
        time=t.tolist(),
        obs=y.tolist(),
        smps=smps.tolist(),
        cases=rows,
        repeat=forecasts,
    )
