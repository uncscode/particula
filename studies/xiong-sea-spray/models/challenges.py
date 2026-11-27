# %% Can plausible chemistry or interval timing change the interpretation?
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from inverse import (
    inputs,
    to_sodium,
    PACKET,
    SALT_DENSITY,
    M_SALT,
    PRODUCT_DENSITY,
    M_PRODUCT,
)
from joint import JointModel

OUT = PACKET / "challenges"
# Absolute sulfate ions per initial sodium atom for each pure product.
SULFATE = np.array([0, 1, 0.5])
# Volumes per mole of sodium (cm3 mol-1), NaCl, NaHSO4, anhydrous Na2SO4.
VNA = np.array(
    [M_SALT / SALT_DENSITY, M_PRODUCT / PRODUCT_DENSITY, 142.04 / (2 * 2.70)]
)


def ternary():
    ds, times, rh, y, sd, c, w = inputs()
    rows = []
    for neutral_max in [1.05, 1.10, 1.20]:
        # Prescribed alternative phase branches. The high-RH range deliberately
        # permits a solid or aqueous neutral component. No fitted neutral GF.
        gl = np.vstack([c - w, np.where(rh <= 70, 0.95, 0.90)]) ** 3
        gu = np.vstack([c + w, np.where(rh <= 70, neutral_max, 2.40)]) ** 3
        for mult in [1, 2]:
            for i, d in enumerate(ds):
                T = len(times)
                A = []
                rhs = []
                for j, t in enumerate(times):
                    err = mult * sd[i, j] + 0.005
                    for h in range(len(rh)):
                        low = (y[i, j, h] - err[h]) ** 3
                        high = (y[i, j, h] + err[h]) ** 3
                        for coeff in [VNA * (gl[:, h] - high), VNA * (low - gu[:, h])]:
                            row = np.zeros((T, 3))
                            row[j] = coeff
                            A.append(row.ravel())
                            rhs.append(0)
                    if j:
                        row = np.zeros((T, 3))
                        row[j - 1] = SULFATE
                        row[j] = -SULFATE
                        A.append(row.ravel())
                        rhs.append(0)
                eq = np.kron(np.eye(T), np.ones((1, 3)))
                for j, t in enumerate(times):
                    cost = np.zeros((T, 3))
                    cost[j] = SULFATE
                    kw = dict(
                        A_ub=A,
                        b_ub=rhs,
                        A_eq=eq,
                        b_eq=np.ones(T),
                        bounds=(0, 1),
                        method="highs",
                    )
                    lo = linprog(cost.ravel(), **kw)
                    hi = linprog(-cost.ravel(), **kw)
                    assert lo.success and hi.success
                    rows.append((neutral_max, mult, d, t, lo.fun, -hi.fun))
    result = pd.DataFrame(
        rows,
        columns=[
            "neutral_low_rh_gf_max",
            "scatter_multiple",
            "diameter_nm",
            "time_h",
            "min_sulfate_per_sodium",
            "max_sulfate_per_sodium",
        ],
    )
    result.to_csv(OUT / "ternary.csv", index=False)
    wide = pd.read_csv(PACKET / "population/crossovers.csv").graph_hi.max()
    bounds = []
    for (cap, m, t), g in result.groupby(
        ["neutral_low_rh_gf_max", "scatter_multiple", "time_h"]
    ):
        v = g.set_index("diameter_nm")
        r = v.loc[300, "min_sulfate_per_sodium"] / v.loc[200, "max_sulfate_per_sodium"]
        bounds.append((cap, m, t, r, r / wide))
    pd.DataFrame(
        bounds,
        columns=[
            "neutral_gf_max",
            "scatter_multiple",
            "time_h",
            "min_sulfate_ratio_per_sodium",
            "min_air_mass_ratio_all_spectra",
        ],
    ).to_csv(OUT / "ternary-ratios.csv", index=False)


def interval_tests():
    ds, ts, rh, y, sd, c, w = inputs()
    salt = (
        pd.read_csv(PACKET / "population/at-sizes.csv")
        .query("source=='authors' and shape=='mean' and time_h==0.5")
        .set_index("diameter_nm")
    )
    mass = np.array(
        [salt.loc[d, "initial_salt_ug_m3_decade"] * 96.06 / M_SALT for d in ds]
    )
    rows = []
    for mult in [1, 2]:
        model = JointModel(y, sd * mult + 0.005, c, w)
        N = model.D * model.T
        bounds, center = model.extremes()
        for j, t in enumerate(ts):
            coef = np.zeros((model.D, model.T))
            for d, sgn in [(200, -1), (300, 1)]:
                i = np.flatnonzero(ds == d)[0]
                coef[i, j] += 2 * sgn * mass[i]
                if j:
                    coef[i, j - 1] -= 2 * sgn * mass[i]

            def objective(x):
                return float(coef.ravel() @ to_sodium(x[:N]))

            def grad(x):
                r = (PRODUCT_DENSITY / M_PRODUCT) / (SALT_DENSITY / M_SALT)
                arr = model.zero.copy()
                arr[:N] = coef.ravel() * r / (1 + (r - 1) * x[:N]) ** 2
                return arr

            achieved = []
            for sign in [-1, 1]:
                x = center.copy()
                for _ in range(20):
                    z = model.solve(sign * grad(x)).x
                    values = [
                        (sign * objective(x + s * (z - x)), s)
                        for s in [0, 0.125, 0.25, 0.5, 0.75, 1]
                    ]
                    _, s = min(values)
                    if s == 0:
                        break
                    x = x + s * (z - x)
                assert np.max(model.A @ x - model.b) < 1e-7
                achieved.append(objective(x))
            lo = to_sodium(bounds[..., 0])
            hi = to_sodium(bounds[..., 1])
            # Conservative interval using monotonic reaction at each size.
            delta_lo = np.maximum(0, lo[:, j] - (hi[:, j - 1] if j else 0))
            delta_hi = hi[:, j] - (lo[:, j - 1] if j else 0)
            a, b = [list(ds).index(d) for d in [200, 300]]
            rows.append(
                (
                    mult,
                    t,
                    2 * (mass[b] * delta_lo[b] - mass[a] * delta_hi[a]),
                    min(achieved),
                    max(achieved),
                    2 * (mass[b] * delta_hi[b] - mass[a] * delta_lo[a]),
                )
            )
    pd.DataFrame(
        rows,
        columns=[
            "scatter_multiple",
            "end_h",
            "outer_low",
            "attained_low",
            "attained_high",
            "outer_high",
        ],
    ).to_csv(OUT / "interval-contrast.csv", index=False)


def holdout():
    ds, ts, rh, y, sd, c, w = inputs()
    rows = []
    for name, mask in [("low", rh <= 70), ("high", rh >= 80)]:
        m = JointModel(y[:, :, mask], sd[:, :, mask] + 0.005, c[:, mask], w[:, mask])
        spans, cent = m.extremes()
        v = cent[: m.D * m.T].reshape(m.D, m.T)
        for k in np.flatnonzero(~mask):
            lows = []
            highs = []
            for vv in [spans[..., 0], spans[..., 1]]:
                for sign in [-1, 1]:
                    a, b = c[:, k] + sign * w[:, k]
                    est = np.cbrt((1 - vv) * a**3 + vv * b**3)
                    lows.append(est)
                    highs.append(est)
            lower_growth = np.min(lows, axis=0)
            u = np.max(highs, axis=0)
            pred = np.cbrt((1 - v) * c[0, k] ** 3 + v * c[1, k] ** 3)
            for i, d in enumerate(ds):
                for j, t in enumerate(ts):
                    err = sd[i, j, k] + 0.005
                    rows.append(
                        (
                            name,
                            rh[k],
                            d,
                            t,
                            y[i, j, k],
                            err,
                            pred[i, j],
                            lower_growth[i, j],
                            u[i, j],
                            lower_growth[i, j] <= y[i, j, k] + err
                            and u[i, j] >= y[i, j, k] - err,
                        )
                    )
    pd.DataFrame(
        rows,
        columns=[
            "fit",
            "held_rh",
            "diameter_nm",
            "time_h",
            "observed",
            "spread_and_rounding",
            "frozen_prediction",
            "prediction_lo",
            "prediction_hi",
            "overlap",
        ],
    ).to_csv(OUT / "held-out.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    ternary()
    interval_tests()
    holdout()
