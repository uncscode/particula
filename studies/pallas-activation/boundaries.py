"""Check profile classifications by explicitly fitting the boundary populations on each date."""

import numpy as np
import pandas as pd
from calibration import basis
from observations import P, cohort, recipes, volume_spectra
from scipy.optimize import minimize
from study import PRIMARY, Population


def minimum_rmse(z, A, b, M, t, extras=()):
    """Minimize weighted fraction error with linear inventory constraints."""

    def objective(w):
        e = M @ w - t
        return 100 * (e @ e), 200 * M.T @ e

    eq = dict(type="eq", fun=lambda w: A @ w - b, jac=lambda w: A)
    res = minimize(
        objective,
        z,
        jac=True,
        bounds=[(0, 1)] * len(z),
        constraints=[eq, *extras],
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 500},
    )
    assert res.success and max(abs(A @ res.x - b)) < 1e-8
    return np.sqrt(res.fun / 100.0)


def main():
    fits = pd.read_csv(P / "fits.csv").pivot(
        index="date", columns="model", values="rmse"
    )
    pr = pd.read_csv(P / "profiles.csv").set_index("date")
    shares = pd.read_csv(P / "shares.csv.gz")
    spec = volume_spectra(fits.index)
    chem = recipes()
    tab = np.load(P / "thresholds.npz")
    scans, _, _ = cohort()
    checks = []
    for date in fits.index:
        pop = Population(tab, chem.loc[date], spec[date], PRIMARY)
        z = shares.query("date==@date").share.to_numpy()
        scan = scans[scans.date == date].sort_values(["j", "Dp"])
        j = scan.j.to_numpy()
        scale = 1 / np.sqrt(5 * np.bincount(j)[j])
        M = pop.design(np.log(scan.Dp.to_numpy()), j) * scale[:, None]
        t = scan.fraction.to_numpy() * scale
        delta = pop.feature(
            np.diff(basis(np.log([50.0, 120.0]), pop.knots), axis=0)[0],
            pop.c[:, 0],
        )

        sign = minimum_rmse(
            pop.seed,
            pop.A,
            pop.b,
            M,
            t,
            [dict(type="ineq", fun=lambda w: -delta @ w, jac=lambda w: -delta)],
        )
        mask = np.tile(pop.c.max(axis=1) == 1, len(pop.knots))
        sugar = pop.feature(pop.volume, pop.c[:, 1])
        A = np.vstack([pop.A, sugar])[:, mask]
        b = np.r_[pop.b, sugar @ z]
        inventories = pop.inventories(z)
        seed = np.tile(
            pop.c[pop.c.max(axis=1) == 1] @ inventories, len(pop.knots)
        )
        pure = minimum_rmse(seed, A, b, M[:, mask], t)
        cap = fits.loc[date, PRIMARY] + 0.01
        both = (fits.loc[date, ["sorbitol only", "squalane only"]] > cap).all()
        assert (sign > cap) == (pr.loc[date, "delta_lo"] > 1e-6)
        assert (pure <= cap) == (pr.loc[date, "chi_lo"] < 1e-6)
        assert both == (
            pr.loc[date, "organic_sorbitol_lo"] > 1e-6
            and pr.loc[date, "organic_sorbitol_hi"] < 1 - 1e-6
        )
        checks.append((date, cap, sign, pure, both))
    pd.DataFrame(
        checks,
        columns=[
            "date",
            "rmse_cap",
            "no_salt_enrichment_rmse",
            "fixed_inventory_external_rmse",
            "both_required",
        ],
    ).to_csv(P / "boundary-checks.csv", index=False)


if __name__ == "__main__":
    main()
