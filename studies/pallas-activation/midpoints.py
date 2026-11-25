"""Test whether the published midpoints alone require salt enrichment."""

import numpy as np
import pandas as pd
from boundaries import minimum_rmse
from calibration import basis
from observations import P, SET, recipes, screened_scans, volume_spectra
from study import PRIMARY, Population


def compare():
    """Fit the same population with and without enrichment at archived D50."""
    archived, _ = screened_scans()
    archived["date"] = archived.date.dt.strftime("%Y-%m-%d")
    whole = pd.read_csv(P / "boundary-checks.csv").set_index("date")
    fits = (
        pd.read_csv(P / "fits.csv").query("model == @PRIMARY").set_index("date")
    )
    chemistry = recipes()
    spectra = volume_spectra(fits.index)
    results = []
    with np.load(P / "thresholds.npz") as table:
        for date, fit in fits.iterrows():
            obs = archived.loc[archived.date == date].sort_values("SS")
            np.testing.assert_allclose(obs.SS, SET)
            pop = Population(table, chemistry.loc[date], spectra[date], PRIMARY)
            response = pop.design(np.log(obs.D.to_numpy()), np.arange(len(SET)))
            scale = np.sqrt(len(SET))
            design = response / scale
            target = np.full(len(SET), 0.5 / scale)
            delta = pop.feature(
                np.diff(basis(np.log([50.0, 120.0]), pop.knots), axis=0)[0],
                pop.c[:, 0],
            )
            unrestricted = minimum_rmse(pop.seed, pop.A, pop.b, design, target)
            restricted = minimum_rmse(
                pop.seed,
                pop.A,
                pop.b,
                design,
                target,
                [
                    dict(
                        type="ineq",
                        fun=lambda w: -delta @ w,
                        jac=lambda w: -delta,
                    )
                ],
            )
            assert restricted >= unrestricted - 1e-6
            results.append(
                dict(
                    date=date,
                    midpoint_rmse=unrestricted,
                    midpoint_no_enrichment=restricted,
                    curve_rmse=fit.rmse,
                    curve_no_enrichment=whole.loc[
                        date, "no_salt_enrichment_rmse"
                    ],
                )
            )
    return pd.DataFrame(results)


if __name__ == "__main__":
    daily = compare()
    daily.to_csv(P / "midpoint-check.csv", index=False, float_format="%.10g")
    for tolerance in (0.005, 0.01, 0.02):
        for source in ("midpoint", "curve"):
            cost = daily[f"{source}_no_enrichment"] - daily[f"{source}_rmse"]
            print(source, tolerance, int((cost > tolerance).sum()), cost.mean())
