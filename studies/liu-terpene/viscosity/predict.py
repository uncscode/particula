"""Forward tests using Figure 7 and the dry farnesene relaxation estimate."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.constants import Boltzmann
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "transport"))
from internal_supply import ref, simulate


def diffusivity(eta, radius_nm):
    return Boltzmann * ref.TEMP / (6 * np.pi * eta * radius_nm * 1e-9)


def predict_all(out):
    out.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(ROOT / "sources/observations.csv")
    props = pd.read_csv(HERE / "figure7/figure7_viscosity.csv")
    fits = pd.read_csv(ROOT / "model/results/fits.csv").query(
        "model == 'primary'"
    )
    records = []
    for (name, rh), markers in obs.groupby(["precursor", "rh"]):
        analog = ref.CYCLIC[name]
        weights = (
            fits.query("precursor == @analog and rh == @rh")
            .sort_values("cstar_ug_m3")
            .fitted_fraction.to_numpy()
        )
        weights /= weights.sum()
        humidities = [75] if rh == "wet" else [0, 5, 10]
        cases = [("well_mixed", -1, np.nan, np.nan, np.inf, False)]
        for row in props.query(
            "precursor == @name and rh_pct in @humidities"
        ).itertuples():
            for radius in (0.3, 0.4, 0.6):
                D = diffusivity(row.displayed_eta_pa_s, radius)
                cases.append(
                    (
                        "figure7",
                        row.rh_pct,
                        row.displayed_eta_pa_s,
                        radius,
                        D,
                        bool(row.at_plot_ceiling),
                    )
                )
        params = pd.DataFrame(
            cases,
            columns="source rh_pct eta_Pa_s radius_nm D_m2_s displayed_ceiling".split(),
        )
        params["log10_D0"] = np.log10(params.D_m2_s)
        params["loss_sensitivity"] = params["long_lived_fraction"] = 0.0
        arguments = (
            weights,
            name,
            ref.SIZE[name],
            markers.time_min.to_numpy(),
            params,
        )
        pilot = simulate(*arguments, 300, rh=rh)
        values = simulate(*arguments, 1050, rh=rh)
        for i, case in enumerate(cases):
            for j, m in enumerate(markers.itertuples()):
                records.append(
                    [
                        name,
                        rh,
                        analog,
                        *case,
                        m.time_min,
                        m.vfr,
                        values[i, j],
                        values[i, j] - m.vfr,
                        abs(values[i, j] - pilot[i, j]),
                    ]
                )
    cols = "precursor rh composition source rh_pct eta_Pa_s radius_nm D_m2_s displayed_ceiling time_min observed predicted residual refinement_shift".split()
    data = pd.DataFrame(records, columns=cols)
    data.to_csv(
        out / "figure7_forecasts.csv", index=False, float_format="%.11g"
    )
    assert data.refinement_shift.max() < 0.0001
    return fits, obs


def switching_tests(out, fits, obs):
    # Independent adaptive integration allows a change of D at exactly 15 minutes.
    # The two switches below are limiting scenarios. Only eta near 15 min was constrained.
    r0 = 70e-9
    rho = ref.DENSITY["farnesene"]
    mass0 = 4 * np.pi * r0**3 * rho / 3
    times = obs.query("precursor == 'farnesene' and rh == 'dry'").sort_values(
        "time_min"
    )
    cases = []
    for name, column, comp in [
        ("caryophyllene", "fitted_fraction", "cyclic_transfer"),
        ("farnesene", "published_fraction", "published_focal_fit"),
    ]:
        weights = (
            fits.query("precursor == @name and rh == 'dry'")
            .sort_values("cstar_ug_m3")[column]
            .to_numpy()
        )
        weights /= weights.sum()
        for eta in (1.4e8, 2.8e8):
            for radius in (0.3, 0.6):
                D = diffusivity(eta, radius)
                for mode, pair in [
                    ("constant", [D, D]),
                    ("fluid_first_15min", [np.inf, D]),
                    ("fluid_after_15min", [D, np.inf]),
                    ("well_mixed", [np.inf, np.inf]),
                ]:
                    if mode == "well_mixed" and (eta, radius) != (1.4e8, 0.3):
                        continue
                    m = weights.copy()
                    for t0, t1, mobility in zip(
                        [0, 15], [15, times.time_min.max()], pair
                    ):

                        def rhs(t, mass):
                            total = mass.sum()
                            r = r0 * np.cbrt(total)
                            rate, kelvin = ref.kinetics(
                                np.array([r]), rho, ref.Scenario()
                            )
                            tau_gas = (
                                mass0
                                * total
                                / (rate[0] * kelvin[0] * 1e-9 * ref.CSTAR)
                            )
                            return (
                                -60
                                * mass
                                / (tau_gas + r**2 / (np.pi**2 * mobility))
                            )

                        sol = solve_ivp(
                            rhs,
                            (t0, t1),
                            m,
                            rtol=1e-9,
                            atol=1e-12,
                            method="BDF",
                            dense_output=True,
                        )
                        assert sol.success
                        m = sol.y[:, -1]
                        for point in times.query(
                            "@t0 < time_min <= @t1"
                        ).itertuples():
                            pred = sol.sol(point.time_min).sum()
                            cases.append(
                                [
                                    comp,
                                    mode,
                                    eta,
                                    radius,
                                    D,
                                    point.time_min,
                                    point.vfr,
                                    pred,
                                    pred - point.vfr,
                                ]
                            )
    pd.DataFrame(
        cases,
        columns="composition mode eta_Pa_s radius_nm D_m2_s time_min observed predicted residual".split(),
    ).to_csv(
        out / "shape_time_scenarios.csv", index=False, float_format="%.11g"
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--out", type=Path, default=HERE / "results")
    args = cli.parse_args()
    fits, obs = predict_all(args.out)
    switching_tests(args.out, fits, obs)
