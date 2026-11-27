# %%
"""Predict dry β-farnesene evaporation from coalescence viscosity and tracer size."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.constants import Boltzmann

DEFAULT_PACKET = Path(__file__).resolve().parents[2]
TEMPERATURE_K = 297.15
CROSSOVER_PA_S = 1e-3
FRACTIONAL_EXPONENT = 0.93


def diffusion(eta, radius_nm, exponent, temperature_k=TEMPERATURE_K):
    radius_m = radius_nm * 1e-9
    reference = (
        Boltzmann * temperature_k / (6 * np.pi * radius_m * CROSSOVER_PA_S)
    )
    return reference * (CROSSOVER_PA_S / eta) ** exponent


def execute(packet, out):
    sys.path.insert(0, str(packet / "transport"))
    from internal_supply import simulate

    out.mkdir(parents=True, exist_ok=True)
    observations = pd.read_csv(packet / "sources" / "observations.csv")
    data = observations.query(
        "precursor=='farnesene' and rh=='dry'"
    ).sort_values("time_min")
    fits = pd.read_csv(packet / "model" / "results" / "fits.csv")
    cyclic = fits.query(
        "precursor=='caryophyllene' and rh=='dry' and model=='primary'"
    ).sort_values("cstar_ug_m3")
    weights = cyclic.fitted_fraction.to_numpy()
    weights /= weights.sum()
    case_rows = []
    for law, exponent in [
        ("classical", 1.0),
        ("Evoy_fractional", FRACTIONAL_EXPONENT),
    ]:
        for eta in [1.4e8, 2.1e8, 2.8e8]:
            for radius_nm in [0.3, 0.38, 0.4, 0.6]:
                diffusivity = diffusion(eta, radius_nm, exponent)
                case_rows.append((law, exponent, eta, radius_nm, diffusivity))
    cases = pd.DataFrame(
        case_rows,
        columns=[
            "law",
            "exponent",
            "viscosity_Pa_s",
            "tracer_radius_nm",
            "diffusivity_m2_s",
        ],
    )
    para = pd.DataFrame(
        dict(
            log10_D0=np.log10(cases.diffusivity_m2_s),
            loss_sensitivity=0.0,
            long_lived_fraction=0.0,
        )
    )
    times = np.unique(
        np.r_[np.linspace(0.0, 1200.0, 121), data.time_min.to_numpy()]
    )
    output = simulate(weights, "farnesene", 140.0, times, para, 800)
    fine = simulate(weights, "farnesene", 140.0, times, para, 2400)
    case_id = np.arange(len(cases))
    cases.insert(0, "case_id", case_id)
    cases["max_refinement_vfr"] = np.max(abs(fine - output), axis=1)
    assert cases.max_refinement_vfr.max() < 1e-4
    ids = np.searchsorted(times, data.time_min.to_numpy())
    score = fine[:, ids] - data.vfr.to_numpy()
    cases["early_rmse_vfr"] = np.mean(score[:, :-1] ** 2, axis=1) ** 0.5
    cases["overnight_predicted_vfr"] = fine[:, ids[-1]]
    cases["overnight_signed_vfr"] = score[:, -1]
    cases["all_mae_vfr"] = np.mean(abs(score), axis=1)
    cases["initial_mixing_min"] = (70e-9) ** 2 / (
        np.pi**2 * cases.diffusivity_m2_s * 60
    )
    rows = pd.DataFrame(
        {
            "case_id": np.repeat(case_id, len(times)),
            "time_min": np.tile(times, len(cases)),
            "predicted_vfr": fine.ravel(),
        }
    )
    joins = rows.merge(
        data[["time_min", "vfr"]].rename(columns={"vfr": "observed_vfr"}),
        on="time_min",
        how="left",
    )
    cases.to_csv(
        out / "fractional_cases.csv", index=False, float_format="%.11g"
    )
    joins.to_csv(
        out / "fractional_curves.csv", index=False, float_format="%.11g"
    )
    cyclic[["cstar_ug_m3", "fitted_fraction"]].to_csv(
        out / "cyclic_composition.csv", index=False, float_format="%.11g"
    )
    # Check Figure 3's highest reported proxy viscosity from Table S5.
    proxy = []
    for law, exponent in [
        ("classical", 1.0),
        ("Evoy_fractional", FRACTIONAL_EXPONENT),
    ]:
        d = diffusion(10**6.60, 0.589, exponent, 295.0)
        proxy.append([law, 6.60, 1.15, 0.589, 295.0, 3.79e-19, 2.75e-19, d])
    pd.DataFrame(
        proxy,
        columns="law log10_eta_Pa_s log10_eta_halfspan radius_nm temperature_K measured_D_m2_s measured_plusminus_D_m2_s prediction_D_m2_s".split(),
    ).to_csv(out / "proxy_anchor.csv", index=False, float_format="%.11g")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    p.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    a = p.parse_args()
    execute(a.packet, a.out)
