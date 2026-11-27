"""Predict humid ocimene evaporation with radial diffusion and fixed cyclic inventory."""

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "radial"))
sys.path.insert(0, str(ROOT / "model"))
import chamber
import radial


def calculate(out):
    source = pd.read_csv(HERE / "figure7/figure7_viscosity.csv")
    viscosity = source.query(
        "precursor == 'ocimene' and rh_pct == 75"
    ).displayed_eta_pa_s.item()
    fits = pd.read_csv(ROOT / "model/results/fits.csv")
    weights = (
        fits.query(
            "model == 'primary' and rh == 'wet' and precursor == 'pinene'"
        )
        .sort_values("cstar_ug_m3")
        .fitted_fraction.to_numpy()
    )
    weights /= weights.sum()
    layers = pd.read_csv(HERE / "radial/results/layer-refine-metrics.csv")
    layer_nm = layers.query(
        "training_shells == 320 and forecast_shells == 480"
    ).thickness_nm.item()
    density = chamber.DENSITY["ocimene"]
    kelvin = radial.get_kelvin_radius(
        radial.SIGMA, density, radial.MW, radial.T
    )
    rates = partial(radial.gas_speeds, density=density, kelvin_radius=kelvin)
    records = []
    for diameter in (90, 120, 165, 220):
        radius = diameter * 0.5e-9
        k, curvature = chamber.kinetics(
            np.array([radius]), density, chamber.Scenario()
        )
        np.testing.assert_allclose(
            rates(radius),
            k[0]
            * curvature[0]
            * 1e-9
            * radial.CSTAR
            / (4 * np.pi * radius**2 * density),
            rtol=1e-12,
        )
        # Compare the two measured times at the experimental size with the 20-hour design point.
        times = [1134.86, 1200.0, 1264.0] if diameter == 220 else [1200.0]
        liquid = chamber.predict(
            weights,
            "ocimene",
            "wet",
            diameter,
            times,
            chamber.Scenario(),
            n=10400,
        )
        for thickness in (0.0, layer_nm):
            for tracer in (0.3, 0.4, 0.6):
                diffusion = (
                    radial.KB
                    * radial.T
                    / (6 * np.pi * viscosity * tracer * 1e-9)
                )
                for shells in (32, 80):
                    ts, volume, checks = radial.run(
                        weights,
                        times,
                        diffusion,
                        shells,
                        thick=thickness,
                        sign=1,
                        initial_radius=radius,
                        gas_rates=rates,
                        first_step=1e-6,
                    )
                    for t, reference in zip(times, liquid):
                        records.append(
                            dict(
                                diameter_nm=diameter,
                                time_min=t,
                                viscosity_Pa_s=viscosity,
                                tracer_radius_nm=tracer,
                                D_m2_s=diffusion,
                                layer_nm=thickness,
                                shells=shells,
                                volume_fraction=np.interp(t, ts, volume),
                                well_mixed=reference,
                                mass_budget_error=checks[
                                    "component_mass_budget"
                                ],
                            )
                        )
    result = pd.DataFrame(records)
    keys = ["diameter_nm", "time_min", "tracer_radius_nm", "layer_nm"]
    grids = result.pivot(index=keys, columns="shells", values="volume_fraction")
    assert (grids[80] - grids[32]).abs().max() < 0.0001
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, float_format="%.12g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "ocimene_size.csv")
    calculate(parser.parse_args().out)
