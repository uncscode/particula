# %%
"""Trace Liu's seven fitted bins through the earlier Wilson et al. S1 model."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import particula as par
from chamber import CSTAR, DENSITY, HERE, REPORTED, SIZE
from scipy.integrate import cumulative_trapezoid


def wilson(fractions, t, diam, rho):
    # Wilson et al. (2015), SI sec S1–S2. Seven C* bins are from Liu.
    diffusivity, mw, sigma, temperature = 5e-6, 0.200, 0.05, 297.15
    r0 = diam * 0.5e-9
    m0 = 4 * np.pi / 3 * r0**3 * rho
    q = np.r_[0.0, np.geomspace(1e-7, 2e4, 3000)]
    remaining = np.exp(-q[:, None] * CSTAR) @ fractions
    r = r0 * np.cbrt(remaining)
    kn = (
        3
        * diffusivity
        / (np.sqrt(8 * 8.314462618 * temperature / (np.pi * mw)) * r)
    )
    correction = par.particles.get_vapor_transition_correction(kn, 1.0)
    k = par.dynamics.get_first_order_mass_transport_k(
        r, correction, diffusivity
    )
    kelvin = par.particles.get_kelvin_term(
        r, par.particles.get_kelvin_radius(sigma, rho, mw, temperature)
    )
    time = cumulative_trapezoid(
        m0 * remaining / (60 * k * 1e-9 * kelvin), q, initial=0.0
    )
    return np.interp(t, time, remaining)


def run(out):
    obs = pd.read_csv(HERE.parent / "sources" / "observations.csv")
    rows = []
    for (name, rh), data in obs.groupby(["precursor", "rh"]):
        for label, diam, rho in [
            ("Wilson_original_geometry", 200.0, 1200.0),
            ("Liu_size_and_density", SIZE[name], DENSITY[name]),
        ]:
            times = np.unique(np.r_[200.0, data.time_min])
            source = dict(zip(data.time_min, data.vfr))
            for t, x in zip(
                times, wilson(REPORTED[name, rh], times, diam, rho)
            ):
                rows.append(
                    (name, rh, label, diam, rho, t, source.get(t, np.nan), x)
                )
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns="precursor rh transport_basis initial_nm density_kg_m3 time_min observed_vfr reconstructed_vfr".split(),
    ).to_csv(out / "published.csv", index=False, float_format="%.10g")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--out", type=Path, default=HERE / "results")
    run(p.parse_args().out)
