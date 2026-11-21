"""Estimate numerical and sampling sensitivity of the two-mode comparison."""

from time import perf_counter

import numpy as np
import warp as wp
from scipy.stats import t
from scripts.mam4xx_coagulation.particles import (
    EDGES_NM,
    NUMBER,
    TIMES,
    cpu_initial_rate,
    trajectory,
)


def ensemble(diameter, ratio, slots, dt, seeds):
    """Summarize independent number trajectories and spectra at fixed inputs."""
    spectra = []
    drift = 0.0
    for seed in seeds:
        series, error = trajectory(diameter, ratio, slots, seed, dt)
        spectra.append(series)
        drift = max(drift, error)
    spectra = np.asarray(spectra)
    n = spectra.sum(axis=2)
    multiplier = t.ppf(0.975, len(seeds) - 1) / np.sqrt(len(seeds))
    return dict(
        diameter_nm=diameter,
        ratio=ratio,
        slots=slots,
        dt=dt,
        seeds=list(seeds),
        number_cm3=(n.mean(axis=0) / 1e6).tolist(),
        ci95_cm3=(multiplier * n.std(axis=0, ddof=1) / 1e6).tolist(),
        spectrum_cm3=(spectra.mean(axis=0) / 1e6).tolist(),
        spectrum_ci95_cm3=(
            multiplier * spectra.std(axis=0, ddof=1) / 1e6
        ).tolist(),
        max_mass_error=drift,
    )


def simulate():
    """Run the CUDA matrix, independent quadrature and timestep controls."""
    cases = []
    start = perf_counter()
    for diameter in (60.0, 110.0, 220.0):
        for ratio in (0.25, 1.0, 4.0):
            rate = cpu_initial_rate(diameter, ratio, 24)
            np.testing.assert_allclose(
                cpu_initial_rate(diameter, ratio, 32), rate, rtol=1e-7
            )
            for slots in (480, 1920, 3840):
                result = ensemble(diameter, ratio, slots, 60.0, range(101, 113))
                result["initial_rate_cm3_s"] = rate / 1e6
                cases.append(result)
            print(
                f"Finished {diameter:g} nm, Aitken/accumulation {ratio:g}",
                flush=True,
            )
    refinements = [
        ensemble(diameter, ratio, 1920, 30.0, range(301, 313))
        for diameter, ratio in ((60.0, 0.25), (110.0, 1.0), (220.0, 4.0))
    ]
    return dict(
        cases=cases,
        refinements=refinements,
        elapsed_s=perf_counter() - start,
        initial_number_cm3=NUMBER / 1e6,
        times_s=TIMES.tolist(),
        edges_nm=EDGES_NM.tolist(),
        device=wp.get_device("cuda:0").name,
    )
