"""Scalar inference with independent synthetic observations and known noise."""

import numpy as np
from scipy.optimize import brentq, least_squares
from scripts import assess_native_condensation as case
from scripts.assess_native_adaptive import local_reference
from scripts.inverse_studies.scalar_reference import growth_time, recover_radius

DURATION = 1.4
SIGMA_UM = 0.01
TRUTHS_UM = (0.804, 0.843, 0.957, 1.113, 1.196)
STARTS_UM = (0.81, 1.0, 1.19)
SEED = 20261119


def quadrature_endpoint(initial_um):
    """Invert the travel time with a bracket below the depletion radius."""
    depletion = np.cbrt(initial_um**3 + case.VAPOR / (case.NUMBER * case.MASS))
    return brentq(
        lambda final: growth_time(initial_um, final) - DURATION,
        initial_um,
        depletion * (1 - 1e-9),
        xtol=1e-13,
    )


def independent_endpoint(initial_um):
    """Integrate the radius ODE with two independent adaptive methods."""
    (mass, _), errors = local_reference(initial_um, case.VAPOR, DURATION, 0)
    if errors[0] > mass * 1e-10:
        raise ArithmeticError("reference integrators disagree")
    return np.cbrt(mass / case.MASS)


def assess():
    """Measure noiseless recovery and Gaussian observation uncertainty."""
    grid = np.linspace(0.8, 1.2, 41)
    endpoints = np.array([quadrature_endpoint(r) for r in grid])
    slopes = np.diff(endpoints) / np.diff(grid)
    if np.min(slopes) <= 0:
        raise ArithmeticError("nonmonotone observation map")
    rng = np.random.default_rng(SEED)
    records = []
    for truth in TRUTHS_UM:
        obs = independent_endpoint(truth)
        recovered = recover_radius(obs)
        estimates = []
        for start in STARTS_UM:
            result = least_squares(
                lambda x, y=obs: [(quadrature_endpoint(x[0]) - y) / SIGMA_UM],
                [start],
                bounds=(0.8, 1.2),
                gtol=1e-13,
            )
            if not result.success:
                raise ArithmeticError(result.message)
            estimates.append(float(result.x[0]))
        if max(abs(x - truth) for x in [recovered, *estimates]) > 0.002:
            raise ArithmeticError("noiseless recovery exceeds 2 nm")
        if abs(quadrature_endpoint(truth) - obs) > 0.001:
            raise ArithmeticError("forward discrepancy exceeds 1 nm")
        noisy = obs + SIGMA_UM * rng.standard_normal(256)
        fits = np.array([recover_radius(y) for y in noisy])
        # Invert the exact Gaussian observation interval using monotonicity.
        lower = np.array([recover_radius(y - 1.96 * SIGMA_UM) for y in noisy])
        upper = np.array([recover_radius(y + 1.96 * SIGMA_UM) for y in noisy])
        records.append(
            dict(
                initial_um=truth,
                final_um=float(obs),
                bracket_fit_um=recovered,
                least_squares_um=estimates,
                bias_nm=float(np.mean(fits - truth) * 1000),
                spread_nm=float(np.std(fits, ddof=1) * 1000),
                coverage=float(np.mean((lower <= truth) & (truth <= upper))),
                at_bound=float(np.mean((fits == 0.8) | (fits == 1.2))),
            )
        )
    return dict(
        noise_um=SIGMA_UM,
        seed=SEED,
        replicates=256,
        starts_um=STARTS_UM,
        slope_range=[float(slopes.min()), float(slopes.max())],
        cases=records,
    )


if __name__ == "__main__":
    import json

    print(json.dumps(assess(), indent=2))
