"""Adaptive pair-capture reference for validating aerosol collision kernels.

All lengths are scaled by contact distance and energies by k_B T.
The solver averages the ballistic capture rate over relative kinetic energy.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


def _extremum(function, nodes, values, maximize=False):
    """Refine every sampled local extremum and both end intervals."""
    sign = -1 if maximize else 1
    costs = sign * np.asarray(values)
    indices = (
        np.flatnonzero((costs[1:-1] <= costs[:-2]) & (costs[1:-1] <= costs[2:]))
        + 1
    )
    intervals = {(0, 1), (len(nodes) - 2, len(nodes) - 1)}
    intervals.update((i - 1, i + 1) for i in indices)
    best = np.min(costs)
    for i, j in intervals:
        result = minimize_scalar(
            lambda x: sign * function(x),
            bounds=(nodes[i], nodes[j]),
            method="bounded",
            options={"xatol": 2e-13},
        )
        best = min(best, result.fun)
    return sign * best


def reference_capture(potential, *, resolution=128, tolerance=2e-7):
    """Return continuum and ballistic rate ratios for a radial potential.

    potential accepts scalar or array x=contact distance/centre distance
    and returns U/kT, including its limiting values at infinity and contact.
    Adaptive quadrature and orbit minimization are independent of the
    production quadrature. Increase resolution and tighten tolerance to
    measure convergence for a specified pair.
    """
    phase = np.linspace(0, np.pi, resolution + 1)
    x = np.sin(phase / 2) ** 2
    energy = potential(x)
    barrier = _extremum(potential, x, energy, maximize=True)
    # Shift the exponent to resolve large repulsive barriers.
    resistance, _ = quad(
        lambda y: np.exp(potential(y) - barrier),
        0,
        1,
        epsabs=tolerance * 0.01,
        epsrel=tolerance,
        limit=200,
    )
    continuum = np.exp(-barrier) / resistance

    def rate_integrand(excess):
        kinetic = barrier + excess
        cost = (kinetic - energy[1:]) / x[1:] ** 2
        minimum = _extremum(
            lambda y: (kinetic - potential(y)) / y**2, x[1:], cost
        )
        return np.exp(-excess) * max(0.0, minimum)

    ballistic, _ = quad(
        rate_integrand,
        0,
        np.inf,
        epsabs=tolerance * 0.1,
        epsrel=tolerance,
        limit=150,
    )
    return continuum, np.exp(-barrier) * ballistic
