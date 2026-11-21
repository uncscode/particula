"""Independent analytical checks for the MAM4xx study's two solvers."""

import numpy as np
from scripts.mam4xx_coagulation.attraction_particles import trajectory
from scripts.mam4xx_coagulation.modal_attraction import (
    HOURS,
    distribution,
    solve_modes,
)


def test_constant_kernel_modal_number_and_volume():
    """All mode pairs together satisfy the exact constant-kernel box law."""
    modes = distribution("marine", "host", True)
    edges = np.geomspace(0.01, 1e7, 110)
    k = 7e-8
    kernel = np.full((109, 109), k)
    run = solve_modes(modes, kernel, edges, np.zeros_like(kernel, dtype=bool))
    n0 = modes[:, 2].sum()
    expected = n0 / (1 + 0.5 * k * n0 * HOURS * 3600)
    np.testing.assert_allclose(
        np.sum(run["mode_n"], axis=1), expected, rtol=1e-7
    )
    v = np.sum(run["mode_v"], axis=1)
    np.testing.assert_allclose(v / v[0], 1.0, rtol=2e-12)


def test_two_distinct_particles_have_one_collision_clock():
    """A known exponential waiting time tests pair weighting and self exclusion."""
    edges = np.geomspace(1, 1e3, 25)
    rate = np.log(2) / 3600
    kernel = np.full((24, 24), rate)
    # Two 80 nm particles become one >100 nm particle after their only event.
    # In a 2 cm³ box their rate is K/2 and survival at one hour is 1/sqrt(2).
    outputs = [
        trajectory([(0.08, 1.0, 1.0)], edges, kernel, 2.0, s)[0]
        for s in range(512)
    ]
    below = np.asarray(outputs)[:, :, 0]
    above = np.asarray(outputs)[:, :, 1]
    np.testing.assert_allclose(below + 2 * above, 1)
    expected = np.exp(-rate * 3600 * HOURS / 2)
    sampling = np.sqrt(expected * (1 - expected) / len(outputs))
    np.testing.assert_array_less(
        abs(below.mean(axis=0) - expected), 4 * sampling + 1e-9
    )


def test_neutral_large_ratio_against_adaptive_orbit_integration():
    """Check the tail outside native capture support with continuous orbits."""
    from particula.dynamics.coagulation._pair_potential import hamaker_energy
    from particula.util.constants import BOLTZMANN_CONSTANT
    from scipy.integrate import quad
    from scipy.optimize import minimize_scalar
    from scripts.mam4xx_coagulation.modal_attraction import HAMAKER, TEMP
    from scripts.mam4xx_coagulation.neutral_geometry import enhancements

    attraction = HAMAKER / (BOLTZMANN_CONSTANT * TEMP)
    for ratio in (32, 128, 1024):
        fraction = 1 / (1 + ratio)

        def energy(x, fraction=fraction):
            return attraction * hamaker_energy(x, fraction)

        resistance, _ = quad(lambda x: np.exp(energy(x)), 0, 1, epsabs=3e-9)

        def flux(t, energy=energy):
            def limit(log_gap):
                x = -np.expm1(-log_gap)
                return (t - energy(x)) / x**2

            orbit = minimize_scalar(limit, bounds=(1e-8, 32), method="bounded")
            return np.exp(-t) * orbit.fun

        ballistic, _ = quad(flux, 0, 50, epsabs=3e-8)
        np.testing.assert_allclose(
            enhancements(ratio, attraction),
            [1 / resistance, ballistic],
            rtol=5e-4,
        )
