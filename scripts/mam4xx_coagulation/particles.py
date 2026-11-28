"""A dry two-mode coagulation box with native Particula Brownian sampling."""

import numpy as np
from particula import Aerosol
from particula.dynamics import BrownianCoagulationStrategy, Coagulation
from particula.dynamics.coagulation.brownian_kernel import (
    get_brownian_kernel_via_system_state,
)
from particula.gas import EnvironmentData, GasData
from particula.gpu.kernels.coagulation import CoagulationSubcyclingConfig
from particula.particles import ParticleData, get_binned_number_spectrum
from scipy.special import ndtri, roots_hermitenorm

NUMBER = 2e11  # per cubic metre in the synthetic process box
DENSITY = 1770.0
TEMPERATURE = 298.15
PRESSURE = 101325.0
WIDTH = np.array([1.6, 1.8])
EDGES_NM = np.geomspace(2.0, 4000.0, 49)
TIMES = np.arange(0, 3601, 600)


def mode_population(diameter_nm, ratio, slots):
    """Match lognormal number and volume moments for each mode."""
    weights = np.array([ratio, 1.0]) / (1 + ratio)
    counts = np.rint(weights * slots).astype(int)
    np.testing.assert_allclose(counts, weights * slots, rtol=0, atol=1e-10)
    diameters = []
    for n, dg, sigma in zip(counts, (26.0, diameter_nm), WIDTH, strict=True):
        raw = dg * np.exp(np.log(sigma) * ndtri((np.arange(n) + 0.5) / n))
        third = dg**3 * np.exp(4.5 * np.log(sigma) ** 2)
        diameters.extend(raw * np.cbrt(third / np.mean(raw**3)))
    mass = np.pi / 6 * DENSITY * (np.array(diameters) * 1e-9) ** 3
    return Aerosol(
        particles=ParticleData(
            masses=mass[None, :, None],
            concentration=np.ones((1, slots)),
            charge=np.zeros((1, slots)),
            density=np.array([DENSITY]),
            volume=np.array([slots / NUMBER]),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["air"], np.array([0.029]), np.zeros((1, 1)), np.array([False])
        ),
        environment=EnvironmentData(
            np.array([TEMPERATURE]), np.array([PRESSURE]), np.zeros((1, 1))
        ),
    )


def cpu_initial_rate(diameter_nm, ratio, order):
    """Integrate the Particula Brownian kernel over both lognormal modes."""
    x, w = roots_hermitenorm(order)
    mode_weights = np.array([ratio, 1.0]) / (1 + ratio)
    weights = (mode_weights[:, None] * w / np.sqrt(2 * np.pi)).ravel()
    d = (
        np.array([26.0, diameter_nm])[:, None]
        * np.exp(x * np.log(WIDTH)[:, None])
    ).ravel() * 1e-9
    k = get_brownian_kernel_via_system_state(
        d / 2, np.pi / 6 * DENSITY * d**3, TEMPERATURE, PRESSURE
    )
    return float(0.5 * NUMBER**2 * (weights @ k @ weights))


def observe(particles):
    """Return total number and the spectrum including outside counts."""
    keep = particles.population[0] > 0
    masses = particles.masses[0, keep, 0]
    concentration = particles.slot_concentration_density[0, keep]
    diameter = 1e9 * np.cbrt(6 * masses / (np.pi * DENSITY))
    bins, low, high = get_binned_number_spectrum(
        diameter * 1e-9,
        concentration,
        EDGES_NM * 1e-9,
        include_upper_edge=False,
    )
    counts = np.r_[low, bins, high]
    np.testing.assert_allclose(counts.sum(), concentration.sum(), rtol=1e-12)
    return counts, np.sum(masses * concentration)


def trajectory(diameter_nm, ratio, slots, seed, dt):
    """Run one collision trajectory with six observation intervals."""
    initial = mode_population(diameter_nm, ratio, slots)
    process = Coagulation(BrownianCoagulationStrategy("particle_resolved"))
    spectra = []
    reference_mass = observe(initial.particles)[1]
    worst = 0.0
    with process.prepare(
        initial,
        device="cuda:0",
        time_step=dt,
        root_seed=seed,
        coagulation_subcycling=CoagulationSubcyclingConfig(
            0.01, 128, 64 * 1024**2
        ),
    ) as owner:
        for time in TIMES:
            if time:
                for _ in range(round(600 / dt)):
                    owner.step()
            spectrum, mass = observe(owner.snapshot().particles)
            worst = max(worst, abs(mass / reference_mass - 1))
            np.testing.assert_allclose(mass, reference_mass, rtol=1e-12, atol=0)
            spectra.append(spectrum)
    numbers = np.sum(spectra, axis=1)
    if np.any(np.diff(numbers) > 1e-5):
        raise ArithmeticError("Coagulation increased total particle number")
    return np.asarray(spectra), worst
