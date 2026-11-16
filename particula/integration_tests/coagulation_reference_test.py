"""Constant-kernel coagulation against the continuous exponential solution."""

import numpy as np
from particula.dynamics import BrownianCoagulationStrategy
from particula.particles import ParticleData


def test_constant_kernel_exponential_solution(monkeypatch, record_property):
    """Compare number and mass separately with a closed-form size solution.

    For initial n(m)=N0/m0 exp(-m/m0) and constant collision kernel K,
    n(m,t)=N0/[m0*(1+tau)**2] exp[-m/(m0*(1+tau))],
    tau=K*N0*t/2. Transform using dm/dr=3*m0*r**2/r0**3.
    Integrate the native radius PDF with independent trapezoid weights.
    This smooth single-material case excludes variable-composition closure.
    """
    errors = []
    for size in (128, 256, 512):
        radius = np.geomspace(1e-9, 1e-6, size)
        scaled_mass = (radius / 1e-7) ** 3
        mass = 4 * np.pi * 1000 * radius**3 / 3
        jacobian = 3 * radius**2 / 1e-21
        initial_number, kernel, duration = 1e9, 1e-15, 2e5
        pdf = initial_number * np.exp(-scaled_mass) * jacobian
        particle = ParticleData(
            masses=mass[None, :, None],
            concentration=pdf[None].copy(),
            charge=np.zeros((1, size)),
            density=np.array([1000.0]),
            volume=np.ones(1),
            distribution_type="continuous_pdf",
            radius_grid=radius,
        )
        strategy = BrownianCoagulationStrategy("continuous_pdf")
        monkeypatch.setattr(
            strategy,
            "kernel",
            lambda *args, size=size, kernel=kernel, **kwargs: np.full(
                (size, size), kernel
            ),
        )
        # Resolve time finely enough that the spatial trend is measurable.
        for _ in range(100):
            strategy.step(particle, 298.15, 101325.0, duration / 100)
        tau = kernel * initial_number * duration / 2
        exact = (
            initial_number
            * np.exp(-scaled_mass / (1 + tau))
            * jacobian
            / (1 + tau) ** 2
        )
        width = np.diff(radius)

        def integrate(values, width=width):
            return np.sum((values[1:] + values[:-1]) * width / 2)

        number_error = abs(
            integrate(particle.concentration[0]) / (initial_number / (1 + tau))
            - 1
        )
        mass_error = abs(
            integrate(particle.concentration[0] * mass) / integrate(pdf * mass)
            - 1
        )
        spectrum_error = (
            integrate(abs(particle.concentration[0] - exact)) / initial_number
        )
        errors.append([number_error, mass_error, spectrum_error])
    record_property("number_mass_spectrum_errors_128_256_512", repr(errors))
    assert np.all(np.array(errors[-1]) < 0.002)
    assert errors[-1][2] < errors[0][2]


def test_sparse_tail_collision_births_are_nonnegative():
    """Positive kernels/populations cannot produce negative collision births."""
    from particula.dynamics.coagulation.coagulation_rate import (
        get_coagulation_gain_rate_continuous,
        get_coagulation_gain_rate_discrete,
    )

    radius = np.geomspace(1e-9, 1e-6, 128)
    pdf = 1e9 * np.exp(-((radius / 1e-7) ** 3)) * 3 * radius**2 / 1e-21
    kernel = np.full((128, 128), 1e-15)
    widths = np.diff(radius, append=2 * radius[-1] - radius[-2])
    for function, population in (
        (get_coagulation_gain_rate_continuous, pdf),
        (get_coagulation_gain_rate_discrete, pdf * widths),
    ):
        gain = function(radius, population, kernel)
        assert np.all(np.isfinite(gain))
        assert np.all(gain >= 0)
        assert np.any(gain > 0)
