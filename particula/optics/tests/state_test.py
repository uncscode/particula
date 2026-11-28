"""Native optical observations respect all supported population measures."""

import numpy as np
import numpy.testing as npt
from particula.optics import (
    OpticalConfig,
    get_mie_efficiencies,
    population_optics,
)
from particula.particles import ParticleData
from particula.particles._population import number_density_weights


def test_pdf_uses_radius_grid_as_quadrature_and_optical_size():
    """Compare absorption and extinction with direct PDF integration."""
    grid = np.array([70e-9, 83e-9, 145e-9, 317e-9])
    density = np.array([[5e15, 9e15, 2e15, 1e15]])
    particles = ParticleData(
        # Hold the composition fixed independently of the PDF nodes.
        masses=np.full((1, len(grid), 1), 2e-17),
        density=np.array([1110.0]),
        concentration=density,
        charge=np.zeros_like(density),
        volume=np.ones(1),
        distribution_type="continuous_pdf",
        radius_grid=grid,
    )
    wavelength = 500e-9
    index = 1.65 + 0.04j
    config = OpticalConfig([wavelength], particle_index=index)
    result = population_optics(particles, config)
    q = get_mie_efficiencies(grid, wavelength, index)
    for name in ("extinction", "absorption"):
        density_integrand = np.pi * grid**2 * getattr(q, name) * density
        expected = np.trapezoid(density_integrand, grid, axis=-1)
        npt.assert_allclose(getattr(result, name)[:, 0], expected, rtol=3e-14)


def test_resolved_boxes_and_empty_population_match_discrete_optics():
    """Matched densities give the same coefficients across physical volumes."""
    radius = np.array([60e-9, 120e-9])
    mass = (4 * np.pi / 3 * 1000 * radius**3)[:, None]
    particles = ParticleData(
        masses=np.stack([mass, mass, np.zeros_like(mass)]),
        concentration=np.array([[1.0, 2.0], [16.0, 32.0], [0.0, 0.0]]),
        charge=np.zeros((3, 2)),
        density=np.array([1000.0]),
        volume=np.array([0.25, 4.0, 1.0]),
        distribution_type="particle_resolved",
    )
    weights = number_density_weights(particles)
    npt.assert_array_equal(weights, [[4, 8], [4, 8], [0, 0]])
    assert not np.shares_memory(weights, particles.concentration)
    npt.assert_allclose(
        particles.species_mass_inventory[1],
        16 * particles.species_mass_inventory[0],
        rtol=1e-14,
        atol=0,
    )
    config = OpticalConfig([500e-9, 650e-9], particle_index=1.65 + 0.04j)
    resolved = population_optics(particles, config)
    discrete = ParticleData(
        masses=mass[None],
        concentration=np.array([[4.0, 8.0]]),
        charge=np.zeros((1, 2)),
        density=particles.density,
        volume=np.ones(1),
        distribution_type="discrete",
    )
    binned = population_optics(discrete, config)
    for column, wavelength in enumerate(config.wavelength):
        q = get_mie_efficiencies(radius, wavelength, 1.65 + 0.04j)
        for name in ("absorption", "extinction"):
            expected = np.sum(np.pi * radius**2 * getattr(q, name) * [4, 8])
            npt.assert_allclose(
                getattr(resolved, name)[:, column],
                [expected, expected, 0],
                rtol=3e-14,
                atol=0,
            )
            npt.assert_allclose(
                getattr(binned, name)[0, column], expected, rtol=3e-14, atol=0
            )
    assert not resolved.extinction_valid[2].any()
    assert not resolved.scattering_valid[2].any()
