"""Native optical observations respect all supported population measures."""

import numpy as np
import numpy.testing as npt
from particula.optics import (
    OpticalConfig,
    get_mie_efficiencies,
    population_optics,
)
from particula.particles import ParticleData


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
