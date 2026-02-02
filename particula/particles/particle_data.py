"""Batched particle data container for multi-box CFD simulations.

This module provides the ParticleData dataclass, a simple data container
that isolates particle data from behavior. All per-particle arrays have
a batch dimension (n_boxes) built-in from the start to support multi-box
CFD simulations.

Example:
    Single-box simulation (n_boxes=1)::

        from particula.particles.particle_data import ParticleData
        import numpy as np

        data = ParticleData(
            masses=np.random.rand(1, 1000, 3) * 1e-18,  # 1000 particles
            concentration=np.ones((1, 1000)),
            charge=np.zeros((1, 1000)),
            density=np.array([1000.0, 1200.0, 800.0]),
            volume=np.array([1e-6]),  # 1 cm^3
        )

    Multi-box CFD simulation (100 boxes)::

        cfd_data = ParticleData(
            masses=np.zeros((100, 10000, 3)),
            concentration=np.ones((100, 10000)),
            charge=np.zeros((100, 10000)),
            density=np.array([1000.0, 1200.0, 800.0]),
            volume=np.ones(100) * 1e-6,
        )
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ParticleData:
    """Batched particle data container for multi-box simulations.

    Simple data container with batch dimension built-in. All per-particle
    arrays have shape (n_boxes, n_particles, ...) to support multi-box CFD.
    Single-box simulations use n_boxes=1.

    This is NOT a frozen dataclass - arrays can be updated in place for
    performance in tight simulation loops. Use copy() if immutability needed.

    Attributes:
        masses: Per-species masses in kg.
            Shape: (n_boxes, n_particles, n_species)
        concentration: Number concentration per particle.
            Shape: (n_boxes, n_particles)
            For particle-resolved: actual count (typically 1).
            For binned: number per m^3.
        charge: Particle charges (dimensionless integer counts).
            Shape: (n_boxes, n_particles)
        density: Material densities in kg/m^3.
            Shape: (n_species,) - shared across all boxes
        volume: Simulation volume per box in m^3.
            Shape: (n_boxes,)

    Raises:
        ValueError: If array shapes are inconsistent.
    """

    masses: NDArray[np.float64]
    concentration: NDArray[np.float64]
    charge: NDArray[np.float64]
    density: NDArray[np.float64]
    volume: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate array shapes are consistent."""
        # Validate masses is 3D
        if self.masses.ndim != 3:
            raise ValueError(
                "masses must be 3D (n_boxes, n_particles, n_species), "
                f"got shape {self.masses.shape}"
            )

        # Extract batch dimensions from masses
        n_boxes = self.masses.shape[0]
        n_particles = self.masses.shape[1]
        n_species = self.masses.shape[2]

        # Validate concentration shape (n_boxes, n_particles)
        expected_2d = (n_boxes, n_particles)
        if self.concentration.shape != expected_2d:
            raise ValueError(
                f"concentration shape {self.concentration.shape} doesn't match "
                f"expected {expected_2d}"
            )

        # Validate charge shape (n_boxes, n_particles)
        if self.charge.shape != expected_2d:
            raise ValueError(
                f"charge shape {self.charge.shape} doesn't match "
                f"expected {expected_2d}"
            )

        # Validate volume shape (n_boxes,)
        expected_1d = (n_boxes,)
        if self.volume.shape != expected_1d:
            raise ValueError(
                f"volume shape {self.volume.shape} doesn't match "
                f"expected {expected_1d}"
            )

        # Validate density is 1D (n_species,)
        if self.density.ndim != 1:
            raise ValueError(
                f"density must be 1D (n_species,), "
                f"got shape {self.density.shape}"
            )

        # Validate n_species matches between masses and density
        if self.density.shape[0] != n_species:
            raise ValueError(
                f"n_species mismatch: masses has {n_species} species, "
                f"but density has {self.density.shape[0]} species"
            )
