"""Provide a batched particle data container for multi-box CFD simulations.

ParticleData isolates per-particle arrays from behavior while embedding the
batch dimension required for CFD experiments spanning multiple boxes.

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

    @property
    def n_boxes(self) -> int:
        """Number of simulation boxes.

        Returns:
            The size of the batch dimension (n_boxes).
        """
        return self.masses.shape[0]

    @property
    def n_particles(self) -> int:
        """Number of particles per box.

        Returns:
            The number of particles (n_particles).
        """
        return self.masses.shape[1]

    @property
    def n_species(self) -> int:
        """Number of chemical species.

        Returns:
            The number of species (n_species).
        """
        return self.masses.shape[2]

    @property
    def radii(self) -> NDArray[np.float64]:
        """Particle radii derived from mass and density.

        Returns:
            Radii in meters with shape (n_boxes, n_particles).
        """
        volumes_per_species = self.masses / self.density
        total_volume = np.sum(volumes_per_species, axis=-1)
        # r = (3V / 4π)^(1/3) for a sphere
        return np.cbrt(3.0 * total_volume / (4.0 * np.pi))

    @property
    def total_mass(self) -> NDArray[np.float64]:
        """Total mass per particle.

        Returns:
            Total mass in kilograms with shape (n_boxes, n_particles).
        """
        return np.sum(self.masses, axis=-1)

    @property
    def effective_density(self) -> NDArray[np.float64]:
        """Mass-weighted effective density per particle.

        Returns:
            Effective density in kg/m^3 with shape (n_boxes, n_particles).
        """
        # Match ParticleRepresentation.get_effective_density():
        # sum_i(m_i * rho_i) / sum_i(m_i)
        mass_weighted_density_sum = np.sum(self.masses * self.density, axis=-1)
        total_mass = self.total_mass
        return np.divide(
            mass_weighted_density_sum,
            total_mass,
            where=total_mass > 0,
            out=np.zeros_like(total_mass),
        )

    @property
    def mass_fractions(self) -> NDArray[np.float64]:
        """Mass fractions per species for each particle.

        Returns:
            Mass fractions with shape (n_boxes, n_particles, n_species).
        """
        total = self.total_mass[..., np.newaxis]
        return np.divide(
            self.masses,
            total,
            where=total > 0,
            out=np.zeros_like(self.masses),
        )

    def copy(self) -> "ParticleData":
        """Create a deep copy of this ParticleData.

        Returns:
            A new ParticleData instance with copied arrays.
        """
        return ParticleData(
            masses=np.copy(self.masses),
            concentration=np.copy(self.concentration),
            charge=np.copy(self.charge),
            density=np.copy(self.density),
            volume=np.copy(self.volume),
        )
