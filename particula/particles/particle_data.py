"""Provide a batched particle data container for multi-box CFD simulations.

ParticleData isolates per-particle arrays from behavior while embedding the
batch dimension required for CFD experiments spanning multiple boxes.

Example:
    Single-box simulation (n_boxes=1)::

        from particula.particles.particle_data import ParticleData
        import numpy as np

        data = ParticleData(
            masses=np.random.rand(1, 1000, 3) * 1e-18,  # 1000 particles
            population=np.ones((1, 1000)),
            charge=np.zeros((1, 1000)),
            density=np.array([1000.0, 1200.0, 800.0]),
            volume=np.array([1e-6]),  # 1 cm^3
            distribution_type="particle_resolved",
        )

    Multi-box CFD simulation (100 boxes)::

        cfd_data = ParticleData(
            masses=np.zeros((100, 10000, 3)),
            population=np.ones((100, 10000)),
            charge=np.zeros((100, 10000)),
            density=np.array([1000.0, 1200.0, 800.0]),
            volume=np.ones(100) * 1e-6,
            distribution_type="particle_resolved",
        )
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from typing_extensions import deprecated

from particula.particles._population import radius_node_widths, slot_density


@dataclass(init=False)
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
        population: Extensive weights of shape (n_boxes, n_particles).
            Bins and resolved slots store counts. Radius PDFs store dN/dr
            in m^-1.
        charge: Particle charges (dimensionless integer counts).
            Shape: (n_boxes, n_particles)
        density: Material densities in kg/m^3.
            Shape: (n_species,) - shared across all boxes
        volume: Simulation volume per box in m^3, shape (n_boxes,).
            All tagged populations use finite positive air volumes.
        distribution_type: Explicit storage interpretation, or ``None`` for
            untagged legacy carriers. New bulk helpers reject untagged data.
        radius_grid: Explicit PDF radius nodes in metres, shape
            (n_particles,); absent for other kinds.

    Raises:
        ValueError: If array shapes are inconsistent.
    """

    masses: NDArray[np.float64]
    population: NDArray[np.float64]
    charge: NDArray[np.float64]
    density: NDArray[np.float64]
    volume: NDArray[np.float64]
    distribution_type: str | None = None
    radius_grid: NDArray[np.float64] | None = None

    def __init__(
        self,
        masses: NDArray[np.float64],
        concentration: NDArray[np.float64] | None = None,
        charge: NDArray[np.float64] | None = None,
        density: NDArray[np.float64] | None = None,
        volume: NDArray[np.float64] | None = None,
        distribution_type: str | None = None,
        radius_grid: NDArray[np.float64] | None = None,
        *,
        population: NDArray[np.float64] | None = None,
    ) -> None:
        """Bind extensive storage or the compatible 0.3 concentration array."""
        if charge is None or density is None or volume is None:
            raise TypeError("charge, density and volume are required")
        self.masses = masses
        self.charge = charge
        self.density = density
        self.volume = volume
        self.distribution_type = distribution_type
        self.radius_grid = radius_grid
        if population is None:
            if concentration is None:
                raise TypeError("population is required")
            self._check_legacy_population()
            population = concentration
        elif concentration is not None:
            raise ValueError("supply either population or concentration")
        self.population = population
        self.__post_init__()
        if distribution_type is not None and (
            np.any(volume <= 0) or not np.all(np.isfinite(volume))
        ):
            raise ValueError(
                "represented air volume must be positive and finite"
            )

    def _check_legacy_population(self) -> None:
        if self.distribution_type in ("discrete", "continuous_pdf") and np.any(
            self.volume != 1
        ):
            raise ValueError(
                "legacy concentration requires binned volume exactly 1 m^3; "
                "use population=density * volume[:, None] for storage or "
                "slot_concentration_density to read density"
            )

    @property
    @deprecated("Use population or slot_concentration_density", category=None)
    def concentration(self) -> NDArray[np.float64]:
        """Deprecated writable 0.3 alias for legacy-compatible volumes."""
        self._check_legacy_population()
        return self.population

    @concentration.setter
    @deprecated("Use population or slot_concentration_density", category=None)
    def concentration(self, value: NDArray[np.float64]) -> None:
        """Replace the legacy-compatible writable population buffer."""
        self._check_legacy_population()
        self.population = value

    def __post_init__(self) -> None:
        """Validate raw array shapes without interpreting distribution kind.

        Raises:
            ValueError: If raw field shapes or species dimensions disagree.
        """
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
        if self.population.shape != expected_2d:
            raise ValueError(
                f"concentration shape {self.population.shape} doesn't match "
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

        # Broadcast scalar density to match n_species, then validate
        if self.density.shape[0] == 1 and n_species > 1:
            self.density = np.broadcast_to(self.density, (n_species,)).copy()
        elif self.density.shape[0] == 0 and n_species > 0:
            # Empty density with non-zero species (e.g., zero particles):
            # fill with zeros so shapes are consistent.
            self.density = np.zeros(n_species, dtype=self.density.dtype)
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

    def validate_representation(self) -> None:  # noqa: C901
        """Validate storage for its declared distribution interpretation.

        Call after directly changing writable arrays. This read-only check
        never repairs state; derived population helpers call it on every read.
        Each box requires positive represented volume. PDF nodes must form an
        increasing, positive radius grid with at least two nodes.

        Raises:
            ValueError: If kind, grid, volume, shape, or values are invalid.
        """
        if self.distribution_type not in (
            "discrete",
            "continuous_pdf",
            "particle_resolved",
        ):
            raise ValueError("explicit distribution_type is required")
        if (
            not isinstance(self.masses, np.ndarray)
            or self.masses.ndim != 3
            or self.masses.dtype != np.float64
        ):
            raise ValueError("masses must be a rank-3 float64 array")
        boxes, slots, species = self.masses.shape
        fields = (
            (self.population, (boxes, slots)),
            (self.charge, (boxes, slots)),
            (self.density, (species,)),
            (self.volume, (boxes,)),
        )
        if any(
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != np.float64
            for value, shape in fields
        ):
            raise ValueError("particle data arrays must have float64 schema")
        if (
            not np.all(np.isfinite(self.masses))
            or np.any(self.masses < 0)
            or not np.all(np.isfinite(self.population))
            or np.any(self.population < 0)
            or not np.all(np.isfinite(self.charge))
            or not np.all(np.isfinite(self.density))
            or np.any(self.density <= 0)
            or not np.all(np.isfinite(self.volume))
        ):
            raise ValueError("particle data must have finite physical values")
        if np.any(self.volume <= 0):
            raise ValueError("represented air volume must be positive")
        if self.distribution_type == "continuous_pdf":
            grid = self.radius_grid
            if (
                not isinstance(grid, np.ndarray)
                or grid.shape != (slots,)
                or grid.dtype != np.float64
                or slots < 2
                or not np.all(np.isfinite(grid))
                or np.any(grid <= 0)
                or not np.all(np.diff(grid) > 0)
            ):
                raise ValueError(
                    "PDF radius grid must be positive and increasing"
                )
        elif self.radius_grid is not None:
            raise ValueError("radius_grid is only valid for continuous_pdf")

    @property
    def slot_concentration_density(self) -> NDArray[np.float64]:
        """Return fresh per-slot number density or radius-PDF density.

        Discrete and resolved slots have units m^-3; PDF nodes have units
        m^-4. The result has shape (n_boxes, n_particles) and is detached
        from the raw population array.

        Returns:
            Per-slot density on the declared representation's grid.

        Raises:
            ValueError: If the kind, grid, volume, or raw data is invalid.
        """
        self.validate_representation()
        return slot_density(self)

    @property
    def number_density(self) -> NDArray[np.float64]:
        """Return fresh per-box number density in m^-3.

        Sum discrete/resolved slot densities; integrate PDF node density
        over the explicit radius grid using trapezoidal quadrature.

        Returns:
            Number density with shape (n_boxes,).

        Raises:
            ValueError: If the representation fails read-only validation.
        """
        slots = self.slot_concentration_density
        if self.distribution_type == "continuous_pdf":
            return np.asarray(
                np.trapezoid(slots, x=self.radius_grid, axis=1),
                dtype=np.float64,
            )
        return np.sum(slots, axis=1)

    @property
    def species_mass_density(self) -> NDArray[np.float64]:
        """Return fresh per-species particle mass density in kg/m^3.

        Weight species masses by raw slot density. For PDFs, integrate
        the product of mass and dN/dr on the explicit radius grid using
        trapezoidal node weights, rather than summing PDF nodes.

        Returns:
            Particle mass density with shape (n_boxes, n_species).

        Raises:
            ValueError: If the representation fails read-only validation.
        """
        slots = self.slot_concentration_density
        if self.distribution_type == "continuous_pdf":
            # Trapezoidal node weights integrate the product without a
            # (boxes, slots, species) weighted-product scratch array.
            grid = self.radius_grid
            if grid is None:  # Defensive narrowing after validation.
                raise ValueError("PDF radius grid is required")
            weights = radius_node_widths(grid)
            return np.einsum("n,bn,bns->bs", weights, slots, self.masses)
        return np.einsum("bn,bns->bs", slots, self.masses)

    @property
    def species_mass_inventory(self) -> NDArray[np.float64]:
        """Return fresh extensive particle mass per species in kg.

        Returns:
            Mass inventory with shape (n_boxes, n_species).

        Raises:
            ValueError: If the representation fails read-only validation.
        """
        self.validate_representation()
        if self.radius_grid is not None:
            widths = radius_node_widths(self.radius_grid)
            return np.einsum(
                "n,bn,bns->bs", widths, self.population, self.masses
            )
        return np.einsum("bn,bns->bs", self.population, self.masses)

    @property
    def effective_density(self) -> NDArray[np.float64]:
        """Mixture density from total mass divided by additive material volume.

        Returns:
            Effective density in kg/m^3 with shape (n_boxes, n_particles).
        """
        total_volume = np.sum(self.masses / self.density, axis=-1)
        total_mass = self.total_mass
        return np.divide(
            total_mass,
            total_volume,
            where=total_volume > 0,
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
            A new ParticleData with detached raw arrays and radius grid,
            preserving the explicit distribution type when present.
        """
        return ParticleData(
            masses=np.copy(self.masses),
            population=np.copy(self.population),
            charge=np.copy(self.charge),
            density=np.copy(self.density),
            volume=np.copy(self.volume),
            distribution_type=self.distribution_type,
            radius_grid=(
                None if self.radius_grid is None else np.copy(self.radius_grid)
            ),
        )
