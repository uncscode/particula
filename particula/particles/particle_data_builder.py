"""Builder for creating validated :class:`ParticleData` instances.

Provides a fluent interface to set particle fields with optional unit
conversion, automatic batch dimension handling, and a zero-initialization
path for cases where masses are not precomputed.

Examples:
    Single-box with explicit masses::

        import numpy as np
        from particula.particles import ParticleDataBuilder

        data = (
            ParticleDataBuilder()
            .set_masses(np.array([[1e-18, 2e-18]]), units="kg")
            .set_density(np.array([1000.0, 1200.0]), units="kg/m^3")
            .set_concentration(np.array([1.0]), units="1/m^3")
            .set_charge(np.array([0.0]))
            .build()
        )

    Zero-init with counts only::

        data = (
            ParticleDataBuilder()
            .set_n_boxes(2)
            .set_n_particles(3)
            .set_n_species(1)
            .set_density(np.array([1500.0]), units="kg/m^3")
            .build()
        )
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from particula.particles.particle_data import ParticleData
from particula.util.convert_units import get_unit_conversion


class ParticleDataBuilder:
    """Fluent builder that prepares arrays for ``ParticleData``.

    Each setter performs a single unit conversion and ensures ``float64``
    dtype. Batch dimensions are inserted automatically when 1D (for
    concentration/charge) or 2D (for masses). A zero-initialization path is
    available when counts are supplied instead of masses.
    """

    def __init__(self) -> None:
        """Initialize empty builder state."""
        self._masses: Optional[NDArray[np.float64]] = None
        self._concentration: Optional[NDArray[np.float64]] = None
        self._charge: Optional[NDArray[np.float64]] = None
        self._density: Optional[NDArray[np.float64]] = None
        self._volume: Optional[NDArray[np.float64]] = None
        self._n_boxes: Optional[int] = None
        self._n_particles: Optional[int] = None
        self._n_species: Optional[int] = None

    def set_masses(
        self, masses: NDArray[np.float64], units: str = "kg"
    ) -> "ParticleDataBuilder":
        """Set per-species masses with optional unit conversion.

        Args:
            masses: Mass array shaped (n_boxes, n_particles, n_species) or
                (n_particles, n_species).
            units: Units of the provided masses. Supported: kg, g, ug, ng.

        Returns:
            Self for fluent chaining.

        Raises:
            ValueError: When masses are not 2D or 3D.
        """
        masses_array = np.asarray(masses, dtype=np.float64)
        if masses_array.ndim == 2:
            masses_array = np.expand_dims(masses_array, axis=0)
        elif masses_array.ndim != 3:
            raise ValueError("masses must be 2D or 3D")

        if units != "kg":
            masses_array = masses_array * get_unit_conversion(units, "kg")

        self._masses = masses_array
        return self

    def set_concentration(
        self, concentration: NDArray[np.float64], units: str = "1/m^3"
    ) -> "ParticleDataBuilder":
        """Set number concentration with optional unit conversion.

        Args:
            concentration: Concentration array shaped (n_boxes, n_particles)
                or (n_particles,).
            units: Units of the provided concentration. Supported: 1/m^3,
                1/cm^3.

        Returns:
            Self for fluent chaining.

        Raises:
            ValueError: When concentration is not 1D or 2D.
        """
        concentration_array = np.asarray(concentration, dtype=np.float64)
        if concentration_array.ndim == 1:
            concentration_array = np.expand_dims(concentration_array, axis=0)
        elif concentration_array.ndim != 2:
            raise ValueError("concentration must be 1D or 2D")

        if units != "1/m^3":
            concentration_array = concentration_array * get_unit_conversion(
                units, "1/m^3"
            )

        self._concentration = concentration_array
        return self

    def set_charge(self, charge: NDArray[np.float64]) -> "ParticleDataBuilder":
        """Set particle charge (unitless) with batch handling.

        Args:
            charge: Charge array shaped (n_boxes, n_particles) or
                (n_particles,).

        Returns:
            Self for fluent chaining.

        Raises:
            ValueError: When charge is not 1D or 2D.
        """
        charge_array = np.asarray(charge, dtype=np.float64)
        if charge_array.ndim == 1:
            charge_array = np.expand_dims(charge_array, axis=0)
        elif charge_array.ndim != 2:
            raise ValueError("charge must be 1D or 2D")

        self._charge = charge_array
        return self

    def set_density(
        self, density: NDArray[np.float64], units: str = "kg/m^3"
    ) -> "ParticleDataBuilder":
        """Set species densities with optional unit conversion.

        Args:
            density: Density array shaped (n_species,).
            units: Units of the provided density. Supported: kg/m^3, g/cm^3.

        Returns:
            Self for fluent chaining.

        Raises:
            ValueError: When density is not 1D.
        """
        density_array = np.asarray(density, dtype=np.float64)
        if density_array.ndim != 1:
            raise ValueError("density must be 1D")

        if units != "kg/m^3":
            density_array = density_array * get_unit_conversion(units, "kg/m^3")

        self._density = density_array
        return self

    def set_volume(
        self, volume: NDArray[np.float64], units: str = "m^3"
    ) -> "ParticleDataBuilder":
        """Set per-box volume with optional unit conversion.

        Args:
            volume: Volume scalar or array shaped (n_boxes,).
            units: Units of the provided volume. Supported: m^3, cm^3, L.

        Returns:
            Self for fluent chaining.

        Raises:
            ValueError: When volume is neither scalar nor 1D.
        """
        volume_array = np.asarray(volume, dtype=np.float64)
        if volume_array.ndim == 0:
            volume_array = volume_array.reshape(1)
        elif volume_array.ndim != 1:
            raise ValueError("volume must be scalar or 1D")

        if units != "m^3":
            volume_array = volume_array * get_unit_conversion(units, "m^3")

        self._volume = volume_array
        return self

    def set_n_boxes(self, n_boxes: int) -> "ParticleDataBuilder":
        """Set number of boxes for zero-init or shape validation."""
        self._n_boxes = int(n_boxes)
        return self

    def set_n_particles(self, n_particles: int) -> "ParticleDataBuilder":
        """Set number of particles for zero-init or shape validation."""
        self._n_particles = int(n_particles)
        return self

    def set_n_species(self, n_species: int) -> "ParticleDataBuilder":
        """Set number of species for zero-init or shape validation."""
        self._n_species = int(n_species)
        return self

    def _infer_counts_from_masses(self) -> tuple[int, int, int]:
        """Derive counts from masses when present."""
        if self._masses is None:
            raise ValueError("Set masses before inferring counts")
        n_boxes, n_particles, n_species = self._masses.shape
        self._n_boxes = self._n_boxes or n_boxes
        self._n_particles = self._n_particles or n_particles
        self._n_species = self._n_species or n_species
        return n_boxes, n_particles, n_species

    def _broadcast_if_needed(
        self,
        array: NDArray[np.float64],
        target_shape: tuple[int, ...],
        label: str,
    ) -> NDArray[np.float64]:
        """Broadcast array to target shape when compatible."""
        if array.shape == target_shape:
            return array
        if len(target_shape) == 2 and array.shape == (1, target_shape[1]):
            return np.broadcast_to(
                array,
                target_shape,
            )
        if len(target_shape) == 1 and array.shape == (1,):
            return np.broadcast_to(
                array,
                target_shape,
            )
        raise ValueError(
            f"{label} shape {array.shape} does not match expected "
            f"{target_shape}"
        )

    def _ensure_counts(self) -> None:
        """Ensure counts are available for zero-init path."""
        if self._n_boxes is None or self._n_particles is None:
            raise ValueError(
                "Set masses or provide n_boxes and n_particles for zero-init"
            )
        if self._n_species is None:
            raise ValueError(
                "Set masses or provide n_species for zero-init masses"
            )

    def _get_counts(self) -> tuple[int, int, int]:
        """Return the integer counts currently tracked."""
        if (
            self._n_boxes is None
            or self._n_particles is None
            or self._n_species is None
        ):
            raise ValueError("Counts are required before building ParticleData")
        return self._n_boxes, self._n_particles, self._n_species

    def build(self) -> ParticleData:
        """Construct a ``ParticleData`` instance with validation.

        Returns:
            A validated ``ParticleData`` object.

        Raises:
            ValueError: When required fields are missing or shapes mismatch.
        """
        if self._masses is None:
            self._ensure_counts()
            n_boxes, n_particles, n_species = self._get_counts()
            masses = np.zeros(
                (n_boxes, n_particles, n_species),
                dtype=np.float64,
            )
        else:
            self._infer_counts_from_masses()
            n_boxes, n_particles, n_species = self._get_counts()
            masses = self._masses
            if masses.shape != (n_boxes, n_particles, n_species):
                raise ValueError(
                    "masses shape mismatch for provided counts and batches"
                )

        if self._density is None:
            raise ValueError("density is required")

        if self._density.shape[0] != n_species:
            raise ValueError(
                "density length must match number of species from masses"
            )
        density = self._density

        conc = self._concentration
        if conc is None:
            conc = np.ones((n_boxes, n_particles))
        else:
            conc = self._broadcast_if_needed(
                conc,
                (n_boxes, n_particles),
                label="concentration",
            )

        charge = self._charge
        if charge is None:
            charge = np.zeros((n_boxes, n_particles))
        else:
            charge = self._broadcast_if_needed(
                charge, (n_boxes, n_particles), label="charge"
            )

        volume = self._volume
        if volume is None:
            volume = np.ones(n_boxes)
        else:
            volume = self._broadcast_if_needed(
                volume, (n_boxes,), label="volume"
            )

        return ParticleData(
            masses=masses,
            concentration=conc,
            charge=charge,
            density=density,
            volume=volume,
        )
