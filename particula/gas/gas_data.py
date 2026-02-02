"""Provide a batched gas data container for multi-box CFD simulations.

GasData isolates gas species arrays from behavior (vapor pressure strategies)
while embedding the batch dimension required for CFD experiments spanning
multiple boxes.

Example:
    Single-box simulation (n_boxes=1)::

        from particula.gas.gas_data import GasData
        import numpy as np

        data = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.array([[1e15, 1e12, 1e10]]),  # (1, 3)
            partitioning=np.array([True, True, True]),
        )

    Multi-box CFD simulation (100 boxes)::

        cfd_data = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.zeros((100, 3)),  # (100, 3)
            partitioning=np.array([True, True, True]),
        )
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class GasData:
    """Batched gas species data container for multi-box simulations.

    Simple data container with batch dimension built-in. Concentration
    arrays have shape (n_boxes, n_species) to support multi-box CFD.
    Single-box simulations use n_boxes=1.

    This is NOT a frozen dataclass - concentrations can be updated in place
    for performance. Use copy() if immutability is needed.

    Attributes:
        name: Species names. List of strings, length n_species.
        molar_mass: Molar masses in kg/mol. Shape: (n_species,)
        concentration: Number concentrations in molecules/m^3.
            Shape: (n_boxes, n_species)
        partitioning: Whether each species can partition to particles.
            Shape: (n_species,) - shared across boxes

    Raises:
        ValueError: If array shapes are inconsistent or species list is empty.
    """

    name: list[str]
    molar_mass: NDArray[np.float64]
    concentration: NDArray[np.float64]
    partitioning: NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Validate array shapes are consistent and enforce boolean mask."""
        # Reject empty species set to avoid ambiguous shapes
        if len(self.name) == 0:
            raise ValueError("name must contain at least one species")

        # Normalize arrays to expected dtypes
        self.molar_mass = np.asarray(self.molar_mass, dtype=np.float64)
        self.concentration = np.asarray(self.concentration, dtype=np.float64)

        # Ensure partitioning is boolean or raise if conversion fails
        try:
            self.partitioning = np.asarray(self.partitioning, dtype=np.bool_)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "partitioning must be boolean convertible"
            ) from exc

        n_species = len(self.name)

        # concentration must be 2D (n_boxes, n_species)
        if self.concentration.ndim != 2:
            raise ValueError(
                "concentration must be 2D (n_boxes, n_species), "
                f"got ndim={self.concentration.ndim}"
            )

        # concentration width must match number of species
        if self.concentration.shape[1] != n_species:
            raise ValueError(
                "concentration n_species dimension does not match name list: "
                f"got {self.concentration.shape[1]}, expected {n_species}"
            )

        # molar_mass shape must align with n_species
        if self.molar_mass.shape != (n_species,):
            raise ValueError(
                "molar_mass shape does not match n_species: "
                f"got {self.molar_mass.shape}, expected ({n_species},)"
            )

        # partitioning must be 1D boolean with n_species elements
        if self.partitioning.shape != (n_species,):
            raise ValueError(
                "partitioning shape does not match n_species: "
                f"got {self.partitioning.shape}, expected ({n_species},)"
            )

    @property
    def n_boxes(self) -> int:
        """Number of simulation boxes (batch dimension)."""
        return int(self.concentration.shape[0])

    @property
    def n_species(self) -> int:
        """Number of gas species."""
        return len(self.name)

    def copy(self) -> "GasData":
        """Create a deep copy of this GasData."""
        return GasData(
            name=list(self.name),
            molar_mass=np.copy(self.molar_mass),
            concentration=np.copy(self.concentration),
            partitioning=np.copy(self.partitioning),
        )
