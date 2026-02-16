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
            concentration=np.array([[1e-6, 5e-9, 2e-10]]),  # (1, 3)
            partitioning=np.array([True, True, True]),
        )

    Multi-box CFD simulation (100 boxes)::

        cfd_data = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.zeros((100, 3)),  # kg/m^3, (100, 3)
            partitioning=np.array([True, True, True]),
        )
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from particula.gas.species import GasSpecies
    from particula.gas.vapor_pressure_strategies import VaporPressureStrategy


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
        concentration: Mass concentrations in kg/m^3.
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


def from_species(species: "GasSpecies", n_boxes: int = 1) -> GasData:
    """Convert existing GasSpecies to GasData.

    Extracts data fields from a GasSpecies instance and creates a GasData
    container with the specified number of boxes. Concentration is copied
    directly from the GasSpecies mass concentration in kg/m^3.

    Args:
        species: Existing GasSpecies instance (single or multi-species).
        n_boxes: Number of boxes to replicate concentration into.

    Returns:
        GasData with batch dimension.

    Example:
        >>> import particula as par
        >>> import numpy as np
        >>> vapor_pressure = par.gas.ConstantVaporPressureStrategy(2330)
        >>> species = par.gas.GasSpecies(
        ...     name="Water",
        ...     molar_mass=0.018,
        ...     vapor_pressure_strategy=vapor_pressure,
        ...     concentration=1e-6,  # kg/m^3
        ... )
        >>> gas_data = from_species(species)
        >>> gas_data.n_boxes
        1
    """
    # Handle single vs multi-species names
    raw_names = species.get_name()
    if isinstance(raw_names, str):
        names: list[str] = [raw_names]
    else:
        names = list(raw_names)

    # Handle single vs multi-species molar mass
    molar_mass = species.get_molar_mass()
    if isinstance(molar_mass, (int, float)):
        molar_mass = np.array([molar_mass], dtype=np.float64)
    else:
        molar_mass = np.asarray(molar_mass, dtype=np.float64)

    # Use GasSpecies mass concentration in kg/m^3 directly
    concentration = species.get_concentration()
    if isinstance(concentration, (int, float)):
        concentration = np.array([concentration], dtype=np.float64)
    else:
        concentration = np.asarray(concentration, dtype=np.float64)

    # Add batch dimension and optionally replicate to n_boxes
    if n_boxes == 1:
        concentration_2d = concentration.reshape(1, -1)
    else:
        concentration_2d = np.tile(concentration, (n_boxes, 1))

    # Handle partitioning (single bool for all species in GasSpecies)
    partitioning = species.get_partitioning()
    n_species = len(names)
    partitioning_array = np.array([partitioning] * n_species, dtype=np.bool_)

    return GasData(
        name=names,
        molar_mass=molar_mass,
        concentration=concentration_2d,
        partitioning=partitioning_array,
    )


def to_species(
    data: GasData,
    vapor_pressure_strategies: Sequence["VaporPressureStrategy"],
    box_index: int = 0,
) -> "GasSpecies":
    """Convert GasData back to GasSpecies.

    Creates a GasSpecies instance from GasData for a single box. Requires
    vapor pressure strategies to be provided since GasData does not store
    behavior. Concentration is passed through as kg/m^3 mass concentration.

    Args:
        data: GasData instance.
        vapor_pressure_strategies: List of vapor pressure strategies,
            one per species.
        box_index: Which box to extract concentration from (default 0).

    Returns:
        GasSpecies for single box.

    Raises:
        ValueError: If vapor_pressure_strategies length doesn't match n_species,
            or if GasData has mixed partitioning values.
        IndexError: If box_index is out of range.

    Example:
        >>> import particula as par
        >>> import numpy as np
        >>> gas_data = GasData(
        ...     name=["Water"],
        ...     molar_mass=np.array([0.018]),
        ...     concentration=np.array([[1e-6]]),  # kg/m^3
        ...     partitioning=np.array([True]),
        ... )
        >>> strategy = par.gas.ConstantVaporPressureStrategy(2330)
        >>> species = to_species(gas_data, [strategy])
    """
    # Import here to avoid circular import at module level
    from particula.gas.species import GasSpecies

    if len(vapor_pressure_strategies) != data.n_species:
        n_strategies = len(vapor_pressure_strategies)
        raise ValueError(
            f"vapor_pressure_strategies length {n_strategies} "
            f"doesn't match n_species {data.n_species}"
        )

    if box_index >= data.n_boxes:
        raise IndexError(
            f"box_index {box_index} out of range for n_boxes {data.n_boxes}"
        )

    # Validate uniform partitioning (GasSpecies requires single bool for all)
    unique_partitioning = np.unique(data.partitioning)
    if len(unique_partitioning) > 1:
        part_list = data.partitioning.tolist()
        raise ValueError(
            f"GasData has mixed partitioning values {part_list}, "
            "but GasSpecies requires uniform partitioning for all species"
        )
    partitioning_value = bool(unique_partitioning[0])

    concentration = data.concentration[box_index, :].reshape(1, -1)
    single_box_data = GasData(
        name=list(data.name),
        molar_mass=np.copy(data.molar_mass),
        concentration=concentration,
        partitioning=np.full(
            data.n_species, partitioning_value, dtype=np.bool_
        ),
    )

    if data.n_species == 1:
        vapor_pressure_strategy = vapor_pressure_strategies[0]
    else:
        vapor_pressure_strategy = list(vapor_pressure_strategies)

    return GasSpecies.from_data(single_box_data, vapor_pressure_strategy)
