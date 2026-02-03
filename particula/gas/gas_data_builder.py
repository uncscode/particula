"""Builder for creating validated :class:`GasData` instances.

Provides a fluent interface to set gas species fields with optional unit
conversion and automatic batch dimension handling.

Examples:
    Single-box with direct values::

        from particula.gas import GasDataBuilder
        import numpy as np

        gas = (
            GasDataBuilder()
            .set_names(["Water", "Ammonia"])
            .set_molar_mass([18, 17], units="g/mol")
            .set_concentration([1e15, 1e12], units="1/m^3")
            .set_partitioning([True, True])
            .build()
        )

    Multi-box with broadcasting::

        gas = (
            GasDataBuilder()
            .set_n_boxes(100)
            .set_names(["Water", "Ammonia"])
            .set_molar_mass([0.018, 0.017], units="kg/mol")
            .set_concentration([1e15, 1e12])  # Broadcast to 100 boxes
            .set_partitioning([True, True])
            .build()
        )
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from particula.gas.gas_data import GasData
from particula.util.convert_units import get_unit_conversion


class GasDataBuilder:
    """Fluent builder that prepares arrays for ``GasData``.

    Each setter performs unit conversion and ensures correct dtypes.
    Batch dimensions are inserted automatically when 1D concentration is
    provided.
    """

    def __init__(self) -> None:
        """Initialize empty builder state."""
        self._names: Optional[list[str]] = None
        self._molar_mass: Optional[NDArray[np.float64]] = None
        self._concentration: Optional[NDArray[np.float64]] = None
        self._partitioning: Optional[NDArray[np.bool_]] = None
        self._n_boxes: Optional[int] = None

    def set_names(self, names: list[str]) -> "GasDataBuilder":
        """Set species names.

        Args:
            names: List of species names.

        Returns:
            Self for fluent chaining.
        """
        self._names = list(names)
        return self

    def set_molar_mass(
        self,
        molar_mass: Union[list[float], NDArray[np.float64]],
        units: str = "kg/mol",
    ) -> "GasDataBuilder":
        """Set molar masses with optional unit conversion.

        Args:
            molar_mass: Molar mass values shaped (n_species,).
            units: Units of the provided values. Supported: kg/mol, g/mol.

        Returns:
            Self for fluent chaining.

        Raises:
            ValueError: When molar_mass is not 1D or any value is non-positive.
        """
        molar_mass_array = np.asarray(molar_mass, dtype=np.float64)
        if molar_mass_array.ndim != 1:
            raise ValueError("molar_mass must be 1D")

        if units != "kg/mol":
            molar_mass_array = molar_mass_array * get_unit_conversion(
                units, "kg/mol"
            )

        if np.any(molar_mass_array <= 0):
            raise ValueError("molar_mass must be positive")

        self._molar_mass = molar_mass_array
        return self

    def set_concentration(
        self,
        concentration: Union[list[float], NDArray[np.float64]],
        units: str = "1/m^3",
    ) -> "GasDataBuilder":
        """Set concentrations with unit conversion and auto batch dimension.

        Args:
            concentration: Concentration values. If 1D (n_species,),
                batch dimension added. If 2D (n_boxes, n_species), used as-is.
            units: Units of the provided concentration. Supported: 1/m^3,
                1/cm^3.

        Returns:
            Self for fluent chaining.

        Raises:
            ValueError: When any concentration is negative or wrong dimension.
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

        if np.any(concentration_array < 0):
            raise ValueError("concentration must be non-negative")

        self._concentration = concentration_array
        return self

    def set_partitioning(
        self, partitioning: Union[list[bool], NDArray[np.bool_]]
    ) -> "GasDataBuilder":
        """Set whether each species can partition.

        Args:
            partitioning: Boolean array indicating which species partition.
                Shape: (n_species,).

        Returns:
            Self for fluent chaining.

        Raises:
            ValueError: When partitioning is not 1D.
        """
        partitioning_array = np.asarray(partitioning, dtype=np.bool_)
        if partitioning_array.ndim != 1:
            raise ValueError("partitioning must be 1D")

        self._partitioning = partitioning_array
        return self

    def set_n_boxes(self, n_boxes: int) -> "GasDataBuilder":
        """Set number of boxes for broadcasting 1D concentration.

        Args:
            n_boxes: Number of simulation boxes.

        Returns:
            Self for fluent chaining.
        """
        self._n_boxes = int(n_boxes)
        return self

    def build(self) -> GasData:
        """Build and return validated GasData.

        Returns:
            A validated GasData instance.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        if self._names is None:
            raise ValueError("names is required")
        if self._molar_mass is None:
            raise ValueError("molar_mass is required")
        if self._concentration is None:
            raise ValueError("concentration is required")
        if self._partitioning is None:
            raise ValueError("partitioning is required")

        # Handle n_boxes broadcasting if set
        if self._n_boxes is not None and self._concentration.shape[0] == 1:
            self._concentration = np.broadcast_to(
                self._concentration,
                (self._n_boxes, self._concentration.shape[1]),
            ).copy()

        return GasData(
            name=self._names,
            molar_mass=self._molar_mass,
            concentration=self._concentration,
            partitioning=self._partitioning,
        )
