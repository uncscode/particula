"""Provide per-box thermodynamic state for gas-phase simulations.

This module defines :class:`EnvironmentData`, a mutable container for the
temperature, pressure, and saturation-ratio fields associated with one or
more simulation boxes. Separating this state from gas-species data lets
multi-box workflows manage shared thermodynamic conditions independently.

`EnvironmentData` is a constructor-validated CPU-side container in
``particula.gas.environment_data`` and is also exported from
``particula.gas`` for package-level imports. It does not yet have CPU↔GPU
conversion helpers.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(eq=False)
class EnvironmentData:
    """Mutable thermodynamic environment data for one or more boxes.

    This class stores box-level temperature and pressure together with a
    per-box, per-species saturation-ratio array. The temperature array is
    the source of truth for the box axis, so derived properties such as
    ``n_boxes`` follow ``temperature.shape[0]``.

    Attributes:
        temperature: Box temperatures in kelvin. Shape: ``(n_boxes,)``.
        pressure: Box pressures in pascals. Shape: ``(n_boxes,)``.
        saturation_ratio: Per-box, per-species saturation ratios.
            Shape: ``(n_boxes, n_species)``.

    Raises:
        ValueError: If any field has invalid dimensionality, shape,
            finiteness, or physical bounds.
    """

    temperature: NDArray[np.float64]
    pressure: NDArray[np.float64]
    saturation_ratio: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Coerce constructor inputs and validate the field contract.

        Raises:
            ValueError: If any field cannot be coerced to ``np.float64`` or
                fails dimensionality, shape, finiteness, or physical-bound
                validation.
        """
        self.temperature = self._coerce_float64_array(
            self.temperature,
            field_name="temperature",
        )
        self.pressure = self._coerce_float64_array(
            self.pressure,
            field_name="pressure",
        )
        self.saturation_ratio = self._coerce_float64_array(
            self.saturation_ratio,
            field_name="saturation_ratio",
        )

        self._validate_dimensionality()
        self._validate_shapes()
        self._validate_finite_values()
        self._validate_physical_bounds()

    @property
    def n_boxes(self) -> int:
        """Return the number of simulation boxes.

        Returns:
            Number of boxes represented by ``temperature.shape[0]``.
        """
        return int(self.temperature.shape[0])

    def copy(self) -> "EnvironmentData":
        """Create an independent copy of the environment state.

        Returns:
            New :class:`EnvironmentData` instance with copied temperature,
            pressure, and saturation-ratio arrays.
        """
        return EnvironmentData(
            temperature=np.copy(self.temperature),
            pressure=np.copy(self.pressure),
            saturation_ratio=np.copy(self.saturation_ratio),
        )

    @staticmethod
    def _coerce_float64_array(
        values: object,
        *,
        field_name: str,
    ) -> NDArray[np.float64]:
        """Convert array-like input to a ``np.float64`` NumPy array.

        Args:
            values: Array-like input supplied for a dataclass field.
            field_name: Name of the field being coerced for error reporting.

        Returns:
            NumPy array with ``np.float64`` dtype.

        Raises:
            ValueError: If ``values`` cannot be converted to a float64 array.
        """
        try:
            return np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{field_name} must be array-like and coercible to float64"
            ) from exc

    def _validate_dimensionality(self) -> None:
        """Validate per-field dimensionality requirements.

        Raises:
            ValueError: If ``temperature`` or ``pressure`` is not 1D, or if
                ``saturation_ratio`` is not 2D.
        """
        if self.temperature.ndim != 1:
            raise ValueError(
                "temperature must be 1D (n_boxes,), "
                f"got ndim={self.temperature.ndim}"
            )

        if self.pressure.ndim != 1:
            raise ValueError(
                f"pressure must be 1D (n_boxes,), got ndim={self.pressure.ndim}"
            )

        if self.saturation_ratio.ndim != 2:
            raise ValueError(
                "saturation_ratio must be 2D (n_boxes, n_species), "
                f"got ndim={self.saturation_ratio.ndim}"
            )

    def _validate_shapes(self) -> None:
        """Validate shared box-count consistency after ndim checks.

        Raises:
            ValueError: If no boxes are provided, or if ``pressure`` or
                ``saturation_ratio`` does not share the same leading box
                dimension as ``temperature``.
        """
        n_boxes = self.temperature.shape[0]

        if n_boxes == 0:
            raise ValueError(
                "EnvironmentData requires at least one box "
                "(temperature, pressure, and saturation_ratio must not be "
                "empty along the box dimension)"
            )

        if self.pressure.shape != (n_boxes,):
            raise ValueError(
                "pressure shape must match temperature n_boxes: "
                f"got {self.pressure.shape}, expected ({n_boxes},)"
            )

        if self.saturation_ratio.shape[0] != n_boxes:
            raise ValueError(
                "saturation_ratio leading dimension must match temperature "
                f"n_boxes: got {self.saturation_ratio.shape[0]}, "
                f"expected {n_boxes}"
            )

    def _validate_finite_values(self) -> None:
        """Require all numeric fields to be finite.

        Raises:
            ValueError: If any field contains ``NaN`` or infinite values.
        """
        if not np.all(np.isfinite(self.temperature)):
            raise ValueError("temperature must contain only finite values")

        if not np.all(np.isfinite(self.pressure)):
            raise ValueError("pressure must contain only finite values")

        if not np.all(np.isfinite(self.saturation_ratio)):
            raise ValueError("saturation_ratio must contain only finite values")

    def _validate_physical_bounds(self) -> None:
        """Require positive thermodynamic state and nonnegative saturation.

        Raises:
            ValueError: If temperature or pressure is nonpositive, or if any
                saturation ratio is negative.
        """
        if np.any(self.temperature <= 0.0):
            raise ValueError("temperature must be strictly positive")

        if np.any(self.pressure <= 0.0):
            raise ValueError("pressure must be strictly positive")

        if np.any(self.saturation_ratio < 0.0):
            raise ValueError("saturation_ratio must be nonnegative")
