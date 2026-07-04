"""Provide per-box thermodynamic environment data for gas simulations.

EnvironmentData isolates mutable temperature, pressure, and
saturation-ratio arrays from gas-species state so multi-box simulations can
manage thermodynamic conditions independently.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class EnvironmentData:
    """Mutable thermodynamic environment data for one or more boxes.

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
        """Coerce array inputs and validate the environment contract."""
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

    @staticmethod
    def _coerce_float64_array(
        values: object,
        *,
        field_name: str,
    ) -> NDArray[np.float64]:
        """Convert array-like input to a float64 NumPy array."""
        try:
            return np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must be array-like and coercible to float64"
            ) from exc

    def _validate_dimensionality(self) -> None:
        """Validate per-field dimensionality requirements."""
        if self.temperature.ndim != 1:
            raise ValueError(
                "temperature must be 1D (n_boxes,), "
                f"got ndim={self.temperature.ndim}"
            )

        if self.pressure.ndim != 1:
            raise ValueError(
                "pressure must be 1D (n_boxes,), "
                f"got ndim={self.pressure.ndim}"
            )

        if self.saturation_ratio.ndim != 2:
            raise ValueError(
                "saturation_ratio must be 2D (n_boxes, n_species), "
                f"got ndim={self.saturation_ratio.ndim}"
            )

    def _validate_shapes(self) -> None:
        """Validate shared box-count consistency after ndim checks."""
        n_boxes = self.temperature.shape[0]

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
        """Require all numeric fields to be finite."""
        if not np.all(np.isfinite(self.temperature)):
            raise ValueError("temperature must contain only finite values")

        if not np.all(np.isfinite(self.pressure)):
            raise ValueError("pressure must contain only finite values")

        if not np.all(np.isfinite(self.saturation_ratio)):
            raise ValueError(
                "saturation_ratio must contain only finite values"
            )

    def _validate_physical_bounds(self) -> None:
        """Require positive thermodynamic state and nonnegative saturation."""
        if np.any(self.temperature <= 0.0):
            raise ValueError("temperature must be strictly positive")

        if np.any(self.pressure <= 0.0):
            raise ValueError("pressure must be strictly positive")

        if np.any(self.saturation_ratio < 0.0):
            raise ValueError("saturation_ratio must be nonnegative")
