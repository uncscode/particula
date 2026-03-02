"""Latent heat strategies for gas species.

Latent heat values are expressed in J/kg for per-species mass-based
calculations. Strategies return scalars for scalar temperature inputs and
arrays matching the input shape for array temperature inputs.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class LatentHeatStrategy(ABC):
    """Abstract base class for latent heat calculations.

    Subclasses implement :meth:`latent_heat` to return the latent heat of
    vaporization in J/kg for scalar or array temperature inputs.
    """

    @abstractmethod
    def latent_heat(
        self, temperature: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """Return latent heat of vaporization for the given temperature.

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            Latent heat of vaporization in J/kg with shape matching the input.
        """


class ConstantLatentHeat(LatentHeatStrategy):
    """Latent heat strategy that returns a constant value.

    Attributes:
        latent_heat_ref: Reference latent heat in J/kg returned by the
            strategy.
    """

    def __init__(self, latent_heat_ref: float) -> None:
        """Initialize the constant latent heat strategy.

        Args:
            latent_heat_ref: Reference latent heat in J/kg.
        """
        self.latent_heat_ref = latent_heat_ref

    def latent_heat(
        self, temperature: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """Return a constant latent heat value.

        Args:
            temperature: Temperature in Kelvin. Accepted for interface
                consistency.

        Returns:
            Latent heat in J/kg. Returns a scalar for scalar input or an array
            matching the input shape.
        """
        if np.isscalar(temperature):
            return float(self.latent_heat_ref)
        return np.full_like(
            np.asarray(temperature, dtype=np.float64),
            self.latent_heat_ref,
            dtype=np.float64,
        )
