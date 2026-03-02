"""Latent heat strategies for gas species.

Latent heat values are expressed in J/kg for per-species mass-based
calculations.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class LatentHeatStrategy(ABC):
    """Abstract base class for latent heat calculations.

    Subclasses implement :meth:`latent_heat` to return the latent heat of
    vaporization for the specified temperature input.
    """

    @abstractmethod
    def latent_heat(
        self, temperature: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """Return latent heat of vaporization for the given temperature.

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            Latent heat of vaporization in J/kg.
        """


class ConstantLatentHeat(LatentHeatStrategy):
    """Latent heat strategy that returns a constant value."""

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
            temperature: Temperature in Kelvin.

        Returns:
            Latent heat in J/kg with scalar or array shape matching input.
        """
        if np.isscalar(temperature):
            return float(self.latent_heat_ref)
        return np.full_like(
            np.asarray(temperature, dtype=np.float64),
            self.latent_heat_ref,
            dtype=np.float64,
        )
