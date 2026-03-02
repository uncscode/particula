"""Latent heat strategies for gas species.

Latent heat values are expressed in J/kg for per-species mass-based
calculations. Strategies return scalars for scalar temperature inputs and
arrays matching the input shape for array temperature inputs.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs


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

    Examples:
        ```py title="ConstantLatentHeat usage example"
        import numpy as np
        from particula.gas.latent_heat_strategies import ConstantLatentHeat

        strategy = ConstantLatentHeat(latent_heat_ref=2.26e6)
        print(strategy.latent_heat(298.15))
        print(strategy.latent_heat(np.array([280.0, 300.0])))
        ```
    """

    @validate_inputs({"latent_heat_ref": "positive"})
    def __init__(self, latent_heat_ref: float) -> None:
        """Initialize the constant latent heat strategy.

        Args:
            latent_heat_ref: Reference latent heat in J/kg.
        """
        self.latent_heat_ref: float = latent_heat_ref

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
        temperature_array = np.asarray(temperature, dtype=np.float64)
        if temperature_array.shape == ():
            return float(self.latent_heat_ref)
        return np.full_like(
            temperature_array,
            self.latent_heat_ref,
            dtype=np.float64,
        )
