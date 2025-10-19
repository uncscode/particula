"""Mixin classes for Condensation strategy builders."""

# pylint: disable=too-few-public-methods

import logging
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from particula.util.convert_units import get_unit_conversion
from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")


class BuilderDiffusionCoefficientMixin:
    """Mixin to set a diffusion coefficient in m^2/s."""

    def __init__(self) -> None:
        """Initialize the diffusion coefficient mixin.

        Sets the diffusion_coefficient attribute to None.
        """
        self.diffusion_coefficient: Optional[
            Union[float, NDArray[np.float64]]
        ] = None

    @validate_inputs({"diffusion_coefficient": "positive"})
    def set_diffusion_coefficient(
        self,
        diffusion_coefficient: Union[float, NDArray[np.float64]],
        diffusion_coefficient_units: str,
    ):
        """Set the diffusion coefficient for the condensing species."""
        if diffusion_coefficient_units == "m^2/s":
            self.diffusion_coefficient = diffusion_coefficient
            return self
        self.diffusion_coefficient = (
            diffusion_coefficient
            * get_unit_conversion(
                diffusion_coefficient_units,
                "m^2/s",
            )
        )
        return self


class BuilderAccommodationCoefficientMixin:
    """Mixin to set the mass accommodation coefficient."""

    def __init__(self) -> None:
        """Initialize the accommodation coefficient mixin.

        Sets the accommodation_coefficient attribute to None.
        """
        self.accommodation_coefficient: Optional[
            Union[float, NDArray[np.float64]]
        ] = None

    @validate_inputs({"accommodation_coefficient": "nonnegative"})
    def set_accommodation_coefficient(
        self,
        accommodation_coefficient: Union[float, NDArray[np.float64]],
        accommodation_coefficient_units: Optional[str] = None,
    ):
        """Set the dimensionless mass accommodation coefficient."""
        if accommodation_coefficient_units is not None:
            logger.warning(
                "Ignoring units for accommodation coefficient parameter."
            )
        self.accommodation_coefficient = accommodation_coefficient
        return self


class BuilderUpdateGasesMixin:
    """Mixin to specify whether the gas phase should be updated."""

    def __init__(self) -> None:
        """Initialize the update gases mixin.

        Sets the update_gases flag to True by default.
        """
        self.update_gases: bool = True

    def set_update_gases(
        self, update_gases: bool, update_gases_units: Optional[str] = None
    ):
        """Set the flag controlling gas-phase updates."""
        if update_gases_units is not None:
            logger.warning("Ignoring units for update_gases parameter.")
        self.update_gases = update_gases
        return self
