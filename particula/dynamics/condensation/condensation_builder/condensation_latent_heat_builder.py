"""Builder for latent-heat-aware condensation strategies.

This builder configures shared condensation transport properties together
with either a latent heat strategy instance or a validated scalar latent
heat fallback before constructing :class:`CondensationLatentHeat`.
"""

import logging
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from particula.abc_builder import BuilderABC
from particula.builder_mixin import BuilderMolarMassMixin
from particula.dynamics.condensation.condensation_strategies import (
    CondensationLatentHeat,
)
from particula.gas.latent_heat_strategies import LatentHeatStrategy

from .condensation_builder_mixin import (
    BuilderAccommodationCoefficientMixin,
    BuilderDiffusionCoefficientMixin,
    BuilderUpdateGasesMixin,
)

logger = logging.getLogger("particula")


class CondensationLatentHeatBuilder(
    BuilderABC,
    BuilderMolarMassMixin,
    BuilderDiffusionCoefficientMixin,
    BuilderAccommodationCoefficientMixin,
    BuilderUpdateGasesMixin,
):
    """Fluent builder for :class:`CondensationLatentHeat`.

    Attributes:
        latent_heat_strategy: Optional latent heat strategy passed directly to
            the constructed condensation strategy.
        latent_heat: Optional positive scalar latent heat fallback in J/kg.
    """

    def __init__(self) -> None:
        """Initialize the builder with required shared parameters.

        The builder starts with the common condensation transport inputs and
        no latent heat override so :meth:`build` only forwards optional latent
        heat configuration when it was explicitly provided.
        """
        required_parameters = [
            "molar_mass",
            "diffusion_coefficient",
            "accommodation_coefficient",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderMolarMassMixin.__init__(self)
        BuilderDiffusionCoefficientMixin.__init__(self)
        BuilderAccommodationCoefficientMixin.__init__(self)
        BuilderUpdateGasesMixin.__init__(self)

        self.latent_heat_strategy: LatentHeatStrategy | None = None
        self.latent_heat: float | None = None
        self._latent_heat_explicitly_set = False

    def _reset_optional_latent_heat_state(self) -> None:
        """Clear optional latent-heat settings for a fresh parameter load."""
        self.latent_heat_strategy = None
        self.latent_heat = None
        self._latent_heat_explicitly_set = False

    def set_parameters(
        self, parameters: dict[str, Any]
    ) -> "CondensationLatentHeatBuilder":
        """Set required and optional parameters from a dictionary.

        Args:
            parameters: Required shared builder parameters plus optional
                ``update_gases``, ``latent_heat_strategy``, and
                ``latent_heat``.

        Returns:
            The builder instance for chaining.

        Raises:
            ValueError: If required keys are missing or any key is invalid.
        """
        required = self.required_parameters
        missing = [param for param in required if param not in parameters]
        if missing:
            error_message = (
                f"Missing required parameter(s): {', '.join(missing)}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        valid_keys = set(
            required
            + [f"{key}_units" for key in required]
            + [
                "update_gases",
                "latent_heat_strategy",
                "latent_heat",
            ]
        )
        if invalid_keys := [key for key in parameters if key not in valid_keys]:
            error_message = (
                f"Trying to set an invalid parameter(s) '{invalid_keys}'. "
                f"The valid parameter(s) '{valid_keys}'."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        self._reset_optional_latent_heat_state()

        default_units = {
            "molar_mass": "kg/mol",
            "diffusion_coefficient": "m^2/s",
            "accommodation_coefficient": None,
        }

        for key in required:
            unit_key = f"{key}_units"
            units = parameters.get(unit_key, default_units[key])
            if unit_key in parameters:
                getattr(self, f"set_{key}")(parameters[key], units)
            else:
                logger.warning("Using default units for parameter: '%s'.", key)
                getattr(self, f"set_{key}")(parameters[key], units)

        if "update_gases" in parameters:
            self.set_update_gases(parameters["update_gases"])
        if "latent_heat_strategy" in parameters:
            self.set_latent_heat_strategy(parameters["latent_heat_strategy"])
        if "latent_heat" in parameters:
            self.set_latent_heat(parameters["latent_heat"])

        return self

    def set_latent_heat_strategy(
        self, latent_heat_strategy: LatentHeatStrategy | None
    ) -> "CondensationLatentHeatBuilder":
        """Store a latent heat strategy for direct passthrough.

        Args:
            latent_heat_strategy: Strategy object used by
                :class:`CondensationLatentHeat`.

        Returns:
            The builder instance for chaining.

        Notes:
            Passing ``None`` clears any previously stored strategy.

        Raises:
            TypeError: If ``latent_heat_strategy`` is not ``None`` and does not
                implement :class:`LatentHeatStrategy`.
        """
        if latent_heat_strategy is not None and not isinstance(
            latent_heat_strategy, LatentHeatStrategy
        ):
            raise TypeError(
                "latent_heat_strategy must be a LatentHeatStrategy or None, "
                f"got {type(latent_heat_strategy).__name__}"
            )
        self.latent_heat_strategy = latent_heat_strategy
        return self

    def set_latent_heat(
        self, latent_heat: float | NDArray[np.float64] | None
    ) -> "CondensationLatentHeatBuilder":
        """Set a positive finite scalar latent heat fallback.

        Args:
            latent_heat: Scalar latent heat value in J/kg.

        Returns:
            The builder instance for chaining.

        Raises:
            ValueError: If ``latent_heat`` is None, array-like, non-finite,
                zero, or negative.
        """
        if latent_heat is None:
            raise ValueError(
                "latent_heat must be a positive finite scalar, got None"
            )

        latent_heat_array = np.asarray(latent_heat, dtype=np.float64)
        if latent_heat_array.shape != ():
            raise ValueError(
                "latent_heat must be a positive finite scalar, got "
                f"array-like value {latent_heat!r}"
            )

        latent_heat_value = float(latent_heat_array)
        if not np.isfinite(latent_heat_value):
            raise ValueError(
                "latent_heat must be a positive finite scalar, got "
                f"non-finite value {latent_heat_value!r}"
            )
        if latent_heat_value <= 0:
            raise ValueError(
                "latent_heat must be a positive finite scalar, got "
                f"non-positive value {latent_heat_value!r}"
            )

        self.latent_heat = latent_heat_value
        self._latent_heat_explicitly_set = True
        return self

    def build(self) -> CondensationLatentHeat:
        """Validate parameters and create a latent-heat strategy.

        Returns:
            A configured :class:`CondensationLatentHeat` instance.

        Notes:
            An explicit ``latent_heat_strategy`` is forwarded unchanged. A
            scalar ``latent_heat`` is only passed when
            :meth:`set_latent_heat` was called successfully.
        """
        self.pre_build_check()

        build_kwargs: dict[str, Any] = {
            "molar_mass": self.molar_mass,
            "diffusion_coefficient": cast(
                float | NDArray[np.float64], self.diffusion_coefficient
            ),
            "accommodation_coefficient": cast(
                float | NDArray[np.float64],
                self.accommodation_coefficient,
            ),
            "update_gases": self.update_gases,
        }
        if self.latent_heat_strategy is not None:
            build_kwargs["latent_heat_strategy"] = self.latent_heat_strategy
        if self._latent_heat_explicitly_set:
            build_kwargs["latent_heat"] = cast(float, self.latent_heat)

        return CondensationLatentHeat(**build_kwargs)
