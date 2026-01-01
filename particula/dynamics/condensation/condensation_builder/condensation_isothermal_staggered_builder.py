"""Builder for the CondensationIsothermalStaggered strategy.

Provides a fluent interface to configure the staggered isothermal
condensation strategy with validation and sensible defaults.

Example:
    >>> from particula.dynamics.condensation.condensation_builder import (
    ...     CondensationIsothermalStaggeredBuilder,
    ... )
    >>> builder = CondensationIsothermalStaggeredBuilder()
    >>> strategy = (
    ...     builder
    ...     .set_molar_mass(0.018, "kg/mol")
    ...     .set_diffusion_coefficient(2e-5, "m^2/s")
    ...     .set_accommodation_coefficient(1.0)
    ...     .set_theta_mode("random")
    ...     .set_num_batches(10)
    ...     .set_shuffle_each_step(True)
    ...     .set_random_state(42)
    ...     .build()
    ... )
"""

import logging
from typing import Any, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray

from particula.abc_builder import BuilderABC
from particula.builder_mixin import BuilderMolarMassMixin
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermalStaggered,
    CondensationStrategy,
)

from .condensation_builder_mixin import (
    BuilderAccommodationCoefficientMixin,
    BuilderDiffusionCoefficientMixin,
    BuilderUpdateGasesMixin,
)

logger = logging.getLogger("particula")


class CondensationIsothermalStaggeredBuilder(
    BuilderABC,
    BuilderMolarMassMixin,
    BuilderDiffusionCoefficientMixin,
    BuilderAccommodationCoefficientMixin,
    BuilderUpdateGasesMixin,
):
    """Fluent builder for :class:`CondensationIsothermalStaggered`.

    Extends the base condensation builder with staggered-stepping-specific
    parameters ``theta_mode``, ``num_batches``, ``shuffle_each_step``, and
    ``random_state``.
    """

    def __init__(self) -> None:
        """Initialize the builder with required parameters and defaults."""
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

        self.theta_mode: str = "half"
        self.num_batches: int = 1
        self.shuffle_each_step: bool = True
        self.random_state: Optional[
            Union[int, np.random.Generator, np.random.RandomState]
        ] = None

    def set_parameters(
        self, parameters: dict[str, Any]
    ) -> "CondensationIsothermalStaggeredBuilder":
        """Set required and optional parameters from a dictionary."""
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
                "theta_mode",
                "num_batches",
                "shuffle_each_step",
                "random_state",
                "update_gases",
            ]
        )
        if invalid_keys := [key for key in parameters if key not in valid_keys]:
            error_message = (
                f"Trying to set an invalid parameter(s) '{invalid_keys}'. "
                f"The valid parameter(s) '{valid_keys}'."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        for key in required:
            unit_key = f"{key}_units"
            if unit_key in parameters:
                getattr(self, f"set_{key}")(
                    parameters[key], parameters[unit_key]
                )
            else:
                logger.warning("Using default units for parameter: '%s'.", key)
                getattr(self, f"set_{key}")(parameters[key])

        if "theta_mode" in parameters:
            self.set_theta_mode(parameters["theta_mode"])
        if "num_batches" in parameters:
            self.set_num_batches(parameters["num_batches"])
        if "shuffle_each_step" in parameters:
            self.set_shuffle_each_step(parameters["shuffle_each_step"])
        if "random_state" in parameters:
            self.set_random_state(parameters["random_state"])
        if "update_gases" in parameters:
            self.set_update_gases(parameters["update_gases"])

        return self

    def set_theta_mode(
        self, theta_mode: str
    ) -> "CondensationIsothermalStaggeredBuilder":
        """Set the staggered stepping mode.

        Args:
            theta_mode: One of ``("half", "random", "batch")``.

        Returns:
            The builder instance for chaining.

        Raises:
            ValueError: If ``theta_mode`` is not supported.
        """
        valid_modes = CondensationIsothermalStaggered.VALID_THETA_MODES
        if theta_mode not in valid_modes:
            raise ValueError(
                f"theta_mode must be one of {valid_modes}, got '{theta_mode}'"
            )
        self.theta_mode = theta_mode
        return self

    def set_num_batches(
        self, num_batches: int
    ) -> "CondensationIsothermalStaggeredBuilder":
        """Set the number of batches for staggered updates.

        Args:
            num_batches: Number of batches; must be at least 1.

        Returns:
            The builder instance for chaining.

        Raises:
            ValueError: If ``num_batches`` is less than 1.
        """
        if num_batches < 1:
            raise ValueError("num_batches must be >= 1.")
        self.num_batches = num_batches
        return self

    def set_shuffle_each_step(
        self, shuffle: bool
    ) -> "CondensationIsothermalStaggeredBuilder":
        """Enable or disable shuffling at each step.

        Args:
            shuffle: Whether to shuffle particle order every step.

        Returns:
            The builder instance for chaining.
        """
        self.shuffle_each_step = shuffle
        return self

    def set_random_state(
        self,
        random_state: Optional[
            Union[int, np.random.Generator, np.random.RandomState]
        ],
    ) -> "CondensationIsothermalStaggeredBuilder":
        """Set the random state for reproducibility.

        Args:
            random_state: Seed or RNG controlling random theta generation.

        Returns:
            The builder instance for chaining.
        """
        self.random_state = random_state
        return self

    def build(self) -> CondensationStrategy:
        """Validate parameters and create a condensation strategy."""
        self.pre_build_check()

        # pre_build_check ensures these are not None
        return CondensationIsothermalStaggered(
            molar_mass=cast(Union[float, NDArray[np.float64]], self.molar_mass),
            diffusion_coefficient=cast(
                Union[float, NDArray[np.float64]], self.diffusion_coefficient
            ),
            accommodation_coefficient=cast(
                Union[float, NDArray[np.float64]],
                self.accommodation_coefficient,
            ),
            update_gases=self.update_gases,
            theta_mode=self.theta_mode,
            num_batches=self.num_batches,
            shuffle_each_step=self.shuffle_each_step,
            random_state=self.random_state,
        )
