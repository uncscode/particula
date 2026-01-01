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

from typing import Optional, Union

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

        if self.molar_mass is None:
            raise ValueError("molar_mass must be set")
        if self.diffusion_coefficient is None:
            raise ValueError("diffusion_coefficient must be set")
        if self.accommodation_coefficient is None:
            raise ValueError("accommodation_coefficient must be set")

        return CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            update_gases=self.update_gases,
            theta_mode=self.theta_mode,
            num_batches=self.num_batches,
            shuffle_each_step=self.shuffle_each_step,
            random_state=self.random_state,
        )
