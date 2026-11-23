"""Builder implementations for equilibria strategies.

Provides fluent builders to configure equilibria strategies with
validation and deterministic parameter handling.
"""

from __future__ import annotations

from typing import Any, Self

from particula.abc_builder import BuilderABC
from particula.equilibria.equilibria_strategies import (
    LiquidVaporPartitioningStrategy,
)


class LiquidVaporPartitioningBuilder(BuilderABC):
    """Builder for :class:`LiquidVaporPartitioningStrategy`.

    Supports fluent configuration with validation for water activity and
    deterministic parameter handling for factory usage.

    Examples:
        >>> builder = LiquidVaporPartitioningBuilder()
        >>> strategy = builder.set_water_activity(0.8).build()
        >>> strategy.water_activity
        0.8

        >>> builder = LiquidVaporPartitioningBuilder()
        >>> strategy = builder.set_parameters({"water_activity": 0.65}).build()
        >>> strategy.water_activity
        0.65
    """

    def __init__(self) -> None:
        """Initialize builder with default water activity."""
        super().__init__(required_parameters=[])
        self._activity_model = None
        self._explicit_water = False
        self._water_activity: float | None = 0.5

    def set_water_activity(self, value: float) -> Self:
        """Set the target water activity.

        Args:
            value: Water activity in range [0, 1].

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If ``value`` is outside the inclusive [0, 1] range.
        """
        water_activity = float(value)
        if not 0 <= water_activity <= 1:
            raise ValueError(f"water_activity must be in [0, 1], got {value}")
        self._explicit_water = True
        self._water_activity = water_activity
        return self

    def set_activity_model(self, model) -> Self:
        """Use a mixture activity provider for gas/liquid partitioning."""
        self._activity_model = model
        return self

    def set_parameters(self, parameters: dict[str, Any]) -> Self:
        """Set parameters from a dictionary with strict key validation.

        Accept a BAT water activity or a configured activity model.
        Unknown keys, including
        unit-suffixed variants, raise ``ValueError`` to keep behavior
        deterministic for factories.

        Args:
            parameters: Parameter dictionary.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If an unknown key is provided or value is invalid.
        """
        allowed_keys = {"water_activity", "activity_model"}
        invalid_keys = set(parameters) - allowed_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid parameter(s): {sorted(invalid_keys)}. "
                "Allowed keys: water_activity, activity_model."
            )

        if "activity_model" in parameters:
            self.set_activity_model(parameters["activity_model"])
        if "water_activity" in parameters:
            self.set_water_activity(parameters["water_activity"])

        return self

    def pre_build_check(self) -> None:
        """Ensure builder is ready to build a strategy."""
        super().pre_build_check()
        if self._water_activity is None:
            raise ValueError("water_activity must be set before build")

    def build(self) -> LiquidVaporPartitioningStrategy:
        """Build the configured :class:`LiquidVaporPartitioningStrategy`."""
        self.pre_build_check()
        if self._water_activity is None:
            raise ValueError("water_activity must be set before build")
        return LiquidVaporPartitioningStrategy(
            water_activity=self._water_activity
            if self._explicit_water
            else None,
            activity_model=self._activity_model,
        )
