"""Process-owned native configuration shared by condensation builders."""

from typing import Any, Self, Sequence

from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas.vapor_pressure_strategies import VaporPressureStrategy
from particula.particles.activity_strategies import ActivityStrategy
from particula.particles.surface_strategies import SurfaceStrategy


class BuilderNativeCondensationMixin:
    """Carry native strategy configuration through fluent and factory setup.

    Legacy construction may omit every field. Explicit native calls validate
    required configuration against current data at execution, including names
    and lane bounds. Builders retain process strategy identities, never data.
    """

    _NATIVE_KEYS = (
        "species_map",
        "activity_strategy",
        "surface_strategy",
        "vapor_pressure_strategy",
        "skip_partitioning_indices",
    )

    def __init__(self) -> None:
        """Initialize optional native configuration."""
        self.species_map: CondensationSpeciesMap | None = None
        self.activity_strategy: ActivityStrategy | None = None
        self.surface_strategy: SurfaceStrategy | None = None
        self.vapor_pressure_strategy: (
            VaporPressureStrategy | tuple[VaporPressureStrategy, ...] | None
        ) = None
        self.skip_partitioning_indices: tuple[int, ...] | None = None

    def set_species_map(
        self, species_map: CondensationSpeciesMap | None
    ) -> Self:
        """Set immutable full-gas provenance and explicit transfer pairs."""
        if species_map is not None and not isinstance(
            species_map, CondensationSpeciesMap
        ):
            raise TypeError(
                "species_map must be CondensationSpeciesMap or None"
            )
        self.species_map = species_map
        return self

    def set_activity_strategy(
        self, activity_strategy: ActivityStrategy | None
    ) -> Self:
        """Set the activity model that reads the full particle composition."""
        if activity_strategy is not None and not isinstance(
            activity_strategy, ActivityStrategy
        ):
            raise TypeError(
                "activity_strategy must be ActivityStrategy or None"
            )
        self.activity_strategy = activity_strategy
        return self

    def set_surface_strategy(
        self, surface_strategy: SurfaceStrategy | None
    ) -> Self:
        """Set the surface model that reads the full particle composition."""
        if surface_strategy is not None and not isinstance(
            surface_strategy, SurfaceStrategy
        ):
            raise TypeError("surface_strategy must be SurfaceStrategy or None")
        self.surface_strategy = surface_strategy
        return self

    def set_vapor_pressure_strategy(
        self,
        vapor_pressure_strategy: VaporPressureStrategy
        | Sequence[VaporPressureStrategy]
        | None,
    ) -> Self:
        """Set one vector model or a detached ordered model sequence."""
        if vapor_pressure_strategy is None or isinstance(
            vapor_pressure_strategy, VaporPressureStrategy
        ):
            self.vapor_pressure_strategy = vapor_pressure_strategy
            return self
        if not isinstance(vapor_pressure_strategy, Sequence) or not all(
            isinstance(model, VaporPressureStrategy)
            for model in vapor_pressure_strategy
        ):
            raise TypeError(
                "vapor pressure models must be VaporPressureStrategy"
            )
        self.vapor_pressure_strategy = tuple(vapor_pressure_strategy)
        return self

    def set_skip_partitioning_indices(
        self, indices: Sequence[int] | None
    ) -> Self:
        """Detach full-gas skip indices; execution checks lane bounds."""
        self.skip_partitioning_indices = (
            None if indices is None else tuple(indices)
        )
        return self

    def _load_native_parameters(
        self, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Load optional native keys and return remaining builder parameters."""
        # Stage in a separate builder fragment so invalid native types do not
        # partially replace a previously valid native configuration.
        candidate = BuilderNativeCondensationMixin()
        for key in self._NATIVE_KEYS:
            if key in parameters:
                getattr(candidate, f"set_{key}")(parameters[key])
        for key in self._NATIVE_KEYS:
            setattr(self, key, getattr(candidate, key))
        return {
            key: value
            for key, value in parameters.items()
            if key not in self._NATIVE_KEYS
        }

    def _native_strategy_kwargs(self) -> dict[str, Any]:
        """Return process configuration without constructing data or facades."""
        return {key: getattr(self, key) for key in self._NATIVE_KEYS}
