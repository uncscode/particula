"""Thermodynamic and chemical equilibria utilities.

This module exposes the equilibria Strategy–Builder–Factory–Runnable
surface plus legacy helper functions for backward compatibility.
Recommended usage is via the strategy or factory interfaces; legacy
function shortcuts remain available but emit deprecation warnings.

Classes:
    EquilibriaStrategy: Abstract base class for equilibria strategies.
    LiquidVaporPartitioningStrategy: Liquid-vapor partitioning strategy.
    EquilibriumResult: Result dataclass for equilibrium calculations.
    PhaseConcentrations: Dataclass for phase concentration data.
    LiquidVaporPartitioningBuilder: Builder for partitioning strategies.
    EquilibriaFactory: Factory for equilibria strategies.
    Equilibria: Runnable wrapper for equilibria strategies.

Functions (legacy):
    liquid_vapor_partitioning: Direct partitioning calculation (legacy).
    get_properties_for_liquid_vapor_partitioning: Legacy property helper.
    liquid_vapor_obj_function: Objective function for partitioning (legacy).

References:
    Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
    Relative-humidity-dependent organic aerosol thermodynamics via an
    efficient reduced-complexity model. Atmospheric Chemistry and
    Physics, 19(19), 13383-13407.
    https://doi.org/10.5194/acp-19-13383-2019
"""

from __future__ import annotations

import warnings

from particula.equilibria import partitioning as _partitioning
from particula.equilibria.equilibria import Equilibria
from particula.equilibria.equilibria_builders import (
    LiquidVaporPartitioningBuilder,
)
from particula.equilibria.equilibria_factories import EquilibriaFactory
from particula.equilibria.equilibria_strategies import (
    EquilibriaStrategy,
    EquilibriumResult,
    LiquidVaporPartitioningStrategy,
    PhaseConcentrations,
)


def liquid_vapor_partitioning(*args, **kwargs):
    """Legacy shortcut for direct liquid-vapor partitioning.

    Emits a :class:`DeprecationWarning` and delegates to the partitioning
    implementation. Prefer :class:`LiquidVaporPartitioningStrategy` or
    :class:`Equilibria` for new code.
    """
    warnings.warn(
        "Direct liquid_vapor_partitioning is deprecated; use "
        "LiquidVaporPartitioningStrategy or Equilibria instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _partitioning.liquid_vapor_partitioning(*args, **kwargs)


def get_properties_for_liquid_vapor_partitioning(*args, **kwargs):
    """Legacy property helper for liquid-vapor partitioning.

    Emits a :class:`DeprecationWarning` and delegates to the partitioning
    implementation. Prefer the strategy interfaces for new code.
    """
    warnings.warn(
        "Direct get_properties_for_liquid_vapor_partitioning is deprecated; "
        "use strategy interfaces instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _partitioning.get_properties_for_liquid_vapor_partitioning(
        *args, **kwargs
    )


def liquid_vapor_obj_function(*args, **kwargs):
    """Legacy objective function for liquid-vapor partitioning.

    Emits a :class:`DeprecationWarning` and delegates to the partitioning
    implementation. Prefer the strategy interfaces for new code.
    """
    warnings.warn(
        "Direct liquid_vapor_obj_function is deprecated; use strategy "
        "interfaces instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _partitioning.liquid_vapor_obj_function(*args, **kwargs)


__all__ = [
    # Strategy pattern (recommended API)
    "EquilibriaStrategy",
    "LiquidVaporPartitioningStrategy",
    "EquilibriumResult",
    "PhaseConcentrations",
    # Builder pattern
    "LiquidVaporPartitioningBuilder",
    # Factory pattern
    "EquilibriaFactory",
    # Runnable
    "Equilibria",
    # Legacy API (backward compatibility)
    "liquid_vapor_partitioning",
    "get_properties_for_liquid_vapor_partitioning",
    "liquid_vapor_obj_function",
]
