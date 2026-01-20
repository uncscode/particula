"""Import verification tests for particula public exports.

Ensures activity and equilibria surfaces remain accessible via the
standard ``import particula as par`` pattern and legacy shortcuts emit
DeprecationWarning while delegating.
"""

from __future__ import annotations

import particula as par
import pytest


# Regression guard: activity strategy/builder exports remain reachable.
def test_activity_imports() -> None:
    """Activity strategies are exposed via particula.particles."""
    assert hasattr(par.particles, "ActivityNonIdealBinary")
    assert hasattr(par.particles, "ActivityNonIdealBinaryBuilder")


# Regression guard: equilibria strategy/builder/factory/runnable exports.
def test_equilibria_imports() -> None:
    """Equilibria surface is exposed via particula.equilibria."""
    assert hasattr(par.equilibria, "EquilibriaStrategy")
    assert hasattr(par.equilibria, "LiquidVaporPartitioningStrategy")
    assert hasattr(par.equilibria, "LiquidVaporPartitioningBuilder")
    assert hasattr(par.equilibria, "EquilibriaFactory")
    assert hasattr(par.equilibria, "Equilibria")
    assert hasattr(par.equilibria, "EquilibriumResult")
    assert hasattr(par.equilibria, "PhaseConcentrations")


# Regression guard: legacy wrapper continues to warn and delegate.
def test_legacy_deprecation_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy wrapper warns and delegates to implementation."""
    sentinel_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _sentinel(*args: object, **kwargs: object) -> str:
        sentinel_calls.append((args, kwargs))
        return "sentinel-return"

    monkeypatch.setattr(
        par.equilibria._partitioning, "liquid_vapor_partitioning", _sentinel
    )

    with pytest.warns(DeprecationWarning):
        result = par.equilibria.liquid_vapor_partitioning("sentinel-arg")

    assert result == "sentinel-return"
    assert sentinel_calls == [(("sentinel-arg",), {})]
