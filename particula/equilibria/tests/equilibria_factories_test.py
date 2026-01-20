from __future__ import annotations

import pytest

from particula.equilibria.equilibria_builders import (
    LiquidVaporPartitioningBuilder,
)
from particula.equilibria.equilibria_factories import EquilibriaFactory
from particula.equilibria.equilibria_strategies import (
    EquilibriaStrategy,
    LiquidVaporPartitioningStrategy,
)


@pytest.fixture()
def factory() -> EquilibriaFactory:
    """Provide a fresh factory per test."""

    return EquilibriaFactory()


def test_get_builders_contains_liquid_vapor(factory: EquilibriaFactory) -> None:
    """Factory exposes liquid_vapor builder."""

    builders = factory.get_builders()

    assert "liquid_vapor" in builders
    assert isinstance(builders["liquid_vapor"], LiquidVaporPartitioningBuilder)


def test_get_builders_returns_fresh_instances(
    factory: EquilibriaFactory,
) -> None:
    """Each call returns new builder instances."""

    first = factory.get_builders()
    second = factory.get_builders()

    assert first is not second
    assert first["liquid_vapor"] is not second["liquid_vapor"]


def test_factory_instances_have_isolated_builders() -> None:
    """Factories do not share builder instances."""

    first_factory = EquilibriaFactory()
    second_factory = EquilibriaFactory()

    first_builders = first_factory.get_builders()
    second_builders = second_factory.get_builders()

    assert first_builders["liquid_vapor"] is not second_builders["liquid_vapor"]


def test_get_strategy_liquid_vapor_returns_strategy(
    factory: EquilibriaFactory,
) -> None:
    """Default strategy is produced with default water_activity."""

    strategy = factory.get_strategy("liquid_vapor")

    assert isinstance(strategy, LiquidVaporPartitioningStrategy)
    assert strategy.water_activity == pytest.approx(0.5)


def test_get_strategy_with_parameters_passes_water_activity(
    factory: EquilibriaFactory,
) -> None:
    """Parameters flow into builder and strategy."""

    strategy = factory.get_strategy(
        "liquid_vapor", parameters={"water_activity": 0.8}
    )

    assert isinstance(strategy, LiquidVaporPartitioningStrategy)
    assert strategy.water_activity == pytest.approx(0.8)


def test_get_strategy_case_insensitive(factory: EquilibriaFactory) -> None:
    """Strategy lookup ignores case and underscores variance."""

    lower = factory.get_strategy("liquid_vapor")
    upper = factory.get_strategy("LIQUID_VAPOR")
    mixed = factory.get_strategy("Liquid_Vapor")

    assert all(
        isinstance(strategy, LiquidVaporPartitioningStrategy)
        for strategy in (lower, upper, mixed)
    )
    assert lower.water_activity == upper.water_activity == mixed.water_activity


def test_get_strategy_unknown_type_raises(factory: EquilibriaFactory) -> None:
    """Unknown strategy names raise ValueError with hint."""

    with pytest.raises(ValueError, match="Unknown strategy type"):
        factory.get_strategy("missing")


def test_get_strategy_invalid_parameters_raises(
    factory: EquilibriaFactory,
) -> None:
    """Invalid parameter values propagate as ValueError."""

    with pytest.raises(ValueError, match="water_activity must be in"):
        factory.get_strategy("liquid_vapor", parameters={"water_activity": 1.5})


def test_empty_parameters_defaults_applied(factory: EquilibriaFactory) -> None:
    """Empty parameter dict uses builder defaults."""

    strategy = factory.get_strategy("liquid_vapor", parameters={})

    assert isinstance(strategy, EquilibriaStrategy)
    assert strategy.water_activity == pytest.approx(0.5)


def test_none_parameters_defaults_applied(factory: EquilibriaFactory) -> None:
    """None parameters use defaults without error."""

    strategy = factory.get_strategy("liquid_vapor", parameters=None)

    assert isinstance(strategy, EquilibriaStrategy)
    assert strategy.water_activity == pytest.approx(0.5)


def test_non_dict_parameters_raises(factory: EquilibriaFactory) -> None:
    """Non-mapping parameter inputs raise type or value errors."""

    with pytest.raises((TypeError, ValueError)):
        factory.get_strategy("liquid_vapor", parameters="bad")
