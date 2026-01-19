"""Tests for :class:`LiquidVaporPartitioningBuilder`."""

from __future__ import annotations

import pytest
from particula.equilibria.equilibria_builders import (
    LiquidVaporPartitioningBuilder,
)
from particula.equilibria.equilibria_strategies import (
    LiquidVaporPartitioningStrategy,
)


def test_default_build_uses_default_water_activity():
    """Ensure the builder produces the default water activity."""
    builder = LiquidVaporPartitioningBuilder()
    strategy = builder.build()

    assert isinstance(strategy, LiquidVaporPartitioningStrategy)
    assert strategy.water_activity == pytest.approx(0.5)


def test_set_water_activity_happy_path_accepts_float_and_int():
    """Validate fluent setter accepts ints and floats."""
    builder = LiquidVaporPartitioningBuilder()
    strategy_float = builder.set_water_activity(0.75).build()
    strategy_int = builder.set_water_activity(1).build()

    assert strategy_float.water_activity == pytest.approx(0.75)
    assert strategy_int.water_activity == pytest.approx(1.0)


def test_set_water_activity_boundaries_and_chaining():
    """Cover the [0, 1] boundaries and method chaining."""
    strategy_zero = (
        LiquidVaporPartitioningBuilder().set_water_activity(0.0).build()
    )
    strategy_one = (
        LiquidVaporPartitioningBuilder().set_water_activity(1.0).build()
    )
    strategy_chain = (
        LiquidVaporPartitioningBuilder()
        .set_water_activity(0.6)
        .set_water_activity(0.4)
        .build()
    )

    assert strategy_zero.water_activity == pytest.approx(0.0)
    assert strategy_one.water_activity == pytest.approx(1.0)
    assert strategy_chain.water_activity == pytest.approx(0.4)


def test_set_water_activity_invalid_values_raise():
    """Reject values outside the inclusive range."""
    builder = LiquidVaporPartitioningBuilder()
    with pytest.raises(ValueError, match="water_activity must be in"):
        builder.set_water_activity(-0.01)
    with pytest.raises(ValueError, match="water_activity must be in"):
        builder.set_water_activity(1.01)


def test_set_parameters_applies_dict_and_rejects_unknown():
    """Confirm dict inputs map to setters and unknown keys fail."""
    builder = LiquidVaporPartitioningBuilder()
    strategy_dict = builder.set_parameters({"water_activity": 0.7}).build()
    strategy_empty = LiquidVaporPartitioningBuilder().set_parameters({}).build()

    assert strategy_dict.water_activity == pytest.approx(0.7)
    assert strategy_empty.water_activity == pytest.approx(0.5)

    with pytest.raises(ValueError, match=r"Invalid parameter\(s\)"):
        builder.set_parameters({"water_activity_units": "%"})
    with pytest.raises(ValueError, match=r"Invalid parameter\(s\)"):
        builder.set_parameters({"unexpected": 0.5})


def test_set_parameters_invalid_type_raises():
    """Non-numeric water_activity inputs raise during coercion."""
    builder = LiquidVaporPartitioningBuilder()
    with pytest.raises((TypeError, ValueError)):
        builder.set_parameters({"water_activity": "bad"})


def test_builder_reuse_multiple_builds():
    """Builders can be reused and mutate state between builds."""
    builder = LiquidVaporPartitioningBuilder()
    strategy_first = builder.set_water_activity(0.3).build()
    strategy_second = builder.set_water_activity(0.9).build()

    assert strategy_first.water_activity == pytest.approx(0.3)
    assert strategy_second.water_activity == pytest.approx(0.9)


def test_pre_build_check_raises_when_water_activity_none():
    """pre_build_check requires water_activity to be set before building."""
    builder = LiquidVaporPartitioningBuilder()
    builder._water_activity = None  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="water_activity must be set"):
        builder.build()
