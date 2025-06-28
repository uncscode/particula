"""Tests for the BrownianCoagulationBuilder class."""

import pytest

from particula.dynamics import (
    BrownianCoagulationBuilder,
    BrownianCoagulationStrategy,
)


def test_build_with_valid_parameters() -> None:
    """Test that building with valid parameters returns a
    BrownianCoagulationStrategy.
    """
    builder = BrownianCoagulationBuilder()
    builder.set_distribution_type("discrete")
    strategy = builder.build()
    assert isinstance(strategy, BrownianCoagulationStrategy)


def test_build_missing_required_parameters() -> None:
    """Test that building without required parameters raises a
    ValueError.
    """
    builder = BrownianCoagulationBuilder()
    with pytest.raises(ValueError):
        builder.build()
    with pytest.raises(ValueError):
        builder.set_distribution_type("NotRight")
