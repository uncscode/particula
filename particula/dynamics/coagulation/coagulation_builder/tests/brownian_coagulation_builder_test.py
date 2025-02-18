import pytest
from particula.dynamics.coagulation.coagulation_builder.brownian_coagulation_builder import (
    BrownianCoagulationBuilder,
)
from particula.dynamics.coagulation.coagulation_strategy import (
    BrownianCoagulationStrategy,
)


def test_build_with_valid_parameters():
    """
    Test that building with valid parameters returns a BrownianCoagulationStrategy.
    """
    builder = BrownianCoagulationBuilder()
    builder.set_distribution_type("discrete")
    strategy = builder.build()
    assert isinstance(strategy, BrownianCoagulationStrategy)


def test_build_missing_required_parameters():
    """
    Test that building without required parameters raises a ValueError.
    """
    builder = BrownianCoagulationBuilder()
    with pytest.raises(ValueError):
        builder.build()
