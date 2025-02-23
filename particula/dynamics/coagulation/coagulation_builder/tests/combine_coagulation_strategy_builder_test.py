"""
Tests for the CombineCoagulationStrategyBuilder class.
"""

import pytest

from particula.dynamics import (
    CombineCoagulationStrategyBuilder,
    CombineCoagulationStrategy,
    BrownianCoagulationBuilder,
    ChargedCoagulationBuilder,
    HardSphereKernelStrategy,
)


def test_build_with_valid_strategies():
    """
    Test that CombineCoagulationStrategyBuilder with valid sub-strategies
    returns a CombineCoagulationStrategy.
    """
    # Example sub-strategies:
    brownian_strategy = (
        BrownianCoagulationBuilder()
        .set_distribution_type(
            "discrete", distribution_type_units="dimensionless"
        )
        .build()
    )
    charged_strategy = (
        ChargedCoagulationBuilder()
        .set_distribution_type(
            "discrete", distribution_type_units="dimensionless"
        )
        .set_charged_kernel_strategy(HardSphereKernelStrategy())
        .build()
    )

    builder = CombineCoagulationStrategyBuilder()
    builder.set_strategies([brownian_strategy, charged_strategy])
    combined = builder.build()

    assert isinstance(combined, CombineCoagulationStrategy)
    assert len(combined.strategies) == 2


def test_build_missing_strategies():
    """
    Test that CombineCoagulationStrategyBuilder without required 'strategies'
    raises a ValueError.
    """
    builder = CombineCoagulationStrategyBuilder()
    with pytest.raises(ValueError):
        builder.build()
