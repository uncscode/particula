"""Tests for the ChargedCoagulationBuilder class."""

import pytest

from particula.dynamics import (
    ChargedCoagulationBuilder,
    ChargedCoagulationStrategy,
    HardSphereKernelStrategy,
)


def test_build_with_valid_parameters():
    """Test that building ChargedCoagulationStrategy."""
    builder = ChargedCoagulationBuilder()
    builder.set_distribution_type(
        "discrete", distribution_type_units="dimensionless"
    )
    builder.set_charged_kernel_strategy(HardSphereKernelStrategy())
    strategy = builder.build()
    assert isinstance(strategy, ChargedCoagulationStrategy)


def test_build_missing_required_parameters():
    """Test that building without required parameters raises a ValueError."""
    builder = ChargedCoagulationBuilder()

    # Missing distribution_type and charged_kernel_strategy
    with pytest.raises(ValueError):
        builder.build()

    # Provide distribution_type, still missing charged_kernel_strategy
    builder.set_distribution_type(
        "discrete", distribution_type_units="dimensionless"
    )
    with pytest.raises(ValueError):
        builder.build()

    # Provide the kernel strategy -> should build successfully
    builder.set_charged_kernel_strategy(HardSphereKernelStrategy())
    strategy = builder.build()
    assert isinstance(strategy, ChargedCoagulationStrategy)
