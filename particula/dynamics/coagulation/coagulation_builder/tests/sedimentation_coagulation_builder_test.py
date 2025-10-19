"""Tests for the SedimentationCoagulationBuilder class."""

import pytest

from particula.dynamics.coagulation.coagulation_builder import (
    sedimentation_coagulation_builder,
)
from particula.dynamics.coagulation.coagulation_strategy import (
    sedimentation_coagulation_strategy,
)


def test_build_with_valid_parameters():
    """Check that building with a valid distribution type
    returns a SedimentationCoagulationStrategy instance.
    """
    builder_class = (
        sedimentation_coagulation_builder.SedimentationCoagulationBuilder
    )
    builder = builder_class()
    builder.set_distribution_type("discrete")
    strategy = builder.build()
    strategy_class = (
        sedimentation_coagulation_strategy.SedimentationCoagulationStrategy
    )
    assert isinstance(strategy, strategy_class)


def test_build_missing_required_parameters():
    """Ensure that missing required parameters or an invalid
    distribution type raises a ValueError.
    """
    builder_class = (
        sedimentation_coagulation_builder.SedimentationCoagulationBuilder
    )
    builder = builder_class()
    with pytest.raises(ValueError):
        builder.build()
    with pytest.raises(ValueError):
        builder.set_distribution_type("NotRight")
