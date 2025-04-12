"""
Tests for the SedimentationCoagulationBuilder class.
"""

import pytest
from particula.dynamics.coagulation.coagulation_builder.sedimentation_coagulation_builder import (
    SedimentationCoagulationBuilder,
)
from particula.dynamics.coagulation.coagulation_strategy.sedimentation_coagulation_strategy import (
    SedimentationCoagulationStrategy,
)


def test_build_with_valid_parameters():
    """
    Check that building with a valid distribution type
    returns a SedimentationCoagulationStrategy instance.
    """
    builder = SedimentationCoagulationBuilder()
    builder.set_distribution_type("discrete")
    strategy = builder.build()
    assert isinstance(strategy, SedimentationCoagulationStrategy)


def test_build_missing_required_parameters():
    """
    Ensure that missing required parameters or an invalid
    distribution type raises a ValueError.
    """
    builder = SedimentationCoagulationBuilder()
    with pytest.raises(ValueError):
        builder.build()
    with pytest.raises(ValueError):
        builder.set_distribution_type("NotRight")
