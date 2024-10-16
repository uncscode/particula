"""Test the BuilderABC and BuilderDensityMixin classes."""

# pylint: disable=redefined-outer-name

import pytest
from particula.abc_builder import BuilderABC


# Example of a concrete class extending BuilderABC for testing
class ConcreteBuilder(BuilderABC):
    """Concrete class extending BuilderABC for testing purposes."""

    def build(self):
        return "Build successful!"


# Setup the fixture for the testing
@pytest.fixture
def builder_fix():
    """Fixture for the BuilderABC class."""
    return ConcreteBuilder(required_parameters=["param1", "param2"])


def test_builder_init(builder_fix):
    """Test the initialization of the BuilderABC class."""
    assert builder_fix.required_parameters == [
        "param1",
        "param2",
    ], "Initialization of required parameters failed"


def test_check_keys_valid(builder_fix):
    """Check if the keys you want to set are present in the parameters"""
    parameters = {"param1": 10, "param2": 20, "param1_units": "meters"}
    try:
        builder_fix.check_keys(parameters)
    except ValueError as e:
        pytest.fail(f"Unexpected ValueError raised: {e}")


def test_check_keys_missing_required(builder_fix):
    """Check if the keys you want to set are present in the parameters"""
    parameters = {"param1": 10}  # missing 'param2'
    with pytest.raises(ValueError) as excinfo:
        builder_fix.check_keys(parameters)
    assert "Missing required parameter(s): param2" in str(excinfo.value)


def test_check_keys_invalid(builder_fix):
    """Check if the keys are invalid in the parameters dictionary."""
    parameters = {"param1": 10, "param2": 20, "param3": 30}
    with pytest.raises(ValueError) as excinfo:
        builder_fix.check_keys(parameters)
    assert "Trying to set an invalid parameter(s)" in str(excinfo.value)


def test_pre_build_check_complete(builder_fix):
    """Check if all required attribute parameters are set before building."""
    builder_fix.param1 = 10
    builder_fix.param2 = 20
    try:
        builder_fix.pre_build_check()
    except ValueError as e:
        pytest.fail(f"Unexpected ValueError raised: {e}")
