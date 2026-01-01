"""Tests for CondensationIsothermalStaggeredBuilder."""
# ruff: noqa: E402

import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("SCIPY_USE_CALC_DOCSTRINGS", "0")

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

lognormal_module: Any = types.ModuleType(
    "particula.particles.properties.lognormal_size_distribution"
)


def _stubbed_lognormal(*_args, **_kwargs):
    raise RuntimeError("lognormal size distribution stubbed for fast tests")


lognormal_module.get_lognormal_pdf_distribution = _stubbed_lognormal
lognormal_module.get_lognormal_pmf_distribution = _stubbed_lognormal
lognormal_module.get_lognormal_sample_distribution = _stubbed_lognormal
sys.modules["particula.particles.properties.lognormal_size_distribution"] = (
    lognormal_module
)

from particula.dynamics.condensation.condensation_builder import (
    CondensationIsothermalStaggeredBuilder,
)
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermalStaggered,
)


def test_builder_instantiation_defaults():
    """Builder initializes with staggered defaults."""
    builder = CondensationIsothermalStaggeredBuilder()

    assert builder.theta_mode == "half"
    assert builder.num_batches == 1
    assert builder.shuffle_each_step is True
    assert builder.random_state is None


def test_set_theta_mode_valid_modes():
    """Valid theta modes are accepted and chainable."""
    builder = CondensationIsothermalStaggeredBuilder()

    for mode in CondensationIsothermalStaggered.VALID_THETA_MODES:
        result = builder.set_theta_mode(mode)
        assert result is builder
        assert builder.theta_mode == mode


def test_set_theta_mode_invalid_raises():
    """Invalid theta mode raises ValueError with message."""
    builder = CondensationIsothermalStaggeredBuilder()

    with pytest.raises(ValueError, match="theta_mode must be one of"):
        builder.set_theta_mode("invalid")


def test_set_num_batches_valid():
    """Setting num_batches succeeds when value is positive."""
    builder = CondensationIsothermalStaggeredBuilder()

    result = builder.set_num_batches(3)

    assert result is builder
    assert builder.num_batches == 3


@pytest.mark.parametrize("value", [0, -1])
def test_set_num_batches_invalid(value):
    """Invalid num_batches values raise ValueError."""
    builder = CondensationIsothermalStaggeredBuilder()

    with pytest.raises(ValueError, match="num_batches must be >= 1."):
        builder.set_num_batches(value)


def test_set_shuffle_each_step_toggle():
    """Shuffle flag can be toggled and chained."""
    builder = CondensationIsothermalStaggeredBuilder()

    result = builder.set_shuffle_each_step(False)
    assert result is builder
    assert builder.shuffle_each_step is False

    builder.set_shuffle_each_step(True)
    assert builder.shuffle_each_step is True


@pytest.mark.parametrize("seed", [42, None])
def test_set_random_state_values(seed):
    """Random state accepts integers or None and is chainable."""
    builder = CondensationIsothermalStaggeredBuilder()

    result = builder.set_random_state(seed)

    assert result is builder
    assert builder.random_state == seed


def test_method_chaining_and_build_returns_strategy():
    """Chaining setters produces configured strategy instance."""
    builder = CondensationIsothermalStaggeredBuilder()

    strategy = (
        builder.set_molar_mass(0.018, "kg/mol")
        .set_diffusion_coefficient(2e-5, "m^2/s")
        .set_accommodation_coefficient(1.0)
        .set_theta_mode("batch")
        .set_num_batches(2)
        .set_shuffle_each_step(False)
        .set_random_state(7)
        .build()
    )

    assert isinstance(strategy, CondensationIsothermalStaggered)
    assert strategy.theta_mode == "batch"
    assert strategy.num_batches == 2
    assert strategy.shuffle_each_step is False
    assert strategy.random_state == 7


def test_build_without_required_parameters_raises():
    """Build fails when required parameters are missing."""
    builder = CondensationIsothermalStaggeredBuilder()

    with pytest.raises(ValueError, match=r"Required parameter\(s\) not set"):
        builder.build()
