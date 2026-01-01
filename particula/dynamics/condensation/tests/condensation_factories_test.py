"""Test cases for CondensationFactory and CondensationIsothermal classes."""

import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("SCIPY_USE_CALC_DOCSTRINGS", "0")

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

lognormal_module = types.ModuleType(
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

from particula.dynamics.condensation.condensation_factories import (
    CondensationFactory,
)
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
    CondensationIsothermalStaggered,
)


def test_isothermal_condensation():
    """Test the creation of an isothermal condensation strategy."""
    factory = CondensationFactory()
    strategy = factory.get_strategy(
        "isothermal",
        {
            "molar_mass": 0.018,
            "molar_mass_units": "kg/mol",
            "diffusion_coefficient": 2e-5,
            "diffusion_coefficient_units": "m^2/s",
            "accommodation_coefficient": 1.0,
        },
    )
    assert isinstance(strategy, CondensationIsothermal)


def test_isothermal_staggered_condensation_defaults():
    """Factory exposes staggered builder mapping and defaults work."""
    factory = CondensationFactory()
    builder_map = factory.get_builders()

    assert "isothermal_staggered" in builder_map

    strategy = (
        builder_map["isothermal_staggered"]
        .set_molar_mass(0.018, "kg/mol")
        .set_diffusion_coefficient(2e-5, "m^2/s")
        .set_accommodation_coefficient(1.0)
        .build()
    )

    assert isinstance(strategy, CondensationIsothermalStaggered)
    assert strategy.theta_mode == "half"
    assert strategy.num_batches == 1
    assert strategy.shuffle_each_step is True
    assert strategy.random_state is None


def test_invalid_condensation_strategy():
    """Test that an invalid condensation strategy raises a ValueError."""
    factory = CondensationFactory()
    with pytest.raises(ValueError):
        factory.get_strategy("nonexistent", {})
