"""Tests for condensation factory strategy registration and construction."""
# ruff: noqa: E402

import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("SCIPY_USE_CALC_DOCSTRINGS", "0")

ROOT = Path(__file__).resolve().parents[4]
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
    CondensationIsothermalBuilder,
    CondensationIsothermalStaggeredBuilder,
    CondensationLatentHeatBuilder,
)
from particula.dynamics.condensation.condensation_factories import (
    CondensationFactory,
)
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
    CondensationIsothermalStaggered,
    CondensationLatentHeat,
)
from particula.gas.latent_heat_strategies import ConstantLatentHeat


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


def test_get_builders_returns_expected_builder_instances() -> None:
    """Factory exposes the expected condensation builders."""
    factory = CondensationFactory()

    builder_map = factory.get_builders()

    assert set(builder_map) == {
        "isothermal",
        "isothermal_staggered",
        "latent_heat",
    }
    assert isinstance(builder_map["isothermal"], CondensationIsothermalBuilder)
    assert isinstance(
        builder_map["isothermal_staggered"],
        CondensationIsothermalStaggeredBuilder,
    )
    assert isinstance(
        builder_map["latent_heat"],
        CondensationLatentHeatBuilder,
    )


def test_isothermal_staggered_condensation_defaults_via_factory():
    """Factory returns staggered strategy with default parameters set."""
    factory = CondensationFactory()
    builder_map = factory.get_builders()

    assert "isothermal_staggered" in builder_map

    strategy = factory.get_strategy(
        "isothermal_staggered",
        {
            "molar_mass": 0.018,
            "molar_mass_units": "kg/mol",
            "diffusion_coefficient": 2e-5,
            "diffusion_coefficient_units": "m^2/s",
            "accommodation_coefficient": 1.0,
        },
    )

    assert isinstance(strategy, CondensationIsothermalStaggered)
    assert strategy.theta_mode == "half"
    assert strategy.num_batches == 1
    assert strategy.shuffle_each_step is True
    assert strategy.random_state is None
    assert strategy.update_gases is True


def test_isothermal_staggered_condensation_custom_parameters_via_factory():
    """Factory propagates non-default staggered parameters to the strategy."""
    factory = CondensationFactory()
    strategy = factory.get_strategy(
        "isothermal_staggered",
        {
            "molar_mass": 0.018,
            "molar_mass_units": "kg/mol",
            "diffusion_coefficient": 2e-5,
            "diffusion_coefficient_units": "m^2/s",
            "accommodation_coefficient": 1.0,
            "theta_mode": "batch",
            "num_batches": 3,
            "shuffle_each_step": False,
            "random_state": 42,
            "update_gases": False,
        },
    )

    assert isinstance(strategy, CondensationIsothermalStaggered)
    assert strategy.theta_mode == "batch"
    assert strategy.num_batches == 3
    assert strategy.shuffle_each_step is False
    assert strategy.random_state == 42
    assert strategy.update_gases is False


def test_latent_heat_condensation_with_strategy_via_factory() -> None:
    """Factory passes latent heat strategy objects through unchanged."""
    factory = CondensationFactory()
    latent_heat_strategy = ConstantLatentHeat(latent_heat_ref=2.26e6)

    strategy = factory.get_strategy(
        "latent_heat",
        {
            "molar_mass": 0.018,
            "molar_mass_units": "kg/mol",
            "diffusion_coefficient": 2e-5,
            "diffusion_coefficient_units": "m^2/s",
            "accommodation_coefficient": 1.0,
            "latent_heat_strategy": latent_heat_strategy,
            "update_gases": False,
        },
    )

    assert isinstance(strategy, CondensationLatentHeat)
    assert strategy.latent_heat_strategy_input is latent_heat_strategy
    assert strategy._latent_heat_strategy is latent_heat_strategy
    assert strategy.update_gases is False


def test_latent_heat_condensation_scalar_fallback_via_factory() -> None:
    """Factory supports scalar latent heat fallback through the builder."""
    factory = CondensationFactory()

    strategy = factory.get_strategy(
        "latent_heat",
        {
            "molar_mass": 0.018,
            "molar_mass_units": "kg/mol",
            "diffusion_coefficient": 2e-5,
            "diffusion_coefficient_units": "m^2/s",
            "accommodation_coefficient": 1.0,
            "latent_heat": 2.26e6,
        },
    )

    assert isinstance(strategy, CondensationLatentHeat)
    assert strategy.latent_heat_input == pytest.approx(2.26e6)
    assert strategy._latent_heat_strategy is not None
    assert strategy.update_gases is True


def test_latent_heat_condensation_strategy_precedence_via_factory() -> None:
    """Explicit latent heat strategy takes precedence over scalar fallback."""
    factory = CondensationFactory()
    latent_heat_strategy = ConstantLatentHeat(latent_heat_ref=1.0e6)

    strategy = factory.get_strategy(
        "latent_heat",
        {
            "molar_mass": 0.018,
            "molar_mass_units": "kg/mol",
            "diffusion_coefficient": 2e-5,
            "diffusion_coefficient_units": "m^2/s",
            "accommodation_coefficient": 1.0,
            "latent_heat_strategy": latent_heat_strategy,
            "latent_heat": 2.26e6,
        },
    )

    assert isinstance(strategy, CondensationLatentHeat)
    assert strategy.latent_heat_strategy_input is latent_heat_strategy
    assert strategy._latent_heat_strategy is latent_heat_strategy
    assert strategy.latent_heat_input == pytest.approx(2.26e6)


def test_latent_heat_condensation_invalid_parameters_propagate_builder_error(  # noqa: E501
) -> None:
    """Factory surfaces builder parameter validation failures unchanged."""
    factory = CondensationFactory()

    with pytest.raises(ValueError, match="invalid parameter"):
        factory.get_strategy(
            "latent_heat",
            {
                "molar_mass": 0.018,
                "molar_mass_units": "kg/mol",
                "diffusion_coefficient": 2e-5,
                "diffusion_coefficient_units": "m^2/s",
                "accommodation_coefficient": 1.0,
                "unexpected": 1,
            },
        )


def test_invalid_condensation_strategy():
    """Test that an invalid condensation strategy raises a ValueError."""
    factory = CondensationFactory()
    with pytest.raises(ValueError, match="Unknown strategy type: nonexistent"):
        factory.get_strategy("nonexistent", {})
