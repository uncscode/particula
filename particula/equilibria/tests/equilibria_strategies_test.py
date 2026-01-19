"""Tests for equilibria strategies and liquid-vapor wrapper."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest
from numpy.testing import assert_allclose
from particula.equilibria import equilibria_strategies as strategies
from particula.equilibria import partitioning
from particula.equilibria.partitioning import PhaseOutput, SystemOutput
from particula.equilibria.tests.partitioning_test import (
    _build_partitioning_inputs,
)
from scipy.optimize import OptimizeResult


@pytest.fixture
def inputs():
    """Generate partitioning inputs for equilibria strategy tests."""
    return _build_partitioning_inputs(species_count=4, water_activity=0.65)


def _run_partitioning_for_compare(
    inputs,
) -> tuple[PhaseOutput, Optional[PhaseOutput], SystemOutput, OptimizeResult]:
    """Run the partitioning routine to compare against the strategy result."""
    (
        c_star_j_dry,
        concentration_organic_matter,
        molar_mass,
        _oxygen2carbon,
        _density,
        gamma_organic_ab,
        mass_fraction_water_ab,
        q_ab,
    ) = inputs
    return partitioning.liquid_vapor_partitioning(
        c_star_j_dry=c_star_j_dry,
        concentration_organic_matter=concentration_organic_matter,
        molar_mass=molar_mass,
        gamma_organic_ab=gamma_organic_ab,
        mass_fraction_water_ab=mass_fraction_water_ab,
        q_ab=q_ab,
    )


def test_get_name_returns_class_name():
    """Ensure get_name() returns the concrete strategy identifier."""
    strategy = strategies.LiquidVaporPartitioningStrategy()
    assert strategy.get_name() == "LiquidVaporPartitioningStrategy"


def test_default_and_custom_water_activity():
    """Verify the default and custom water activity settings."""
    default_strategy = strategies.LiquidVaporPartitioningStrategy()
    assert default_strategy.water_activity == pytest.approx(0.5)

    custom_strategy = strategies.LiquidVaporPartitioningStrategy(0.8)
    assert custom_strategy.water_activity == pytest.approx(0.8)


def test_invalid_water_activity_raises():
    """Ensure invalid water activity values are rejected."""
    with pytest.raises(ValueError, match=r"water_activity must be in \[0, 1\]"):
        strategies.LiquidVaporPartitioningStrategy(-0.1)
    with pytest.raises(ValueError, match=r"water_activity must be in \[0, 1\]"):
        strategies.LiquidVaporPartitioningStrategy(1.1)


def test_phase_concentrations_dataclass_fields():
    """Verify PhaseConcentrations stores species, water, and total data."""
    phase = strategies.PhaseConcentrations(
        species_concentrations=np.array([1.0, 2.0]),
        water_concentration=0.3,
        total_concentration=3.0,
    )
    assert phase.species_concentrations.shape == (2,)
    assert phase.water_concentration == pytest.approx(0.3)
    assert phase.total_concentration == pytest.approx(3.0)


def test_equilibrium_result_mapping_shapes(inputs):
    """Ensure solve returns results with shapes aligned to inputs."""
    (
        c_star_j_dry,
        concentration_organic_matter,
        molar_mass,
        oxygen2carbon,
        density,
        _gamma,
        _mass_fraction,
        _q_ab,
    ) = inputs
    strategy = strategies.LiquidVaporPartitioningStrategy()
    result = strategy.solve(
        c_star_j_dry=c_star_j_dry,
        concentration_organic_matter=concentration_organic_matter,
        molar_mass=molar_mass,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )

    assert isinstance(result, strategies.EquilibriumResult)
    assert result.alpha_phase.species_concentrations.shape == c_star_j_dry.shape
    if result.beta_phase is not None:
        assert (
            result.beta_phase.species_concentrations.shape == c_star_j_dry.shape
        )
    assert result.partition_coefficients.shape == c_star_j_dry.shape
    assert np.isfinite(result.error)


def test_solve_matches_partitioning_outputs(inputs):
    """Compare strategy output to helper partitioning results."""
    (
        c_star_j_dry,
        concentration_organic_matter,
        molar_mass,
        oxygen2carbon,
        density,
        _gamma,
        _mass_fraction,
        _q_ab,
    ) = inputs

    strategy = strategies.LiquidVaporPartitioningStrategy(water_activity=0.65)
    strategy_result = strategy.solve(
        c_star_j_dry=c_star_j_dry,
        concentration_organic_matter=concentration_organic_matter,
        molar_mass=molar_mass,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )

    alpha, beta, system, fit_result = _run_partitioning_for_compare(inputs)
    direct_result = strategy._convert_to_result(
        alpha=alpha,
        beta=beta,
        system=system,
        error_value=float(fit_result.fun),
    )

    assert_allclose(
        strategy_result.partition_coefficients,
        direct_result.partition_coefficients,
        rtol=1e-6,
        atol=1e-9,
    )
    assert_allclose(
        strategy_result.alpha_phase.total_concentration,
        direct_result.alpha_phase.total_concentration,
    )


def test_solve_handles_beta_none(monkeypatch, inputs):
    """Confirm solver returns `None` beta phase when partitioning omits it."""
    (
        c_star_j_dry,
        concentration_organic_matter,
        molar_mass,
        oxygen2carbon,
        density,
        _gamma,
        _mass_fraction,
        _q_ab,
    ) = inputs

    def _no_beta_partition(*args, **kwargs):
        alpha, _beta, system = partitioning.liquid_vapor_obj_function(
            e_j_partition_guess=np.full_like(c_star_j_dry, 0.5),
            c_star_j_dry=c_star_j_dry,
            concentration_organic_matter=concentration_organic_matter,
            gamma_organic_ab=np.ones((c_star_j_dry.size, 2)),
            mass_fraction_water_ab=np.zeros((c_star_j_dry.size, 2)),
            q_ab=np.column_stack(
                (np.ones(c_star_j_dry.size), np.zeros(c_star_j_dry.size))
            ),
            molar_mass=molar_mass,
            error_only=False,
        )
        return alpha, None, system, type("FitResult", (), {"fun": 0.0})()

    monkeypatch.setattr(
        strategies.partitioning, "liquid_vapor_partitioning", _no_beta_partition
    )

    strategy = strategies.LiquidVaporPartitioningStrategy()
    result = strategy.solve(
        c_star_j_dry=c_star_j_dry,
        concentration_organic_matter=concentration_organic_matter,
        molar_mass=molar_mass,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )

    assert result.beta_phase is None
    assert result.water_content[1] == pytest.approx(0.0)


def test_convert_to_result_rejects_malformed_tuple(inputs):
    """Reject malformed phase/system tuples in the conversion helper."""
    (
        c_star_j_dry,
        concentration_organic_matter,
        molar_mass,
        oxygen2carbon,
        density,
        _gamma,
        _mass_fraction,
        _q_ab,
    ) = inputs
    strategy = strategies.LiquidVaporPartitioningStrategy()

    alpha, beta, system, _ = _run_partitioning_for_compare(inputs)
    with pytest.raises(ValueError):
        strategy._convert_to_result(
            alpha=alpha, beta=beta, system=system[:3], error_value=0.0
        )
    with pytest.raises(ValueError):
        strategy._convert_to_result(
            alpha=alpha[:3], beta=beta, system=system, error_value=0.0
        )


def test_empty_inputs_raise():
    """Reject empty arrays as invalid inputs."""
    strategy = strategies.LiquidVaporPartitioningStrategy()
    with pytest.raises(ValueError):
        strategy.solve(
            c_star_j_dry=np.array([]),
            concentration_organic_matter=np.array([]),
            molar_mass=np.array([]),
            oxygen2carbon=np.array([]),
            density=np.array([]),
        )


def test_error_field_propagated(monkeypatch, inputs):
    """Ensure error field from partitioning surfaces in the result."""
    (
        c_star_j_dry,
        concentration_organic_matter,
        molar_mass,
        oxygen2carbon,
        density,
        _gamma,
        _mass_fraction,
        _q_ab,
    ) = inputs

    def _fake_partitioning(**kwargs):
        alpha, beta, system = partitioning.liquid_vapor_obj_function(
            e_j_partition_guess=np.full_like(c_star_j_dry, 0.5),
            c_star_j_dry=c_star_j_dry,
            concentration_organic_matter=concentration_organic_matter,
            gamma_organic_ab=np.ones((c_star_j_dry.size, 2)),
            mass_fraction_water_ab=np.zeros((c_star_j_dry.size, 2)),
            q_ab=np.full((c_star_j_dry.size, 2), 0.5),
            molar_mass=molar_mass,
            error_only=False,
        )
        system_out: SystemOutput = (
            system[0],
            system[1],
            np.ones_like(c_star_j_dry),
            0.123,
        )
        return alpha, beta, system_out, type("FitResult", (), {"fun": 0.0})()

    monkeypatch.setattr(
        strategies.partitioning, "liquid_vapor_partitioning", _fake_partitioning
    )

    strategy = strategies.LiquidVaporPartitioningStrategy()
    result = strategy.solve(
        c_star_j_dry=c_star_j_dry,
        concentration_organic_matter=concentration_organic_matter,
        molar_mass=molar_mass,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )

    assert result.error == pytest.approx(0.123)
    assert np.all(result.partition_coefficients == 1.0)
