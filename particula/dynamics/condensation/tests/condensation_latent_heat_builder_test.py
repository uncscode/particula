"""Tests for the latent-heat condensation builder."""

import numpy as np
import pytest
from particula.dynamics.condensation.condensation_builder import (
    CondensationLatentHeatBuilder,
)
from particula.dynamics.condensation.condensation_strategies import (
    CondensationLatentHeat,
)
from particula.gas.latent_heat_strategies import ConstantLatentHeat


def _make_builder() -> CondensationLatentHeatBuilder:
    """Return a builder with required shared inputs configured."""
    return (
        CondensationLatentHeatBuilder()
        .set_molar_mass(0.018, "kg/mol")
        .set_diffusion_coefficient(2e-5, "m^2/s")
        .set_accommodation_coefficient(1.0)
    )


def test_build_with_latent_heat_strategy_returns_condensation_latent_heat() -> (
    None
):
    """Builder returns CondensationLatentHeat when given a strategy."""
    latent_heat_strategy = ConstantLatentHeat(latent_heat_ref=2.26e6)

    strategy = (
        _make_builder().set_latent_heat_strategy(latent_heat_strategy).build()
    )

    assert isinstance(strategy, CondensationLatentHeat)
    assert strategy._latent_heat_strategy is latent_heat_strategy


def test_build_missing_required_parameters_raises_value_error() -> None:
    """Build fails when required shared parameters were not set."""
    with pytest.raises(
        ValueError,
        match="Required parameter",
    ):
        CondensationLatentHeatBuilder().build()


def test_set_latent_heat_strategy_passes_strategy_through() -> None:
    """Latent heat strategy object is forwarded unchanged."""
    latent_heat_strategy = ConstantLatentHeat(latent_heat_ref=2.26e6)
    builder = _make_builder().set_latent_heat_strategy(latent_heat_strategy)

    strategy = builder.build()

    assert strategy._latent_heat_strategy is latent_heat_strategy
    assert strategy.latent_heat_strategy_input is latent_heat_strategy


def test_set_latent_heat_accepts_positive_scalar() -> None:
    """Positive scalar latent heat is normalized and forwarded."""
    strategy = _make_builder().set_latent_heat(2.26e6).build()

    assert strategy.latent_heat_input == pytest.approx(2.26e6)
    assert strategy._latent_heat_strategy is not None


def test_set_latent_heat_rejects_none() -> None:
    """None is rejected at the builder boundary."""
    with pytest.raises(
        ValueError,
        match="latent_heat must be a positive finite scalar, got None",
    ):
        _make_builder().set_latent_heat(None)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (
            np.array([2.26e6, 1.5e6]),
            (
                "latent_heat must be a positive finite scalar, got "
                "array-like value"
            ),
        ),
        (
            np.nan,
            (
                "latent_heat must be a positive finite scalar, got "
                "non-finite value"
            ),
        ),
        (
            np.inf,
            (
                "latent_heat must be a positive finite scalar, got "
                "non-finite value"
            ),
        ),
        (
            0.0,
            (
                "latent_heat must be a positive finite scalar, got "
                "non-positive value"
            ),
        ),
        (
            -1.0,
            (
                "latent_heat must be a positive finite scalar, got "
                "non-positive value"
            ),
        ),
    ],
)
def test_set_latent_heat_rejects_invalid_values(
    value: float | np.ndarray,
    message: str,
) -> None:
    """Array-like, non-finite, and non-positive latent heat are rejected."""
    with pytest.raises(ValueError, match=message):
        _make_builder().set_latent_heat(value)


def test_update_gases_defaults_true_and_can_be_overridden() -> None:
    """Update-gases behavior defaults to True and propagates overrides."""
    default_strategy = _make_builder().build()
    override_strategy = _make_builder().set_update_gases(False).build()

    assert default_strategy.update_gases is True
    assert override_strategy.update_gases is False


def test_set_parameters_accepts_optional_latent_heat_keys() -> None:
    """set_parameters handles optional latent heat and update keys."""
    latent_heat_strategy = ConstantLatentHeat(latent_heat_ref=2.26e6)
    strategy = (
        CondensationLatentHeatBuilder()
        .set_parameters(
            {
                "molar_mass": 0.018,
                "molar_mass_units": "kg/mol",
                "diffusion_coefficient": 2e-5,
                "diffusion_coefficient_units": "m^2/s",
                "accommodation_coefficient": 1.0,
                "latent_heat_strategy": latent_heat_strategy,
                "latent_heat": 2.26e6,
                "update_gases": False,
            }
        )
        .build()
    )

    assert strategy._latent_heat_strategy is latent_heat_strategy
    assert strategy.latent_heat_strategy_input is latent_heat_strategy
    assert strategy.latent_heat_input == pytest.approx(2.26e6)
    assert strategy.update_gases is False


def test_set_parameters_missing_required_key_raises_value_error() -> None:
    """set_parameters rejects missing required shared builder inputs."""
    with pytest.raises(
        ValueError,
        match="accommodation_coefficient",
    ):
        CondensationLatentHeatBuilder().set_parameters(
            {
                "molar_mass": 0.018,
                "molar_mass_units": "kg/mol",
                "diffusion_coefficient": 2e-5,
                "diffusion_coefficient_units": "m^2/s",
            }
        )


def test_set_parameters_rejects_unknown_keys() -> None:
    """Unknown set_parameters keys raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="invalid parameter"):
        CondensationLatentHeatBuilder().set_parameters(
            {
                "molar_mass": 0.018,
                "molar_mass_units": "kg/mol",
                "diffusion_coefficient": 2e-5,
                "diffusion_coefficient_units": "m^2/s",
                "accommodation_coefficient": 1.0,
                "unexpected": 1,
            }
        )


def test_build_with_strategy_and_scalar_preserves_constructor_precedence() -> (
    None
):
    """Explicit strategy remains active when scalar latent heat is also set."""
    latent_heat_strategy = ConstantLatentHeat(latent_heat_ref=1.0e6)
    strategy = (
        _make_builder()
        .set_latent_heat_strategy(latent_heat_strategy)
        .set_latent_heat(2.26e6)
        .build()
    )

    assert strategy._latent_heat_strategy is latent_heat_strategy
    assert strategy.latent_heat_input == pytest.approx(2.26e6)


def test_unset_latent_heat_uses_constructor_default_without_none_passthrough() -> (
    None
):
    """Unset scalar latent heat leaves constructor default state intact."""
    latent_heat_strategy = ConstantLatentHeat(latent_heat_ref=2.26e6)
    strategy = (
        _make_builder().set_latent_heat_strategy(latent_heat_strategy).build()
    )

    assert strategy._latent_heat_strategy is latent_heat_strategy
    assert strategy.latent_heat_input == 0.0


def test_builder_package_export_imports_cleanly() -> None:
    """Builder package export returns the new builder class."""
    from particula.dynamics.condensation.condensation_builder import (
        CondensationLatentHeatBuilder as ExportedBuilder,
    )

    assert ExportedBuilder is CondensationLatentHeatBuilder
