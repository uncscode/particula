"""Tests for fresh P4 nucleation factory dispatch."""

import pytest
from particula.dynamics.nucleation import (
    ActivationNucleationBuilder,
    KineticNucleationStrategy,
    NucleationFactory,
)


def _parameters() -> dict[str, object]:
    return {
        "coefficient": 2.0,
        "coefficient_units": "s^-1",
        "coefficient_provenance": "laboratory fit",
        "precursor_number_concentration_lower": 3.0,
        "precursor_number_concentration_lower_units": "1/cm^3",
        "precursor_number_concentration_upper": 4.0,
        "precursor_number_concentration_upper_units": "1/cm^3",
        "temperature_lower": 250.0,
        "temperature_lower_units": "K",
        "temperature_upper": 300.0,
        "temperature_upper_units": "K",
        "injection_composition": [1, 2],
        "formation_diameter": 2.0,
        "formation_diameter_units": "nm",
    }


def test_factory_has_only_fresh_canonical_builders() -> None:
    """Each factory query returns new builders under exactly two keys."""
    factory = NucleationFactory()
    first = factory.get_builders()
    second = factory.get_builders()
    assert set(first) == {"activation", "kinetic"}
    assert isinstance(first["activation"], ActivationNucleationBuilder)
    assert first["activation"] is not second["activation"]


def test_factory_dispatches_case_insensitively_without_mutating_mapping() -> (
    None
):
    """Factory copies caller parameters and returns the selected strategy."""
    parameters = _parameters()
    parameters["coefficient_units"] = "m^3/s"
    original = parameters.copy()
    strategy = NucleationFactory().get_strategy("KiNeTiC", parameters)
    assert isinstance(strategy, KineticNucleationStrategy)
    assert parameters == original


@pytest.mark.parametrize(
    "strategy_type", [None, 3, "unknown", "activation_alias"]
)
def test_factory_rejects_unknown_or_invalid_identifiers(
    strategy_type: object,
) -> None:
    """Only string activation and kinetic identifiers are accepted."""
    with pytest.raises(ValueError):
        NucleationFactory().get_strategy(strategy_type, _parameters())  # type: ignore[arg-type]


def test_rejected_factory_call_does_not_contaminate_later_call() -> None:
    """A failed fresh builder is discarded before the next factory call."""
    factory = NucleationFactory()
    bad = _parameters()
    bad["coefficient_units"] = "bad"
    with pytest.raises(ValueError, match="coefficient_units"):
        factory.get_strategy("activation", bad)
    assert factory.get_strategy("activation", _parameters()).coefficient == 2.0


@pytest.mark.parametrize("parameters", [None, [], {}, {"coefficient": 1.0}])
def test_factory_rejects_non_dict_or_incomplete_parameters(
    parameters: object,
) -> None:
    """Factory requires a complete strict dictionary payload."""
    with pytest.raises(ValueError):
        NucleationFactory().get_strategy("activation", parameters)  # type: ignore[arg-type]
