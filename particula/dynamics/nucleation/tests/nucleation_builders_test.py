"""Regression tests for strict P4 nucleation builders."""

import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from particula.dynamics.nucleation import (
    ActivationNucleationBuilder,
    ActivationNucleationStrategy,
    KineticNucleationBuilder,
    KineticNucleationStrategy,
    NucleationSourceConfig,
    NucleationSourceConfigBuilder,
    NucleationStrategy,
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


def test_activation_mapping_normalizes_and_defaults_survival() -> None:
    """Mapping configuration creates a normalized immutable strategy."""
    strategy = (
        ActivationNucleationBuilder().set_parameters(_parameters()).build()
    )

    assert isinstance(strategy, ActivationNucleationStrategy)
    assert strategy.coefficient == 2.0
    assert strategy.coefficient_provenance == "laboratory fit"
    assert strategy.validity_domain.precursor_number_concentration.lower == 3e6
    assert strategy.validity_domain.precursor_number_concentration.upper == 4e6
    assert strategy.formation_metadata.formation_diameter == 2e-9
    assert (
        strategy.formation_metadata.diameter_convention == "mobility_diameter"
    )
    assert strategy.survival_factor == 1.0
    with pytest.raises(FrozenInstanceError):
        strategy.coefficient = 1.0  # type: ignore[misc]


def test_kinetic_mapping_converts_coefficient_and_optional_intervals() -> None:
    """Kinetic builder applies the accepted kinetic and saturation units."""
    parameters = _parameters()
    parameters.update(
        {
            "coefficient_units": "cm^3/s",
            "saturation_lower": 1.0,
            "saturation_lower_units": "dimensionless",
            "saturation_upper": 2.0,
            "saturation_upper_units": "dimensionless",
            "survival_factor": 0.5,
            "survival_factor_units": "dimensionless",
        }
    )
    strategy = KineticNucleationBuilder().set_parameters(parameters).build()

    assert strategy.coefficient == 2e-6
    assert strategy.survival_factor == 0.5
    assert strategy.validity_domain.saturation is not None


def test_mapping_retains_an_explicit_zero_survival_factor() -> None:
    """An explicit zero survival factor survives mapping and yields no events."""
    parameters = _parameters()
    parameters.update(
        {
            "survival_factor": 0.0,
            "survival_factor_units": "dimensionless",
        }
    )

    strategy = ActivationNucleationBuilder().set_parameters(parameters).build()

    assert strategy.survival_factor == 0.0
    assert strategy.potential_rate(1.0e-12, 0.1, 400.0) == 0.0


@pytest.mark.parametrize(
    "setter,value,units,expected",
    [
        ("set_coefficient", 2.0, "cm^3/s", 2e-6),
        (
            "set_precursor_number_concentration_lower",
            3.0,
            "1/cm^3",
            3e6,
        ),
        ("set_formation_diameter", 2.0, "nm", 2e-9),
    ],
)
def test_kinetic_fluent_setters_normalize_accepted_units(
    setter: str,
    value: float,
    units: str,
    expected: float,
) -> None:
    """Each supported non-SI conversion stores its exact normalized value."""
    builder = KineticNucleationBuilder()
    getattr(builder, setter)(value, units)

    attribute = setter.removeprefix("set_")
    assert getattr(builder, attribute) == expected


@pytest.mark.parametrize(
    "change, message",
    [
        ({"coefficient_units": "1/s"}, "coefficient_units"),
        ({"temperature_lower_units": "C"}, "temperature_lower_units"),
        ({"formation_diameter_units": "um"}, "formation_diameter_units"),
        (
            {
                "saturation_lower": 1.0,
                "saturation_lower_units": "dimensionless",
            },
            "saturation_lower",
        ),
        ({"legacy": 1}, "Unknown"),
    ],
)
def test_mapping_rejects_units_schema_and_unpaired_saturation(
    change: dict[str, object], message: str
) -> None:
    """The strict mapping schema fails closed for invalid inputs."""
    parameters = _parameters()
    parameters.update(change)
    with pytest.raises(ValueError, match=message):
        ActivationNucleationBuilder().set_parameters(parameters)


@pytest.mark.parametrize(
    "builder_type, coefficient_units",
    [
        (ActivationNucleationBuilder, "s^-1"),
        (KineticNucleationBuilder, "m^3/s"),
    ],
)
def test_failed_mapping_is_atomic_and_builder_recovers(
    builder_type: type[ActivationNucleationBuilder | KineticNucleationBuilder],
    coefficient_units: str,
) -> None:
    """Invalid staged domains cannot alter a valid builder configuration."""
    parameters = _parameters()
    parameters["coefficient_units"] = coefficient_units
    builder = builder_type().set_parameters(parameters)
    baseline = builder.build()
    malformed = parameters.copy()
    malformed["temperature_lower"] = 301.0

    with pytest.raises(ValueError, match="less than or equal"):
        builder.set_parameters(malformed)

    assert builder.build() == baseline
    corrected = parameters.copy()
    corrected["coefficient"] = 3.0
    assert builder.set_parameters(corrected).build().coefficient == 3.0


@pytest.mark.parametrize(
    "setter, value, units, message",
    [
        (
            "set_precursor_number_concentration_lower",
            sys.float_info.max,
            "1/cm^3",
            "finite",
        ),
        ("set_formation_diameter", 5e-324, "nm", "remain positive"),
    ],
)
def test_unit_conversion_rejects_invalid_normalized_values(
    setter: str,
    value: float,
    units: str,
    message: str,
) -> None:
    """Conversion rejects overflow and positivity-destroying underflow."""
    with pytest.raises(ValueError, match=message):
        getattr(ActivationNucleationBuilder(), setter)(value, units)


@pytest.mark.parametrize("composition", [[1, 2], (1, 2)])
def test_injection_composition_accepts_ordered_sequences(
    composition: list[int] | tuple[int, int],
) -> None:
    """Ordered, bounded composition sequences are copied into the strategy."""
    strategy = (
        ActivationNucleationBuilder()
        .set_parameters({**_parameters(), "injection_composition": composition})
        .build()
    )

    assert strategy.injection_composition.molecule_counts == (1, 2)


@pytest.mark.parametrize(
    "composition", [(item for item in [1, 2]), {1: 2}, {1, 2}]
)
def test_injection_composition_rejects_unbounded_or_unordered_inputs(
    composition: object,
) -> None:
    """Composition cannot consume generators or accept unordered containers."""
    with pytest.raises(TypeError, match="sequence"):
        ActivationNucleationBuilder().set_injection_composition(composition)


def test_fluent_builder_requires_configuration_and_copies_composition() -> None:
    """Fluent construction validates required fields and owns composition data."""
    builder = ActivationNucleationBuilder()
    with pytest.raises(ValueError, match="Required parameter"):
        builder.build()
    composition = [1, 2]
    strategy = (
        builder.set_coefficient(1.0, "s^-1")
        .set_coefficient_provenance("fit")
        .set_precursor_number_concentration_lower(1.0, "1/m^3")
        .set_precursor_number_concentration_upper(2.0, "1/m^3")
        .set_temperature_lower(250.0, "K")
        .set_temperature_upper(300.0, "K")
        .set_injection_composition(composition)
        .set_formation_diameter(1.0, "m")
        .build()
    )
    composition[0] = 99
    assert strategy.injection_composition.molecule_counts == (1, 2)


def test_builder_reuse_creates_independent_nested_records() -> None:
    """Each build creates fresh immutable configuration records."""
    builder = ActivationNucleationBuilder().set_parameters(_parameters())

    first = builder.build()
    second = builder.build()

    assert first == second
    assert first is not second
    assert first.validity_domain is not second.validity_domain
    assert first.injection_composition is not second.injection_composition
    assert first.formation_metadata is not second.formation_metadata


def test_source_configuration_builder_is_metadata_only() -> None:
    """Source builder accepts a strategy and validated nonnegative index."""
    strategy = (
        ActivationNucleationBuilder().set_parameters(_parameters()).build()
    )
    config = (
        NucleationSourceConfigBuilder()
        .set_parameters({"strategy": strategy, "precursor_index": 0})
        .build()
    )
    assert config.strategy is strategy
    assert config.precursor_index == 0
    assert not hasattr(config, "finalize_particle_source")


@pytest.mark.parametrize(
    "builder_type,strategy_type",
    [
        (ActivationNucleationBuilder, ActivationNucleationStrategy),
        (KineticNucleationBuilder, KineticNucleationStrategy),
    ],
)
def test_source_configuration_record_accepts_supported_strategies(
    builder_type: type[ActivationNucleationBuilder | KineticNucleationBuilder],
    strategy_type: type[
        ActivationNucleationStrategy | KineticNucleationStrategy
    ],
) -> None:
    """Direct source metadata accepts each concrete P4 strategy type."""
    parameters = _parameters()
    if builder_type is KineticNucleationBuilder:
        parameters["coefficient_units"] = "m^3/s"
    strategy = builder_type().set_parameters(parameters).build()

    config = NucleationSourceConfig(strategy, 2)

    assert isinstance(config.strategy, strategy_type)
    assert config.precursor_index == 2
    with pytest.raises(FrozenInstanceError):
        config.precursor_index = 3  # type: ignore[misc]


def test_source_configuration_accepts_numpy_integer_index() -> None:
    """Source metadata accepts non-Boolean NumPy integer precursor indexes."""
    strategy = (
        ActivationNucleationBuilder().set_parameters(_parameters()).build()
    )

    config = NucleationSourceConfig(strategy, np.int64(2))  # type: ignore[arg-type]

    assert config.strategy is strategy
    assert config.precursor_index == 2


def test_source_configuration_rejects_abstract_strategy_subclass() -> None:
    """Source metadata does not accept arbitrary strategy implementations."""

    class UnsupportedStrategy(NucleationStrategy):
        """Concrete strategy outside the approved P4 source boundary."""

        def potential_rate(
            self,
            precursor_mass_concentration: float,
            precursor_molar_mass: float,
            temperature: float,
            saturation: float | None = None,
        ) -> float:
            return 0.0

    with pytest.raises(ValueError, match="strategy"):
        NucleationSourceConfig(UnsupportedStrategy(), 0)


@pytest.mark.parametrize(
    "index, exception_type",
    [
        (True, TypeError),
        (np.bool_(True), TypeError),
        (-1, ValueError),
        (np.int32(-1), ValueError),
        (1.5, TypeError),
        (np.float64(1.5), TypeError),
    ],
)
def test_source_configuration_builder_rejects_invalid_index(
    index: object,
    exception_type: type[Exception],
) -> None:
    """Source configuration validates its index through the immutable record."""
    strategy = (
        ActivationNucleationBuilder().set_parameters(_parameters()).build()
    )
    with pytest.raises(exception_type, match="precursor_index"):
        NucleationSourceConfigBuilder().set_parameters(
            {"strategy": strategy, "precursor_index": index}
        )


@pytest.mark.parametrize(
    "index, exception_type, message",
    [
        (True, TypeError, "precursor_index must be an integer"),
        (-1, ValueError, "precursor_index must be nonnegative"),
    ],
)
def test_source_builder_invalid_index_is_atomic_with_exact_error(
    index: object,
    exception_type: type[Exception],
    message: str,
) -> None:
    """Invalid indexes preserve staged metadata and retain exact errors."""
    strategy = (
        ActivationNucleationBuilder().set_parameters(_parameters()).build()
    )
    builder = NucleationSourceConfigBuilder().set_parameters(
        {"strategy": strategy, "precursor_index": 1}
    )

    with pytest.raises(exception_type, match=f"^{message}$"):
        builder.set_parameters({"strategy": strategy, "precursor_index": index})

    assert builder.build().precursor_index == 1


@pytest.mark.parametrize(
    "parameters, message",
    [
        (None, "dict"),
        ({}, "Missing required"),
        ({"coefficient": 1.0}, "coefficient requires"),
        ({"coefficient_units": "s^-1"}, "coefficient_units requires"),
    ],
)
def test_strategy_mapping_rejects_incomplete_schema(
    parameters: object, message: str
) -> None:
    """Strict mapping construction rejects non-dicts and incomplete schemas."""
    with pytest.raises((TypeError, ValueError), match=message):
        ActivationNucleationBuilder().set_parameters(parameters)  # type: ignore[arg-type]


def test_source_builder_rejects_extra_keys_atomically() -> None:
    """Source builder rejects P2-style keys without altering prior metadata."""
    strategy = (
        ActivationNucleationBuilder().set_parameters(_parameters()).build()
    )
    builder = NucleationSourceConfigBuilder().set_parameters(
        {"strategy": strategy, "precursor_index": 1}
    )

    with pytest.raises(ValueError, match="only strategy"):
        builder.set_parameters(
            {
                "strategy": strategy,
                "precursor_index": 2,
                "duration": 1.0,
            }
        )

    assert builder.build().precursor_index == 1


@pytest.mark.parametrize(
    "invalid_parameters, exception_type",
    [
        ({"strategy": None, "precursor_index": 1}, ValueError),
        ({"strategy": None, "precursor_index": -1}, ValueError),
        ({"strategy": None, "precursor_index": True}, ValueError),
    ],
)
def test_source_builder_invalid_mapping_retains_prior_configuration(
    invalid_parameters: dict[str, object],
    exception_type: type[Exception],
) -> None:
    """Schema-valid invalid source mappings leave prior metadata buildable."""
    strategy = (
        ActivationNucleationBuilder().set_parameters(_parameters()).build()
    )
    builder = NucleationSourceConfigBuilder().set_parameters(
        {"strategy": strategy, "precursor_index": 1}
    )

    with pytest.raises(exception_type):
        builder.set_parameters(invalid_parameters)

    assert builder.build().precursor_index == 1


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"strategy": None, "precursor_index": 0}, "strategy"),
        ({"strategy": None, "precursor_index_units": "dimensionless"}, "only"),
        (
            {"strategy": None, "precursor_index": 0, "duration": 1.0},
            "only",
        ),
    ],
)
def test_source_builder_mapping_rejects_invalid_p4_only_schema(
    payload: dict[str, object],
    message: str,
) -> None:
    """P4 source metadata rejects units, P2 fields, and invalid strategies."""
    with pytest.raises((TypeError, ValueError), match=message):
        NucleationSourceConfigBuilder().set_parameters(payload)
