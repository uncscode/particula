"""Tests for bounded scalar nucleation potential-rate strategies."""

from dataclasses import FrozenInstanceError
from typing import cast

import numpy as np
import numpy.testing as npt
import particula.dynamics
import pytest
from particula.dynamics.nucleation.nucleation_strategies import (
    AVOGADRO_NUMBER,
    ActivationNucleationStrategy,
    ClosedInterval,
    FormationMetadata,
    InjectionComposition,
    KineticNucleationStrategy,
    NucleationStrategy,
    NucleationValidityDomain,
)


@pytest.fixture
def domain() -> NucleationValidityDomain:
    """Return a valid domain with an inclusive saturation interval."""
    return NucleationValidityDomain(
        precursor_number_concentration=ClosedInterval(1.0e10, 1.0e30),
        temperature=ClosedInterval(250.0, 320.0),
        saturation=ClosedInterval(1.0, 2.0),
    )


@pytest.fixture
def composition() -> InjectionComposition:
    """Return valid future injection composition metadata."""
    return InjectionComposition((1, 2))


@pytest.fixture
def metadata() -> FormationMetadata:
    """Return valid potential-rate formation metadata."""
    return FormationMetadata(1.5e-9)


@pytest.fixture
def strategies(
    domain: NucleationValidityDomain,
    composition: InjectionComposition,
    metadata: FormationMetadata,
) -> tuple[ActivationNucleationStrategy, KineticNucleationStrategy]:
    """Return valid activation and kinetic strategies."""
    return (
        ActivationNucleationStrategy(
            2.0e-3, domain, composition, metadata, 0.5
        ),
        KineticNucleationStrategy(3.0e-20, domain, composition, metadata, 0.5),
    )


@pytest.mark.parametrize("multiplier", [1.0, 2.0])
def test_potential_rate_equations_and_mass_scaling(
    strategies: tuple[ActivationNucleationStrategy, KineticNucleationStrategy],
    multiplier: float,
) -> None:
    """Activation and kinetic equations use linear and quadratic C scaling."""
    mass_concentration = multiplier * 1.0e-12
    molar_mass = 0.1
    concentration = mass_concentration / molar_mass * AVOGADRO_NUMBER
    activation, kinetic = strategies

    npt.assert_allclose(
        activation.potential_rate(mass_concentration, molar_mass, 298.15, 1.5),
        2.0e-3 * concentration * 0.5,
    )
    npt.assert_allclose(
        kinetic.potential_rate(mass_concentration, molar_mass, 298.15, 1.5),
        3.0e-20 * concentration**2 * 0.5,
    )


def test_doubling_mass_doubles_activation_and_quadruples_kinetic(
    strategies: tuple[ActivationNucleationStrategy, KineticNucleationStrategy],
) -> None:
    """Rates retain the distinct activation and kinetic concentration scaling."""
    activation, kinetic = strategies
    base_mass_concentration = 1.0e-12
    molar_mass = 0.1

    activation_base = activation.potential_rate(
        base_mass_concentration, molar_mass, 298.15, 1.5
    )
    activation_doubled = activation.potential_rate(
        2.0 * base_mass_concentration, molar_mass, 298.15, 1.5
    )
    kinetic_base = kinetic.potential_rate(
        base_mass_concentration, molar_mass, 298.15, 1.5
    )
    kinetic_doubled = kinetic.potential_rate(
        2.0 * base_mass_concentration, molar_mass, 298.15, 1.5
    )

    npt.assert_allclose(activation_doubled / activation_base, 2.0)
    npt.assert_allclose(kinetic_doubled / kinetic_base, 4.0)


def test_inclusive_domain_endpoints_accept_numpy_scalars(
    composition: InjectionComposition,
    metadata: FormationMetadata,
) -> None:
    """Closed domain endpoints accept Python and NumPy real scalars."""
    domain = NucleationValidityDomain(
        precursor_number_concentration=ClosedInterval(
            AVOGADRO_NUMBER, 2 * AVOGADRO_NUMBER
        ),
        temperature=ClosedInterval(280.0, 300.0),
        saturation=ClosedInterval(1.0, 2.0),
    )
    strategy = ActivationNucleationStrategy(1.0, domain, composition, metadata)

    assert strategy.potential_rate(1.0, 1.0, 280.0, 1.0) > 0.0
    assert (
        strategy.potential_rate(
            np.float64(2.0), np.float64(1.0), np.float64(300.0), np.float64(2.0)
        )
        > 0.0
    )


@pytest.mark.parametrize("kind", ["activation", "kinetic"])
def test_zero_paths_and_below_saturation_gate_return_exact_zero(
    kind: str,
    domain: NucleationValidityDomain,
    composition: InjectionComposition,
    metadata: FormationMetadata,
) -> None:
    """Zero coefficients, precursor, survival, and below gate are no-ops."""
    strategy_type = (
        ActivationNucleationStrategy
        if kind == "activation"
        else KineticNucleationStrategy
    )
    for coefficient, survival, mass, saturation in (
        (0.0, 1.0, 1.0e-12, 1.5),
        (1.0, 0.0, 1.0e-12, 1.5),
        (1.0, 1.0, 0.0, 1.5),
        (1.0, 1.0, 1.0e-12, 0.5),
    ):
        strategy = strategy_type(
            coefficient, domain, composition, metadata, survival
        )
        assert strategy.potential_rate(mass, 0.1, 298.15, saturation) == 0.0


def test_zero_precursor_bypasses_positive_concentration_lower_bound(
    composition: InjectionComposition,
    metadata: FormationMetadata,
) -> None:
    """A valid zero precursor remains a no-op below a positive lower bound."""
    domain = NucleationValidityDomain(
        ClosedInterval(1.0, 2.0), ClosedInterval(250.0, 300.0)
    )
    strategy = ActivationNucleationStrategy(1.0, domain, composition, metadata)

    assert strategy.potential_rate(0.0, 0.1, 280.0) == 0.0


def test_domain_without_saturation_rejects_extraneous_saturation(
    composition: InjectionComposition,
    metadata: FormationMetadata,
) -> None:
    """A saturation-free domain requires an explicitly absent saturation."""
    domain = NucleationValidityDomain(
        ClosedInterval(0.0, 1.0e30), ClosedInterval(250.0, 300.0)
    )
    strategy = ActivationNucleationStrategy(1.0, domain, composition, metadata)

    with pytest.raises(ValueError, match="saturation"):
        strategy.potential_rate(1.0e-12, 0.1, 280.0, 1.0)


@pytest.mark.parametrize("coefficient,survival", [(0.0, 1.0), (1.0, 0.0)])
@pytest.mark.parametrize("saturation", [None, 1.5])
def test_zero_configurations_do_not_bypass_saturation_preflight(
    coefficient: float,
    survival: float,
    saturation: float | None,
    domain: NucleationValidityDomain,
    composition: InjectionComposition,
    metadata: FormationMetadata,
) -> None:
    """Zero configuration still requires the domain's saturation input."""
    strategy = ActivationNucleationStrategy(
        coefficient, domain, composition, metadata, survival
    )
    if saturation is None:
        with pytest.raises(TypeError, match="saturation"):
            strategy.potential_rate(1.0e-12, 0.1, 298.15, saturation)
    else:
        assert strategy.potential_rate(1.0e-12, 0.1, 298.15, saturation) == 0.0


@pytest.mark.parametrize(
    "coefficient,survival,mass",
    [(0.0, 1.0, 1.0e-12), (1.0, 0.0, 1.0e-12), (1.0, 1.0, 0.0)],
)
@pytest.mark.parametrize("saturation", [True, "bad", np.array(1.0)])
def test_zero_paths_do_not_bypass_malformed_saturation(
    coefficient: float,
    survival: float,
    mass: float,
    saturation: object,
    domain: NucleationValidityDomain,
    composition: InjectionComposition,
    metadata: FormationMetadata,
) -> None:
    """Zero paths still reject malformed required saturation inputs."""
    strategy = ActivationNucleationStrategy(
        coefficient, domain, composition, metadata, survival
    )

    with pytest.raises(TypeError, match="saturation"):
        strategy.potential_rate(
            mass, 0.1, 298.15, cast(float | None, saturation)
        )


@pytest.mark.parametrize(
    "coefficient,survival,mass",
    [(0.0, 1.0, 1.0e-12), (1.0, 0.0, 1.0e-12), (1.0, 1.0, 0.0)],
)
def test_zero_paths_do_not_bypass_extraneous_saturation(
    coefficient: float,
    survival: float,
    mass: float,
    composition: InjectionComposition,
    metadata: FormationMetadata,
) -> None:
    """Zero paths still reject saturation outside a saturation-free domain."""
    domain = NucleationValidityDomain(
        ClosedInterval(0.0, 1.0e30), ClosedInterval(250.0, 320.0)
    )
    strategy = ActivationNucleationStrategy(
        coefficient, domain, composition, metadata, survival
    )

    with pytest.raises(ValueError, match="saturation"):
        strategy.potential_rate(mass, 0.1, 298.15, 1.0)


@pytest.mark.parametrize(
    "mass,molar_mass,temperature,fragment",
    [
        (-1.0, 0.1, 298.15, "precursor_mass_concentration"),
        (1.0e-12, 0.0, 298.15, "precursor_molar_mass"),
        (1.0e-12, 0.1, 0.0, "temperature"),
    ],
)
def test_zero_configuration_does_not_bypass_physical_preflight(
    mass: float,
    molar_mass: float,
    temperature: float,
    fragment: str,
    domain: NucleationValidityDomain,
    composition: InjectionComposition,
    metadata: FormationMetadata,
) -> None:
    """Zero coefficient does not hide invalid basic physical inputs."""
    strategy = ActivationNucleationStrategy(0.0, domain, composition, metadata)
    with pytest.raises(ValueError, match=fragment):
        strategy.potential_rate(mass, molar_mass, temperature, 1.5)


@pytest.mark.parametrize("coefficient,survival", [(0.0, 1.0), (1.0, 0.0)])
def test_conversion_overflow_precedes_zero_configuration_return(
    coefficient: float,
    survival: float,
    domain: NucleationValidityDomain,
    composition: InjectionComposition,
    metadata: FormationMetadata,
) -> None:
    """Overflowing C conversion raises even for otherwise zero paths."""
    strategy = ActivationNucleationStrategy(
        coefficient, domain, composition, metadata, survival
    )
    with pytest.raises(ValueError, match="precursor_number_concentration"):
        strategy.potential_rate(1.0e308, 1.0e-308, 298.15, 1.5)


@pytest.mark.parametrize(
    "value", [True, "bad", None, np.array(1.0), np.array([1.0])]
)
def test_evaluation_rejects_non_scalar_mass_inputs(
    value: object,
    strategies: tuple[ActivationNucleationStrategy, KineticNucleationStrategy],
) -> None:
    """Evaluation accepts only non-bool Python or NumPy real scalars."""
    for strategy in strategies:
        with pytest.raises(TypeError, match="precursor_mass_concentration"):
            strategy.potential_rate(value, 0.1, 298.15, 1.5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "argument,value,fragment",
    [
        ("precursor_molar_mass", True, "precursor_molar_mass"),
        ("precursor_molar_mass", np.array(0.1), "precursor_molar_mass"),
        ("temperature", None, "temperature"),
        ("temperature", "bad", "temperature"),
        ("saturation", np.array([1.5]), "saturation"),
    ],
)
def test_evaluation_rejects_non_scalar_other_inputs(
    argument: str,
    value: object,
    fragment: str,
    strategies: tuple[ActivationNucleationStrategy, KineticNucleationStrategy],
) -> None:
    """Every evaluation argument enforces the scalar-only type contract."""
    for strategy in strategies:
        arguments: dict[str, object] = {
            "precursor_mass_concentration": 1.0e-12,
            "precursor_molar_mass": 0.1,
            "temperature": 298.15,
            "saturation": 1.5,
        }
        arguments[argument] = value
        with pytest.raises(TypeError, match=fragment):
            strategy.potential_rate(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1.0])
def test_evaluation_rejects_invalid_mass_values(
    value: float,
    strategies: tuple[ActivationNucleationStrategy, KineticNucleationStrategy],
) -> None:
    """Evaluation rejects nonfinite and negative mass concentrations."""
    for strategy in strategies:
        with pytest.raises(ValueError, match="precursor_mass_concentration"):
            strategy.potential_rate(value, 0.1, 298.15, 1.5)


def test_domain_rejections_and_saturation_contract(
    strategies: tuple[ActivationNucleationStrategy, KineticNucleationStrategy],
) -> None:
    """Nonzero paths reject concentration, temperature, and upper saturation."""
    for strategy in strategies:
        with pytest.raises(ValueError, match="precursor_number_concentration"):
            strategy.potential_rate(1.0e-20, 0.1, 298.15, 1.5)
        with pytest.raises(ValueError, match="temperature"):
            strategy.potential_rate(1.0e-12, 0.1, 400.0, 1.5)
        with pytest.raises(ValueError, match="saturation"):
            strategy.potential_rate(1.0e-12, 0.1, 298.15, 3.0)


@pytest.mark.parametrize(
    "factory,fragment",
    [
        (lambda: ClosedInterval(2.0, 1.0), "lower"),
        (lambda: ClosedInterval(np.nan, 1.0), "lower"),
        (lambda: ClosedInterval(True, 1.0), "lower"),
        (
            lambda: NucleationValidityDomain(
                ClosedInterval(0, 1), cast(ClosedInterval, "bad")
            ),
            "temperature",
        ),
        (
            lambda: NucleationValidityDomain(
                cast(ClosedInterval, "bad"), ClosedInterval(250, 300)
            ),
            "precursor_number_concentration",
        ),
        (
            lambda: NucleationValidityDomain(
                ClosedInterval(0, 1),
                ClosedInterval(250, 300),
                cast(ClosedInterval, "bad"),
            ),
            "saturation",
        ),
        (lambda: InjectionComposition(()), "molecule_counts"),
        (lambda: InjectionComposition((0, 0)), "molecule_counts"),
        (lambda: InjectionComposition((1.0,)), "molecule_counts"),  # type: ignore[arg-type]
        (lambda: InjectionComposition([1]), "molecule_counts"),  # type: ignore[arg-type]
        (lambda: FormationMetadata(0.0), "formation_diameter"),
        (
            lambda: FormationMetadata(1.0, "geometric"),
            "diameter_convention",
        ),
    ],
)
def test_records_validate_invalid_configuration(
    factory: object,
    fragment: str,
) -> None:
    """Frozen configuration records reject malformed values."""
    with pytest.raises((TypeError, ValueError), match=fragment):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "strategy_type",
    [ActivationNucleationStrategy, KineticNucleationStrategy],
)
@pytest.mark.parametrize("coefficient,survival", [(-1.0, 1.0), (1.0, np.inf)])
def test_strategies_validate_coefficient_and_survival(
    strategy_type: type[
        ActivationNucleationStrategy | KineticNucleationStrategy
    ],
    coefficient: float,
    survival: float,
    domain: NucleationValidityDomain,
    composition: InjectionComposition,
    metadata: FormationMetadata,
) -> None:
    """Both strategies reject invalid coefficients and survival factors."""
    with pytest.raises(ValueError, match="coefficient|survival_factor"):
        strategy_type(coefficient, domain, composition, metadata, survival)


@pytest.mark.parametrize(
    "strategy_type",
    [ActivationNucleationStrategy, KineticNucleationStrategy],
)
def test_potential_rate_rejects_finite_input_rate_overflow(
    strategy_type: type[
        ActivationNucleationStrategy | KineticNucleationStrategy
    ],
    composition: InjectionComposition,
    metadata: FormationMetadata,
) -> None:
    """Finite inputs that overflow the final rate raise ValueError."""
    domain = NucleationValidityDomain(
        ClosedInterval(0.0, 1.0e30), ClosedInterval(250.0, 320.0)
    )
    strategy = strategy_type(1.0e308, domain, composition, metadata)

    with pytest.raises(ValueError, match="potential_rate"):
        strategy.potential_rate(1.0, 1.0, 298.15)


def test_records_are_frozen_and_abstract_interface_cannot_instantiate(
    domain: NucleationValidityDomain,
) -> None:
    """Configuration is immutable and the strategy interface is abstract."""
    interval = ClosedInterval(0.0, 1.0)
    with pytest.raises(FrozenInstanceError):
        interval.lower = -1.0  # type: ignore[misc]
    with pytest.raises(TypeError, match="abstract"):
        NucleationStrategy()  # type: ignore[abstract]
    with pytest.raises(FrozenInstanceError):
        domain.saturation = None  # type: ignore[misc]


def test_strategies_are_concrete_module_only() -> None:
    """P1 strategy symbols remain intentionally absent from dynamics exports."""
    assert not hasattr(particula.dynamics, "ActivationNucleationStrategy")
    assert not hasattr(particula.dynamics, "KineticNucleationStrategy")
