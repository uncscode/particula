"""Validated potential-rate strategies for particle nucleation.

This module implements activation and kinetic potential event rates in SI
units.  It deliberately does not create particles or mutate gas, particle, or
slot state.

Kulmala, M., et al. (2006). Toward direct measurement of atmospheric
nucleation. Science, 318, 89--92. https://doi.org/10.1126/science.1144124

Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and Physics.
The caller supplies survival factors in the context of Kerminen--Kulmala.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import cast

import numpy as np

from particula.util.constants import AVOGADRO_NUMBER


def _is_scalar_real(value: object) -> bool:
    """Return whether a value is an accepted Python or NumPy real scalar."""
    return (
        isinstance(value, (Real, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and not isinstance(value, np.ndarray)
    )


def _finite_scalar(value: object, name: str) -> float:
    """Validate and normalize a finite real scalar."""
    if not _is_scalar_real(value):
        raise TypeError(f"{name} must be a real scalar")
    try:
        normalized = float(cast(Real, value))
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _nonnegative_scalar(value: object, name: str) -> float:
    """Validate a finite nonnegative real scalar."""
    normalized = _finite_scalar(value, name)
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _positive_scalar(value: object, name: str) -> float:
    """Validate a finite positive real scalar."""
    normalized = _finite_scalar(value, name)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


@dataclass(frozen=True)
class ClosedInterval:
    """A finite inclusive interval for a scalar quantity."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Validate interval endpoints."""
        lower = _finite_scalar(self.lower, "lower")
        upper = _finite_scalar(self.upper, "upper")
        if lower > upper:
            raise ValueError("lower must be less than or equal to upper")

    def contains(self, value: float) -> bool:
        """Return whether a validated value belongs to this interval."""
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class NucleationValidityDomain:
    """Inclusive physical domain for a nucleation-rate parameterization."""

    precursor_number_concentration: ClosedInterval
    temperature: ClosedInterval
    saturation: ClosedInterval | None = None

    def __post_init__(self) -> None:
        """Validate nested interval records."""
        if not isinstance(self.precursor_number_concentration, ClosedInterval):
            raise ValueError(
                "precursor_number_concentration must be ClosedInterval"
            )
        if not isinstance(self.temperature, ClosedInterval):
            raise ValueError("temperature must be ClosedInterval")
        if self.saturation is not None and not isinstance(
            self.saturation, ClosedInterval
        ):
            raise ValueError("saturation must be ClosedInterval or None")


@dataclass(frozen=True)
class InjectionComposition:
    """Future species-order molecule counts for one formed particle."""

    molecule_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the fixed injection composition."""
        if not isinstance(self.molecule_counts, tuple):
            raise TypeError("molecule_counts must be a tuple")
        if not self.molecule_counts:
            raise ValueError("molecule_counts must be nonempty")
        if any(
            not isinstance(count, (int, np.integer))
            or isinstance(count, (bool, np.bool_))
            for count in self.molecule_counts
        ):
            raise TypeError("molecule_counts must contain integers")
        if any(count < 0 for count in self.molecule_counts):
            raise ValueError("molecule_counts must be nonnegative")
        if not any(count > 0 for count in self.molecule_counts):
            raise ValueError("molecule_counts must include a positive count")


@dataclass(frozen=True)
class FormationMetadata:
    """Potential-rate formation size metadata."""

    formation_diameter: float
    diameter_convention: str = "mobility_diameter"

    def __post_init__(self) -> None:
        """Validate formation representation metadata."""
        _positive_scalar(self.formation_diameter, "formation_diameter")
        if self.diameter_convention != "mobility_diameter":
            raise ValueError("diameter_convention must be mobility_diameter")


class NucleationStrategy(ABC):
    """Abstract interface for scalar nucleation potential-rate strategies."""

    @abstractmethod
    def potential_rate(
        self,
        precursor_mass_concentration: float,
        precursor_molar_mass: float,
        temperature: float,
        saturation: float | None = None,
    ) -> float:
        """Calculate a potential event rate in #/m^3/s."""


def _validate_strategy_configuration(
    coefficient: object,
    validity_domain: object,
    injection_composition: object,
    formation_metadata: object,
    survival_factor: object,
) -> None:
    """Validate shared immutable strategy configuration."""
    _nonnegative_scalar(coefficient, "coefficient")
    _nonnegative_scalar(survival_factor, "survival_factor")
    if not isinstance(validity_domain, NucleationValidityDomain):
        raise ValueError("validity_domain must be NucleationValidityDomain")
    if not isinstance(injection_composition, InjectionComposition):
        raise ValueError("injection_composition must be InjectionComposition")
    if not isinstance(formation_metadata, FormationMetadata):
        raise ValueError("formation_metadata must be FormationMetadata")


def _convert_number_concentration(
    precursor_mass_concentration: float,
    precursor_molar_mass: float,
) -> float:
    """Convert mass concentration to molecule number concentration."""
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            concentration = (
                np.float64(precursor_mass_concentration)
                / np.float64(precursor_molar_mass)
                * np.float64(AVOGADRO_NUMBER)
            )
    except (FloatingPointError, OverflowError) as error:
        raise ValueError(
            "precursor_number_concentration must be finite"
        ) from error
    if not np.isfinite(concentration):
        raise ValueError("precursor_number_concentration must be finite")
    return float(concentration)


def _preflight_rate_inputs(
    validity_domain: NucleationValidityDomain,
    precursor_mass_concentration: object,
    precursor_molar_mass: object,
    temperature: object,
    saturation: object,
) -> tuple[float, float, float | None]:
    """Validate rate inputs before converting the precursor concentration."""
    mass_concentration = _nonnegative_scalar(
        precursor_mass_concentration,
        "precursor_mass_concentration",
    )
    molar_mass = _positive_scalar(precursor_molar_mass, "precursor_molar_mass")
    normalized_temperature = _positive_scalar(temperature, "temperature")

    saturation_interval = validity_domain.saturation
    if saturation_interval is None:
        if saturation is not None:
            _nonnegative_scalar(saturation, "saturation")
            raise ValueError(
                "saturation must be None outside the validity domain"
            )
        normalized_saturation = None
    else:
        if saturation is None:
            raise TypeError("saturation must be a real scalar")
        normalized_saturation = _nonnegative_scalar(saturation, "saturation")

    concentration = _convert_number_concentration(
        mass_concentration, molar_mass
    )
    return concentration, normalized_temperature, normalized_saturation


def _validate_rate_domain(
    validity_domain: NucleationValidityDomain,
    concentration: float,
    temperature: float,
    saturation: float | None,
) -> bool:
    """Validate nonzero-path domains and return saturation gate acceptance."""
    if not validity_domain.precursor_number_concentration.contains(
        concentration
    ):
        raise ValueError(
            "precursor_number_concentration outside validity domain"
        )
    if not validity_domain.temperature.contains(temperature):
        raise ValueError("temperature outside validity domain")
    saturation_interval = validity_domain.saturation
    if saturation_interval is None:
        return True
    if saturation is None:
        raise ValueError(
            "saturation must be provided within the validity domain"
        )
    if saturation < saturation_interval.lower:
        return False
    if saturation > saturation_interval.upper:
        raise ValueError("saturation outside validity domain")
    return True


def _calculate_rate(
    coefficient: float,
    concentration: float,
    survival_factor: float,
    kinetic: bool,
) -> float:
    """Calculate a finite activation or kinetic potential rate."""
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            rate = np.float64(coefficient) * np.float64(survival_factor)
            if kinetic:
                rate *= np.float64(concentration) * np.float64(concentration)
            else:
                rate *= np.float64(concentration)
    except FloatingPointError as error:
        raise ValueError("potential_rate must be finite") from error
    if not np.isfinite(rate):
        raise ValueError("potential_rate must be finite")
    return float(rate)


@dataclass(frozen=True)
class ActivationNucleationStrategy(NucleationStrategy):
    """Potential activation rate, ``J = A * C * survival_factor``.

    The coefficient has units of s^-1 and ``C`` is precursor molecule number
    concentration in #/m^3.
    """

    coefficient: float
    validity_domain: NucleationValidityDomain
    injection_composition: InjectionComposition
    formation_metadata: FormationMetadata
    survival_factor: float = 1.0

    def __post_init__(self) -> None:
        """Validate activation strategy configuration."""
        _validate_strategy_configuration(
            self.coefficient,
            self.validity_domain,
            self.injection_composition,
            self.formation_metadata,
            self.survival_factor,
        )

    def potential_rate(
        self,
        precursor_mass_concentration: float,
        precursor_molar_mass: float,
        temperature: float,
        saturation: float | None = None,
    ) -> float:
        """Calculate the activation potential event rate in #/m^3/s."""
        concentration, normalized_temperature, normalized_saturation = (
            _preflight_rate_inputs(
                self.validity_domain,
                precursor_mass_concentration,
                precursor_molar_mass,
                temperature,
                saturation,
            )
        )
        if (
            self.coefficient == 0
            or concentration == 0
            or self.survival_factor == 0
        ):
            return 0.0
        if not _validate_rate_domain(
            self.validity_domain,
            concentration,
            normalized_temperature,
            normalized_saturation,
        ):
            return 0.0
        return _calculate_rate(
            self.coefficient,
            concentration,
            self.survival_factor,
            kinetic=False,
        )


@dataclass(frozen=True)
class KineticNucleationStrategy(NucleationStrategy):
    """Potential kinetic rate, ``J = K * C**2 * survival_factor``.

    The coefficient has units of m^3/s and ``C`` is precursor molecule number
    concentration in #/m^3.
    """

    coefficient: float
    validity_domain: NucleationValidityDomain
    injection_composition: InjectionComposition
    formation_metadata: FormationMetadata
    survival_factor: float = 1.0

    def __post_init__(self) -> None:
        """Validate kinetic strategy configuration."""
        _validate_strategy_configuration(
            self.coefficient,
            self.validity_domain,
            self.injection_composition,
            self.formation_metadata,
            self.survival_factor,
        )

    def potential_rate(
        self,
        precursor_mass_concentration: float,
        precursor_molar_mass: float,
        temperature: float,
        saturation: float | None = None,
    ) -> float:
        """Calculate the kinetic potential event rate in #/m^3/s."""
        concentration, normalized_temperature, normalized_saturation = (
            _preflight_rate_inputs(
                self.validity_domain,
                precursor_mass_concentration,
                precursor_molar_mass,
                temperature,
                saturation,
            )
        )
        if (
            self.coefficient == 0
            or concentration == 0
            or self.survival_factor == 0
        ):
            return 0.0
        if not _validate_rate_domain(
            self.validity_domain,
            concentration,
            normalized_temperature,
            normalized_saturation,
        ):
            return 0.0
        return _calculate_rate(
            self.coefficient,
            concentration,
            self.survival_factor,
            kinetic=True,
        )
