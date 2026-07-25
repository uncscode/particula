"""Validated CPU-only potential-rate strategies for particle nucleation.

Activation and kinetic parameterizations return potential rates [#/m³/s] from
precursor number concentration [#/m³]. They do not create particles or mutate
gas, particle, or slot state. This concrete module is intentionally not
re-exported through ``particula.dynamics``.

Kulmala, M., et al. (2006). Toward direct measurement of atmospheric
nucleation. *Science*, 318, 89--92.
https://doi.org/10.1126/science.1144124

Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and Physics*.
Kerminen, V.-M., & Kulmala, M. (2002). Analytical formulae connecting the
"real" and the "apparent" nucleation rate. *JGR*, 107(D22), 4627.
https://doi.org/10.1029/2002JD002184
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import cast

import numpy as np

from particula.util.constants import AVOGADRO_NUMBER


def _is_scalar_real(value: object) -> bool:
    """Return whether a value is a non-Boolean Python or NumPy real scalar.

    Args:
        value: Candidate value to classify.

    Returns:
        ``True`` when ``value`` is an accepted scalar real value.
    """
    return (
        isinstance(value, (Real, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and not isinstance(value, np.ndarray)
    )


def _finite_scalar(value: object, name: str) -> float:
    """Normalize a finite scalar real value.

    Args:
        value: Candidate scalar value.
        name: Parameter name used in validation errors.

    Returns:
        Normalized finite value.

    Raises:
        TypeError: If ``value`` is not an accepted scalar real value.
        ValueError: If conversion overflows or the value is nonfinite.
    """
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
    """Normalize a finite nonnegative scalar real value.

    Args:
        value: Candidate scalar value.
        name: Parameter name used in validation errors.

    Returns:
        Normalized nonnegative value.

    Raises:
        TypeError: If ``value`` is not an accepted scalar real value.
        ValueError: If the value is nonfinite or negative.
    """
    normalized = _finite_scalar(value, name)
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _positive_scalar(value: object, name: str) -> float:
    """Normalize a finite positive scalar real value.

    Args:
        value: Candidate scalar value.
        name: Parameter name used in validation errors.

    Returns:
        Normalized positive value.

    Raises:
        TypeError: If ``value`` is not an accepted scalar real value.
        ValueError: If the value is nonfinite or not positive.
    """
    normalized = _finite_scalar(value, name)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


@dataclass(frozen=True)
class ClosedInterval:
    """Finite inclusive interval for a scalar physical quantity.

    Attributes:
        lower: Finite inclusive lower endpoint.
        upper: Finite inclusive upper endpoint, no less than ``lower``.
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Validate finite ordered interval endpoints.

        Raises:
            TypeError: If an endpoint is not a scalar real value.
            ValueError: If an endpoint is nonfinite or ``lower > upper``.
        """
        lower = _finite_scalar(self.lower, "lower")
        upper = _finite_scalar(self.upper, "upper")
        if lower > upper:
            raise ValueError("lower must be less than or equal to upper")

    def contains(self, value: float) -> bool:
        """Test inclusive membership for an already validated value.

        Args:
            value: Normalized scalar value to test.

        Returns:
            ``True`` if ``lower <= value <= upper``.
        """
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class NucleationValidityDomain:
    """Inclusive physical domain for a nucleation-rate parameterization.

    Attributes:
        precursor_number_concentration: Allowed precursor concentration [#/m³].
        temperature: Allowed temperature [K].
        saturation: Optional allowed dimensionless saturation interval.
    """

    precursor_number_concentration: ClosedInterval
    temperature: ClosedInterval
    saturation: ClosedInterval | None = None

    def __post_init__(self) -> None:
        """Validate the nested interval records.

        Raises:
            ValueError: If a required field is not a ``ClosedInterval`` or an
                optional saturation field is neither one nor ``None``.
        """
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
    """Future species-order molecule counts for one formed particle.

    The records preserve composition metadata but do not inject a particle in
    this potential-rate-only implementation.

    Attributes:
        molecule_counts: Nonempty molecule counts by future species order.
    """

    molecule_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the nonempty, nonnegative injection composition.

        Raises:
            TypeError: If counts are not a tuple of non-Boolean integers.
            ValueError: If no count is positive or any count is negative.
        """
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
    """Metadata for the size represented by a potential formation rate.

    Attributes:
        formation_diameter: Positive mobility diameter [m].
        diameter_convention: Required ``"mobility_diameter"`` convention.
    """

    formation_diameter: float
    diameter_convention: str = "mobility_diameter"

    def __post_init__(self) -> None:
        """Validate the potential-rate size representation.

        Raises:
            TypeError: If the diameter is not a scalar real value.
            ValueError: If the diameter is invalid or its convention differs.
        """
        _positive_scalar(self.formation_diameter, "formation_diameter")
        if self.diameter_convention != "mobility_diameter":
            raise ValueError("diameter_convention must be mobility_diameter")


class NucleationStrategy(ABC):
    """Abstract interface for scalar nucleation potential-rate strategies.

    Implementations return a potential event rate only; they do not represent
    particle sources, survival inference, or state mutation.
    """

    @abstractmethod
    def potential_rate(
        self,
        precursor_mass_concentration: float,
        precursor_molar_mass: float,
        temperature: float,
        saturation: float | None = None,
    ) -> float:
        """Calculate a potential formation-event rate [#/m³/s].

        Args:
            precursor_mass_concentration: Precursor mass concentration [kg/m³].
            precursor_molar_mass: Precursor molar mass [kg/mol].
            temperature: Temperature [K].
            saturation: Optional dimensionless saturation.

        Returns:
            Potential formation-event rate [#/m³/s].
        """


def _validate_strategy_configuration(
    coefficient: object,
    validity_domain: object,
    injection_composition: object,
    formation_metadata: object,
    survival_factor: object,
) -> None:
    """Validate shared immutable potential-rate strategy configuration.

    Args:
        coefficient: Nonnegative activation [s⁻¹] or kinetic [m³/s] coefficient.
        validity_domain: Inclusive domain for rate evaluation.
        injection_composition: Metadata for a future formed particle.
        formation_metadata: Metadata for potential-rate formation size.
        survival_factor: Nonnegative caller-supplied dimensionless factor.

    Raises:
        TypeError: If a scalar configuration value is not a scalar real value.
        ValueError: If a scalar is invalid or a metadata record has wrong type.
    """
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
    """Convert precursor mass concentration to number concentration [#/m³].

    Args:
        precursor_mass_concentration: Validated precursor mass concentration
            [kg/m³].
        precursor_molar_mass: Validated precursor molar mass [kg/mol].

    Returns:
        Finite precursor molecule number concentration [#/m³].

    Raises:
        ValueError: If float64 conversion or multiplication is nonfinite.
    """
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
    """Validate basic rate inputs and convert precursor concentration.

    Saturation is required only when its interval is configured. This preflight
    intentionally precedes zero-rate and domain checks.

    Args:
        validity_domain: Domain that determines saturation input semantics.
        precursor_mass_concentration: Candidate precursor concentration [kg/m³].
        precursor_molar_mass: Candidate precursor molar mass [kg/mol].
        temperature: Candidate temperature [K].
        saturation: Candidate dimensionless saturation.

    Returns:
        Precursor number concentration [#/m³], temperature [K], and optional
        saturation.

    Raises:
        TypeError: If a required input is not an accepted scalar real value.
        ValueError: If a physical input is invalid, conversion is nonfinite, or
            saturation is supplied outside a saturation domain.
    """
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
    """Validate nonzero-path domains and apply the lower saturation gate.

    Args:
        validity_domain: Inclusive concentration, temperature, and saturation
            constraints.
        concentration: Validated precursor number concentration [#/m³].
        temperature: Validated temperature [K].
        saturation: Validated optional dimensionless saturation.

    Returns:
        ``False`` only when valid saturation is below its lower interval bound.

    Raises:
        ValueError: If concentration, temperature, or saturation exceeds its
            applicable validity domain.
    """
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
    """Calculate a finite activation or kinetic potential rate [#/m³/s].

    Args:
        coefficient: Validated activation [s⁻¹] or kinetic [m³/s] coefficient.
        concentration: Validated precursor number concentration [#/m³].
        survival_factor: Validated dimensionless survival factor.
        kinetic: Whether to use the kinetic ``C²`` relation instead of ``C``.

    Returns:
        Finite potential formation-event rate [#/m³/s].

    Raises:
        ValueError: If float64 rate evaluation overflows or is nonfinite.
    """
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
    """Activation potential-rate parameterization, ``J = A × C × S``.

    ``A`` is the nonnegative activation coefficient [s⁻¹], ``C`` is precursor
    number concentration [#/m³], and ``S`` is a caller-supplied dimensionless
    survival factor. A valid zero coefficient, concentration, or survival
    factor returns exactly ``0.0`` after basic input validation and conversion.

    Attributes:
        coefficient: Activation coefficient [s⁻¹].
        validity_domain: Inclusive parameterization domain.
        injection_composition: Metadata for a future formed particle.
        formation_metadata: Potential-rate formation-size metadata.
        survival_factor: Caller-supplied nonnegative dimensionless factor.
    """

    coefficient: float
    validity_domain: NucleationValidityDomain
    injection_composition: InjectionComposition
    formation_metadata: FormationMetadata
    survival_factor: float = 1.0

    def __post_init__(self) -> None:
        """Validate immutable activation-strategy configuration.

        Raises:
            TypeError: If a scalar configuration value is not a scalar real.
            ValueError: If configuration values or metadata records are invalid.
        """
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
        """Calculate the activation potential event rate [#/m³/s].

        Basic inputs and number-concentration conversion are validated before
        exact zero paths. Nonzero inputs must lie in the configured inclusive
        concentration and temperature intervals; below-gate saturation returns
        exactly ``0.0``.

        Args:
            precursor_mass_concentration: Precursor mass concentration [kg/m³].
            precursor_molar_mass: Precursor molar mass [kg/mol].
            temperature: Temperature [K].
            saturation: Dimensionless saturation when configured, else ``None``.

        Returns:
            Potential activation event rate [#/m³/s].

        Raises:
            TypeError: If required inputs are not accepted scalar real values.
            ValueError: If inputs are physically invalid, out of domain, or
                yield a nonfinite conversion or rate.
        """
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
    """Kinetic potential-rate parameterization, ``J = K × C² × S``.

    ``K`` is the nonnegative kinetic coefficient [m³/s], ``C`` is precursor
    number concentration [#/m³], and ``S`` is a caller-supplied dimensionless
    survival factor. A valid zero coefficient, concentration, or survival
    factor returns exactly ``0.0`` after basic input validation and conversion.

    Attributes:
        coefficient: Kinetic coefficient [m³/s].
        validity_domain: Inclusive parameterization domain.
        injection_composition: Metadata for a future formed particle.
        formation_metadata: Potential-rate formation-size metadata.
        survival_factor: Caller-supplied nonnegative dimensionless factor.
    """

    coefficient: float
    validity_domain: NucleationValidityDomain
    injection_composition: InjectionComposition
    formation_metadata: FormationMetadata
    survival_factor: float = 1.0

    def __post_init__(self) -> None:
        """Validate immutable kinetic-strategy configuration.

        Raises:
            TypeError: If a scalar configuration value is not a scalar real.
            ValueError: If configuration values or metadata records are invalid.
        """
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
        """Calculate the kinetic potential event rate [#/m³/s].

        Basic inputs and number-concentration conversion are validated before
        exact zero paths. Nonzero inputs must lie in the configured inclusive
        concentration and temperature intervals; below-gate saturation returns
        exactly ``0.0``.

        Args:
            precursor_mass_concentration: Precursor mass concentration [kg/m³].
            precursor_molar_mass: Precursor molar mass [kg/mol].
            temperature: Temperature [K].
            saturation: Dimensionless saturation when configured, else ``None``.

        Returns:
            Potential kinetic event rate [#/m³/s].

        Raises:
            TypeError: If required inputs are not accepted scalar real values.
            ValueError: If inputs are physically invalid, out of domain, or
                yield a nonfinite conversion or rate.
        """
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
