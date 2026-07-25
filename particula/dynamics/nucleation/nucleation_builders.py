"""Build immutable, normalized P4 nucleation configurations.

The builders accept only the documented physical units and make configuration
from mappings atomic. They construct potential-rate metadata only; P2/P3
source-demand finalization and particle or gas mutation remain in the concrete
``particle_source`` module.
"""

from abc import ABC
from typing import Self, cast

from particula.abc_builder import BuilderABC
from particula.dynamics.nucleation.nucleation_configuration import (
    NucleationSourceConfig,
)
from particula.dynamics.nucleation.nucleation_strategies import (
    ActivationNucleationStrategy,
    ClosedInterval,
    FormationMetadata,
    InjectionComposition,
    KineticNucleationStrategy,
    NucleationStrategy,
    NucleationValidityDomain,
    _nonnegative_scalar,
    _positive_scalar,
)

_REQUIRED = (
    "coefficient",
    "coefficient_provenance",
    "precursor_number_concentration_lower",
    "precursor_number_concentration_upper",
    "temperature_lower",
    "temperature_upper",
    "injection_composition",
    "formation_diameter",
)
_SCALARS = {
    "coefficient",
    "precursor_number_concentration_lower",
    "precursor_number_concentration_upper",
    "temperature_lower",
    "temperature_upper",
    "formation_diameter",
    "saturation_lower",
    "saturation_upper",
    "survival_factor",
}
_STAGED_ATTRIBUTES = (
    *_SCALARS,
    "coefficient_provenance",
    "injection_composition",
    "diameter_convention",
)


class _NucleationStrategyBuilder(BuilderABC, ABC):
    """Provide shared strict normalization for potential-rate builders.

    Subclasses define their accepted coefficient units and construct the
    matching immutable strategy. Required scalar fields are normalized to SI
    units before they are retained on the builder.
    """

    coefficient_units: set[str]

    def __init__(self) -> None:
        """Initialize an unconfigured strategy builder."""
        super().__init__(required_parameters=list(_REQUIRED))
        self.coefficient: float | None = None
        self.coefficient_provenance: str | None = None
        self.precursor_number_concentration_lower: float | None = None
        self.precursor_number_concentration_upper: float | None = None
        self.temperature_lower: float | None = None
        self.temperature_upper: float | None = None
        self.injection_composition: InjectionComposition | None = None
        self.formation_diameter: float | None = None
        self.saturation_lower: float | None = None
        self.saturation_upper: float | None = None
        self.survival_factor: float | None = None
        self.diameter_convention: str | None = None

    @staticmethod
    def _unit(unit: object, accepted: set[str], name: str) -> str:
        if not isinstance(unit, str) or unit not in accepted:
            raise ValueError(f"{name}_units must be one of {sorted(accepted)}")
        return unit

    @staticmethod
    def _converted(
        value: object,
        unit: object,
        accepted: set[str],
        name: str,
    ) -> float:
        normalized = _nonnegative_scalar(value, name)
        normalized_unit = _NucleationStrategyBuilder._unit(unit, accepted, name)
        factors = {"cm^3/s": 1e-6, "1/cm^3": 1e6, "nm": 1e-9}
        return normalized * factors.get(normalized_unit, 1.0)

    def set_coefficient(self, value: object, units: object) -> Self:
        """Set the strategy coefficient after unit normalization.

        Args:
            value: Nonnegative scalar coefficient.
            units: Unit accepted by the concrete strategy builder.

        Returns:
            This builder for fluent configuration.

        Raises:
            TypeError: If ``value`` is not a scalar real value.
            ValueError: If ``value`` is invalid or ``units`` is unsupported.
        """
        self.coefficient = self._converted(
            value, units, self.coefficient_units, "coefficient"
        )
        return self

    def set_coefficient_provenance(self, value: object) -> Self:
        """Set the required nonempty coefficient-origin description.

        Args:
            value: Nonempty, non-whitespace provenance string.

        Returns:
            This builder for fluent configuration.

        Raises:
            ValueError: If ``value`` is not a nonempty string.
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError("coefficient_provenance must be a nonempty string")
        self.coefficient_provenance = value
        return self

    def set_precursor_number_concentration_lower(
        self, value: object, units: object
    ) -> Self:
        """Set the lower precursor-number-concentration endpoint in #/m³.

        Args:
            value: Nonnegative endpoint value.
            units: ``"1/m^3"`` or ``"1/cm^3"``.

        Returns:
            This builder for fluent configuration.
        """
        self.precursor_number_concentration_lower = self._converted(
            value,
            units,
            {"1/m^3", "1/cm^3"},
            "precursor_number_concentration_lower",
        )
        return self

    def set_precursor_number_concentration_upper(
        self, value: object, units: object
    ) -> Self:
        """Set the upper precursor-number-concentration endpoint in #/m³.

        Args:
            value: Nonnegative endpoint value.
            units: ``"1/m^3"`` or ``"1/cm^3"``.

        Returns:
            This builder for fluent configuration.
        """
        self.precursor_number_concentration_upper = self._converted(
            value,
            units,
            {"1/m^3", "1/cm^3"},
            "precursor_number_concentration_upper",
        )
        return self

    def _set_positive(
        self,
        name: str,
        value: object,
        units: object,
        accepted: set[str],
    ) -> float:
        normalized_unit = self._unit(units, accepted, name)
        normalized = _positive_scalar(value, name)
        if normalized_unit == "nm":
            return normalized * 1e-9
        return normalized

    def set_temperature_lower(self, value: object, units: object) -> Self:
        """Set the positive lower temperature endpoint in K.

        Args:
            value: Positive temperature endpoint.
            units: Required ``"K"`` unit.

        Returns:
            This builder for fluent configuration.
        """
        self.temperature_lower = self._set_positive(
            "temperature_lower", value, units, {"K"}
        )
        return self

    def set_temperature_upper(self, value: object, units: object) -> Self:
        """Set the positive upper temperature endpoint in K.

        Args:
            value: Positive temperature endpoint.
            units: Required ``"K"`` unit.

        Returns:
            This builder for fluent configuration.
        """
        self.temperature_upper = self._set_positive(
            "temperature_upper", value, units, {"K"}
        )
        return self

    def set_formation_diameter(self, value: object, units: object) -> Self:
        """Set the positive formation mobility diameter in m.

        Args:
            value: Positive formation diameter.
            units: ``"m"`` or ``"nm"``.

        Returns:
            This builder for fluent configuration.
        """
        self.formation_diameter = self._set_positive(
            "formation_diameter", value, units, {"m", "nm"}
        )
        return self

    def set_saturation_lower(self, value: object, units: object) -> Self:
        """Set the nonnegative lower dimensionless saturation endpoint.

        Args:
            value: Nonnegative saturation endpoint.
            units: Required ``"dimensionless"`` unit.

        Returns:
            This builder for fluent configuration.
        """
        self.saturation_lower = self._converted(
            value, units, {"dimensionless"}, "saturation_lower"
        )
        return self

    def set_saturation_upper(self, value: object, units: object) -> Self:
        """Set the nonnegative upper dimensionless saturation endpoint.

        Args:
            value: Nonnegative saturation endpoint.
            units: Required ``"dimensionless"`` unit.

        Returns:
            This builder for fluent configuration.
        """
        self.saturation_upper = self._converted(
            value, units, {"dimensionless"}, "saturation_upper"
        )
        return self

    def set_survival_factor(self, value: object, units: object) -> Self:
        """Set the nonnegative dimensionless survival factor.

        Args:
            value: Nonnegative survival factor.
            units: Required ``"dimensionless"`` unit.

        Returns:
            This builder for fluent configuration.
        """
        self.survival_factor = self._converted(
            value, units, {"dimensionless"}, "survival_factor"
        )
        return self

    def set_injection_composition(self, value: object) -> Self:
        """Copy and validate future formed-particle composition metadata.

        Args:
            value: Iterable nonnegative molecule counts with at least one
                positive count.

        Returns:
            This builder for fluent configuration.

        Raises:
            TypeError: If ``value`` is not an iterable accepted by the record.
            ValueError: If the copied composition is invalid.
        """
        try:
            counts: tuple[object, ...] = tuple(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise TypeError(
                "injection_composition must be a sequence"
            ) from error
        self.injection_composition = InjectionComposition(counts)  # type: ignore[arg-type]
        return self

    def set_diameter_convention(self, value: object) -> Self:
        """Set the required ``"mobility_diameter"`` convention.

        Args:
            value: Required convention identifier.

        Returns:
            This builder for fluent configuration.

        Raises:
            ValueError: If ``value`` is not ``"mobility_diameter"``.
        """
        if value != "mobility_diameter":
            raise ValueError("diameter_convention must be mobility_diameter")
        self.diameter_convention = "mobility_diameter"
        return self

    @staticmethod
    def _validate_schema(parameters: dict[str, object]) -> None:
        value_keys = set(_REQUIRED) | _SCALARS | {"diameter_convention"}
        valid = value_keys | {f"{key}_units" for key in _SCALARS}
        unknown = set(parameters) - valid
        if unknown:
            raise ValueError(f"Unknown parameter(s): {sorted(unknown)}")
        for key in _SCALARS:
            unit_key = f"{key}_units"
            if unit_key in parameters and key not in parameters:
                raise ValueError(f"{unit_key} requires {key}")
            if key in parameters and unit_key not in parameters:
                raise ValueError(f"{key} requires {unit_key}")
        missing = [key for key in _REQUIRED if key not in parameters]
        if missing:
            message = ", ".join(missing)
            raise ValueError(f"Missing required parameter(s): {message}")
        lower = "saturation_lower" in parameters
        upper = "saturation_upper" in parameters
        if lower != upper:
            raise ValueError(
                "saturation_lower and saturation_upper must be supplied "
                "together"
            )

    def _set_mapping_scalar(
        self,
        key: str,
        value: object,
        units: object,
    ) -> None:
        setters = {
            "coefficient": self.set_coefficient,
            "precursor_number_concentration_lower": (
                self.set_precursor_number_concentration_lower
            ),
            "precursor_number_concentration_upper": (
                self.set_precursor_number_concentration_upper
            ),
            "temperature_lower": self.set_temperature_lower,
            "temperature_upper": self.set_temperature_upper,
            "formation_diameter": self.set_formation_diameter,
            "saturation_lower": self.set_saturation_lower,
            "saturation_upper": self.set_saturation_upper,
            "survival_factor": self.set_survival_factor,
        }
        setters[key](value, units)

    def set_parameters(self, parameters: object) -> Self:
        """Atomically configure this builder from its exact mapping schema.

        Scalar physical values require matching ``*_units`` entries. Required
        values, paired saturation bounds, and all units are validated on a
        staging builder before this builder is changed.

        Args:
            parameters: Dictionary containing exactly supported configuration
                keys and required unit metadata.

        Returns:
            This builder after successful replacement of its configuration.

        Raises:
            TypeError: If ``parameters`` is not a dictionary.
            ValueError: If keys, units, required values, or paired bounds are
                invalid.
        """
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dict")
        self._validate_schema(parameters)
        staged = type(self)()
        for key in _SCALARS:
            if key in parameters:
                staged._set_mapping_scalar(
                    key,
                    parameters[key],
                    parameters[f"{key}_units"],
                )
        staged.set_coefficient_provenance(parameters["coefficient_provenance"])
        staged.set_injection_composition(parameters["injection_composition"])
        staged.set_diameter_convention(
            parameters.get("diameter_convention", "mobility_diameter")
        )
        for name in _STAGED_ATTRIBUTES:
            setattr(self, name, getattr(staged, name))
        return self

    def _configuration(
        self,
    ) -> tuple[
        NucleationValidityDomain,
        InjectionComposition,
        FormationMetadata,
        float,
    ]:
        self.pre_build_check()
        if (self.saturation_lower is None) != (self.saturation_upper is None):
            raise ValueError(
                "saturation_lower and saturation_upper must be supplied "
                "together"
            )
        concentration_lower = cast(
            float, self.precursor_number_concentration_lower
        )
        concentration_upper = cast(
            float, self.precursor_number_concentration_upper
        )
        temperature_lower = cast(float, self.temperature_lower)
        temperature_upper = cast(float, self.temperature_upper)
        composition_record = cast(
            InjectionComposition,
            self.injection_composition,
        )
        formation_diameter = cast(float, self.formation_diameter)
        saturation = None
        if self.saturation_lower is not None:
            saturation = ClosedInterval(
                self.saturation_lower,
                cast(float, self.saturation_upper),
            )
        domain = NucleationValidityDomain(
            ClosedInterval(
                concentration_lower,
                concentration_upper,
            ),
            ClosedInterval(temperature_lower, temperature_upper),
            saturation,
        )
        composition = InjectionComposition(
            tuple(composition_record.molecule_counts)
        )
        metadata = FormationMetadata(
            formation_diameter,
            self.diameter_convention or "mobility_diameter",
        )
        survival_factor = (
            1.0 if self.survival_factor is None else self.survival_factor
        )
        return domain, composition, metadata, survival_factor


class ActivationNucleationBuilder(_NucleationStrategyBuilder):
    """Build immutable activation potential-rate strategies.

    The activation coefficient must use ``"s^-1"`` and all retained physical
    values are normalized to SI units before construction.
    """

    coefficient_units = {"s^-1"}

    def build(self) -> ActivationNucleationStrategy:
        """Build a fully validated immutable activation strategy.

        Returns:
            Activation strategy with normalized metadata and a survival default
            of ``1.0`` when no survival factor was supplied.

        Raises:
            ValueError: If required builder fields or paired saturation bounds
                are missing or invalid.
        """
        domain, composition, metadata, survival = self._configuration()
        return ActivationNucleationStrategy(
            cast(float, self.coefficient),
            domain,
            composition,
            metadata,
            survival,
            cast(str, self.coefficient_provenance),
        )


class KineticNucleationBuilder(_NucleationStrategyBuilder):
    """Build immutable kinetic potential-rate strategies.

    The kinetic coefficient accepts ``"m^3/s"`` or ``"cm^3/s"`` and is
    normalized to m³/s before construction.
    """

    coefficient_units = {"m^3/s", "cm^3/s"}

    def build(self) -> KineticNucleationStrategy:
        """Build a fully validated immutable kinetic strategy.

        Returns:
            Kinetic strategy with normalized metadata and a survival default of
            ``1.0`` when no survival factor was supplied.

        Raises:
            ValueError: If required builder fields or paired saturation bounds
                are missing or invalid.
        """
        domain, composition, metadata, survival = self._configuration()
        return KineticNucleationStrategy(
            cast(float, self.coefficient),
            domain,
            composition,
            metadata,
            survival,
            cast(str, self.coefficient_provenance),
        )


class NucleationSourceConfigBuilder(BuilderABC):
    """Build immutable P4 source-selection metadata.

    This builder selects a supported potential-rate strategy and precursor
    index only. It does not expose P2/P3 particle-source finalization or
    mutation behavior.
    """

    def __init__(self) -> None:
        """Initialize an unconfigured P4 source-selection builder."""
        super().__init__(required_parameters=["strategy", "precursor_index"])
        self.strategy: NucleationStrategy | None = None
        self.precursor_index: int | None = None

    def set_strategy(self, strategy: object) -> Self:
        """Set the candidate supported nucleation strategy.

        Args:
            strategy: Activation or kinetic strategy to select.

        Returns:
            This builder for fluent configuration.
        """
        self.strategy = strategy  # type: ignore[assignment]
        return self

    def set_precursor_index(self, precursor_index: object) -> Self:
        """Set the candidate nonnegative precursor species index.

        Args:
            precursor_index: Index validated when the configuration is built.

        Returns:
            This builder for fluent configuration.
        """
        self.precursor_index = precursor_index  # type: ignore[assignment]
        return self

    def set_parameters(self, parameters: object) -> Self:
        """Atomically configure the exact two-key source metadata schema.

        Args:
            parameters: Dictionary containing only ``strategy`` and
                ``precursor_index``.

        Returns:
            This builder after a valid configuration is staged and copied.

        Raises:
            TypeError: If ``parameters`` is not a dictionary.
            ValueError: If keys or source-selection values are invalid.
        """
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dict")
        if set(parameters) != {"strategy", "precursor_index"}:
            raise ValueError(
                "parameters must contain only strategy and precursor_index"
            )
        staged = type(self)()
        staged.set_strategy(parameters["strategy"])
        staged.set_precursor_index(parameters["precursor_index"])
        staged.build()
        self.strategy = staged.strategy
        self.precursor_index = staged.precursor_index
        return self

    def build(self) -> NucleationSourceConfig:
        """Build validated immutable P4 source-selection metadata.

        Returns:
            Metadata selecting a supported strategy and precursor index.

        Raises:
            ValueError: If required builder fields are missing or invalid.
            TypeError: If the precursor index is not an accepted integer.
        """
        self.pre_build_check()
        return NucleationSourceConfig(
            cast(NucleationStrategy, self.strategy),
            cast(int, self.precursor_index),
        )
