"""Builder classes for latent heat strategy objects.

These builders mirror the vapor pressure builder pattern. They validate
required parameters, optionally convert units via
:func:`get_unit_conversion`, and construct latent heat strategies that return
J/kg values.
"""

import logging
from typing import cast

from particula.abc_builder import BuilderABC
from particula.gas.latent_heat_strategies import (
    ConstantLatentHeat,
    LinearLatentHeat,
    PowerLawLatentHeat,
)
from particula.util import get_unit_conversion
from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")


class ConstantLatentHeatBuilder(BuilderABC):
    """Build ConstantLatentHeat strategies with validated inputs.

    Use :meth:`set_latent_heat_ref` to set the reference latent heat in J/kg
    before calling :meth:`build`.

    Attributes:
        latent_heat_ref: Reference latent heat in J/kg once set.
    """

    def __init__(self) -> None:
        """Initialize the constant latent heat builder."""
        required_parameters = ["latent_heat_ref"]
        super().__init__(required_parameters)
        self.latent_heat_ref: float | None = None

    @validate_inputs({"latent_heat_ref": "positive"})
    def set_latent_heat_ref(
        self,
        latent_heat_ref: float,
        latent_heat_ref_units: str | None = None,
    ) -> "ConstantLatentHeatBuilder":
        """Set the reference latent heat value.

        Args:
            latent_heat_ref: Reference latent heat value.
            latent_heat_ref_units: Units for the reference latent heat.
                Defaults to J/kg when None.

        Returns:
            The builder instance for method chaining.

        Raises:
            ValueError: If ``latent_heat_ref`` is not positive.
            pint.errors.UndefinedUnitError: If ``latent_heat_ref_units`` is
                invalid and pint is available.
        """
        if latent_heat_ref_units is None or latent_heat_ref_units == "J/kg":
            self.latent_heat_ref = latent_heat_ref
            return self
        self.latent_heat_ref = float(
            latent_heat_ref
            * get_unit_conversion(
                latent_heat_ref_units,
                "J/kg",
            )
        )
        return self

    def build(self) -> ConstantLatentHeat:
        """Validate parameters and build the ConstantLatentHeat strategy.

        Returns:
            The configured ConstantLatentHeat strategy.

        Raises:
            ValueError: If required parameters are missing.
        """
        self.pre_build_check()
        return ConstantLatentHeat(
            latent_heat_ref=cast(float, self.latent_heat_ref)
        )


class LinearLatentHeatBuilder(BuilderABC):
    """Build LinearLatentHeat strategies with validated inputs.

    Requires a reference latent heat, linear slope, and reference temperature.

    Attributes:
        latent_heat_ref: Reference latent heat in J/kg once set.
        slope: Linear slope in J/(kg*K) once set.
        temperature_ref: Reference temperature in K once set.
    """

    def __init__(self) -> None:
        """Initialize the linear latent heat builder."""
        required_parameters = ["latent_heat_ref", "slope", "temperature_ref"]
        super().__init__(required_parameters)
        self.latent_heat_ref: float | None = None
        self.slope: float | None = None
        self.temperature_ref: float | None = None

    @validate_inputs({"latent_heat_ref": "positive"})
    def set_latent_heat_ref(
        self,
        latent_heat_ref: float,
        latent_heat_ref_units: str | None = None,
    ) -> "LinearLatentHeatBuilder":
        """Set the reference latent heat value.

        Args:
            latent_heat_ref: Reference latent heat value.
            latent_heat_ref_units: Units for the reference latent heat.
                Defaults to J/kg when None.

        Returns:
            The builder instance for method chaining.

        Raises:
            ValueError: If ``latent_heat_ref`` is not positive.
            pint.errors.UndefinedUnitError: If ``latent_heat_ref_units`` is
                invalid and pint is available.
        """
        if latent_heat_ref_units is None or latent_heat_ref_units == "J/kg":
            self.latent_heat_ref = latent_heat_ref
            return self
        self.latent_heat_ref = float(
            latent_heat_ref
            * get_unit_conversion(
                latent_heat_ref_units,
                "J/kg",
            )
        )
        return self

    @validate_inputs({"slope": "finite"})
    def set_slope(
        self,
        slope: float,
        slope_units: str | None = None,
    ) -> "LinearLatentHeatBuilder":
        """Set the linear slope value.

        Args:
            slope: Linear slope value.
            slope_units: Units for the slope. Defaults to J/(kg*K).

        Returns:
            The builder instance for method chaining.

        Raises:
            ValueError: If ``slope`` is not finite.
            pint.errors.UndefinedUnitError: If ``slope_units`` is invalid and
                pint is available.
        """
        if slope_units is None or slope_units == "J/(kg*K)":
            self.slope = slope
            return self
        self.slope = float(
            slope
            * get_unit_conversion(
                slope_units,
                "J/(kg*K)",
            )
        )
        return self

    @validate_inputs({"temperature_ref": "positive"})
    def set_temperature_ref(
        self,
        temperature_ref: float,
        temperature_ref_units: str | None = None,
    ) -> "LinearLatentHeatBuilder":
        """Set the reference temperature.

        Args:
            temperature_ref: Reference temperature value.
            temperature_ref_units: Units for the temperature. Defaults to K.

        Returns:
            The builder instance for method chaining.

        Raises:
            ValueError: If ``temperature_ref`` is not positive.
            pint.errors.UndefinedUnitError: If ``temperature_ref_units`` is
                invalid and pint is available.
        """
        if temperature_ref_units is None or temperature_ref_units == "K":
            self.temperature_ref = temperature_ref
            return self
        self.temperature_ref = float(
            get_unit_conversion(
                temperature_ref_units,
                "K",
                temperature_ref,
            )
        )
        return self

    def build(self) -> LinearLatentHeat:
        """Validate parameters and build the LinearLatentHeat strategy.

        Returns:
            The configured LinearLatentHeat strategy.

        Raises:
            ValueError: If required parameters are missing.
        """
        self.pre_build_check()
        return LinearLatentHeat(
            latent_heat_ref=cast(float, self.latent_heat_ref),
            slope=cast(float, self.slope),
            temperature_ref=cast(float, self.temperature_ref),
        )


class PowerLawLatentHeatBuilder(BuilderABC):
    """Build PowerLawLatentHeat strategies with validated inputs.

    Requires a reference latent heat, critical temperature, and exponent.

    Attributes:
        latent_heat_ref: Reference latent heat in J/kg once set.
        critical_temperature: Critical temperature in K once set.
        beta: Power-law exponent (dimensionless) once set.
    """

    def __init__(self) -> None:
        """Initialize the power-law latent heat builder."""
        required_parameters = [
            "latent_heat_ref",
            "critical_temperature",
            "beta",
        ]
        super().__init__(required_parameters)
        self.latent_heat_ref: float | None = None
        self.critical_temperature: float | None = None
        self.beta: float | None = None

    @validate_inputs({"latent_heat_ref": "positive"})
    def set_latent_heat_ref(
        self,
        latent_heat_ref: float,
        latent_heat_ref_units: str | None = None,
    ) -> "PowerLawLatentHeatBuilder":
        """Set the reference latent heat value.

        Args:
            latent_heat_ref: Reference latent heat value.
            latent_heat_ref_units: Units for the reference latent heat.
                Defaults to J/kg when None.

        Returns:
            The builder instance for method chaining.

        Raises:
            ValueError: If ``latent_heat_ref`` is not positive.
            pint.errors.UndefinedUnitError: If ``latent_heat_ref_units`` is
                invalid and pint is available.
        """
        if latent_heat_ref_units is None or latent_heat_ref_units == "J/kg":
            self.latent_heat_ref = latent_heat_ref
            return self
        self.latent_heat_ref = float(
            latent_heat_ref
            * get_unit_conversion(
                latent_heat_ref_units,
                "J/kg",
            )
        )
        return self

    @validate_inputs({"critical_temperature": "positive"})
    def set_critical_temperature(
        self,
        critical_temperature: float,
        critical_temperature_units: str | None = None,
    ) -> "PowerLawLatentHeatBuilder":
        """Set the critical temperature value.

        Args:
            critical_temperature: Critical temperature value.
            critical_temperature_units: Units for the temperature. Defaults to
                K.

        Returns:
            The builder instance for method chaining.

        Raises:
            ValueError: If ``critical_temperature`` is not positive.
            pint.errors.UndefinedUnitError: If ``critical_temperature_units``
                is invalid and pint is available.
        """
        if (
            critical_temperature_units is None
            or critical_temperature_units == "K"
        ):
            self.critical_temperature = critical_temperature
            return self
        self.critical_temperature = float(
            get_unit_conversion(
                critical_temperature_units,
                "K",
                critical_temperature,
            )
        )
        return self

    @validate_inputs({"beta": "nonnegative"})
    def set_beta(
        self,
        beta: float,
        beta_units: str | None = None,
    ) -> "PowerLawLatentHeatBuilder":
        """Set the power-law exponent.

        Args:
            beta: Power-law exponent (dimensionless).
            beta_units: Units for the exponent. Ignored if provided.

        Returns:
            The builder instance for method chaining.

        Raises:
            ValueError: If ``beta`` is negative.
        """
        if beta_units is not None:
            logger.warning("Ignoring units for beta: '%s'.", beta_units)
        self.beta = beta
        return self

    def build(self) -> PowerLawLatentHeat:
        """Validate parameters and build the PowerLawLatentHeat strategy.

        Returns:
            The configured PowerLawLatentHeat strategy.

        Raises:
            ValueError: If required parameters are missing.
        """
        self.pre_build_check()
        return PowerLawLatentHeat(
            latent_heat_ref=cast(float, self.latent_heat_ref),
            critical_temperature=cast(float, self.critical_temperature),
            beta=cast(float, self.beta),
        )
