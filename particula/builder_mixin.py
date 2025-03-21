"""Mixin classes for Builder classes to set attributes and units."""

# pylint: disable=too-few-public-methods

import logging
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs
from particula.util.convert_units import get_unit_conversion

logger = logging.getLogger("particula")


class BuilderDensityMixin:
    """
    Mixin class for setting density and density_units.

    This class provides a method to assign a particle's density in kg/m^3,
    optionally converting from other units.

    Attributes:
        - density : Stores the density in kg/m^3 after conversion.

    Methods:
        - set_density: Assign the density attribute, converting from given
            units to kg/m^3.

    Examples:
        ```py title="Setting particle density"
        builder = MyBuilderClass()
        builder.set_density(1000, "g/m^3")
        # density is now 1.0 kg/m^3
        ```
    """

    def __init__(self):
        self.density = None

    @validate_inputs({"density": "positive"})
    def set_density(
        self,
        density: Union[float, NDArray[np.float64]],
        density_units: str,
    ):
        """
        Set the density of the particle in kg/m^3.

        Arguments:
            - density : Density value.
            - density_units : Units of the provided density.
                Default is "kg/m^3".

        Returns:
            - self : The class instance for method chaining.

        Examples:
            ```py
            builder.set_density(1000, "g/m^3")
            # density is now 1.0 kg/m^3
            ```
        """
        if density_units == "kg/m^3":
            self.density = density
            return self
        self.density = density * get_unit_conversion(density_units, "kg/m^3")
        return self


class BuilderSurfaceTensionMixin:
    """
    Mixin class for setting surface_tension.

    This class provides a method to assign a particle's surface tension,
    in N/m units, optionally converting from other units.

    Attributes:
        - surface_tension : Stores the surface tension in N/m after conversion.

    Methods:
        - set_surface_tension: Assign the surface_tension, converting from
            other units as needed.

    References:
        - No references available yet.
    """

    def __init__(self):
        self.surface_tension = None

    @validate_inputs({"surface_tension": "positive"})
    def set_surface_tension(
        self,
        surface_tension: Union[float, NDArray[np.float64]],
        surface_tension_units: str,
    ):
        """
        Set the surface tension of the particle in N/m.

        Arguments:
            - surface_tension : Surface tension value.
            - surface_tension_units : Units of the provided surface tension.
                Default is "N/m".

        Returns:
            - self : The class instance for method chaining.

        Examples:
            ```py
            builder.set_surface_tension(0.072, "N/m")
            ```
        """
        if surface_tension_units == "N/m":
            self.surface_tension = surface_tension
            return self
        self.surface_tension = surface_tension * get_unit_conversion(
            surface_tension_units, "N/m"
        )
        return self


class BuilderMolarMassMixin:
    """
    Mixin class for setting molar_mass and molar_mass_units.

    This class provides a method to assign a particle's molar mass in kg/mol,
    optionally converting from other units.

    Attributes:
        - molar_mass : Stores the molar mass in kg/mol.

    Methods:
        - set_molar_mass: Assign the molar_mass, converting units as necessary.

    References:
        - No references available yet.
    """

    def __init__(self):
        self.molar_mass = None

    @validate_inputs({"molar_mass": "positive"})
    def set_molar_mass(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        molar_mass_units: str,
    ):
        """
        Set the molar mass of the particle in kg/mol.

        Arguments:
            - molar_mass : Molar mass value.
            - molar_mass_units : Units of the provided molar mass.
                Default is "kg/mol".

        Returns:
            - self : The class instance for method chaining.

        Examples:
            ```py
            builder.set_molar_mass(18, "g/mol")
            # molar_mass is now 0.018 kg/mol
            ```
        """
        if molar_mass_units == "kg/mol":
            self.molar_mass = molar_mass
            return self
        self.molar_mass = molar_mass * get_unit_conversion(
            molar_mass_units, "kg/mol"
        )
        return self


class BuilderConcentrationMixin:
    """
    Mixin class for setting concentration in a mixture.

    This class provides a method to assign a particle or species concentration
    in kg/m^3 by default, optionally converting from other units.

    Attributes:
        - concentration : The concentration in the default units.
        - default_units : The default concentration units (e.g., "kg/m^3").

    Methods:
        - set_concentration: Assign the concentration, converting units
            as needed.

    Examples:
        ```py title="Example usage"
        builder = MyBuilderClass(default_units="g/m^3")
        builder.set_concentration(500, "g/m^3")
        ```
    """

    def __init__(self, default_units: str = "kg/m^3"):
        self.concentration = None
        self.default_units = default_units if default_units else "kg/m^3"

    @validate_inputs({"concentration": "nonnegative"})
    def set_concentration(
        self,
        concentration: Union[float, NDArray[np.float64]],
        concentration_units: str,
    ):
        """
        Set the concentration in the mixture.

        Arguments:
            - concentration : Concentration value.
            - concentration_units : Units of the provided concentration.

        Returns:
            - self : The class instance for method chaining.

        Examples:
            ```py
            builder.set_concentration(0.5, "kg/m^3")
            # stored as 0.5 in the default_units
            ```
        """
        if concentration_units == self.default_units:
            self.concentration = concentration
            return self
        self.concentration = concentration * get_unit_conversion(
            concentration_units, self.default_units
        )
        return self


class BuilderChargeMixin:
    """
    Mixin class for setting a particle's charge.

    This class provides a method to assign charge in terms of number of
    elemental charges (dimensionless), ignoring units.

    Attributes:
        - charge : The assigned charge.

    Methods:
        - set_charge: Assign the particle's charge.

    References:
        - No references available yet.
    """

    def __init__(self):
        self.charge = None

    def set_charge(
        self,
        charge: Union[float, NDArray[np.float64]],
        charge_units: Optional[str] = None,
    ):
        """
        Set the number of elemental charges on the particle.

        Arguments:
            - charge : Numeric value of the charge.
            - charge_units : Optional; if provided, a warning is logged
                and ignored.

        Returns:
            - self : The class instance for method chaining.

        Examples:
            ```py
            builder.set_charge(10)
            # charge is now 10 elementary charges
            ```
        """
        if charge_units is not None:
            logger.warning("Ignoring units for charge parameter.")
        self.charge = charge
        return self


class BuilderMassMixin:
    """
    Mixin class for setting particle mass in kg.

    This class provides a method to assign mass in kg, optionally converting
    from other units.

    Attributes:
        - mass : The mass of the particle in kg.

    Methods:
        - set_mass: Assign the mass, converting from specified units.
    """

    def __init__(self):
        self.mass = None

    @validate_inputs({"mass": "nonnegative"})
    def set_mass(
        self,
        mass: Union[float, NDArray[np.float64]],
        mass_units: str,
    ):
        """
        Set the mass of the particle in kg.

        Arguments:
            - mass : Numeric mass value.
            - mass_units : Units of the provided mass. Default is "kg".

        Returns:
            - self : The class instance for method chaining.

        Examples:
            ```py
            builder.set_mass(1.0, "g")
            # mass is now 0.001 kg
            ```
        """
        if mass_units == "kg":
            self.mass = mass
            return self
        self.mass = mass * get_unit_conversion(mass_units, "kg")
        return self


class BuilderVolumeMixin:
    """
    Mixin class for setting volume in m^3.

    This class provides a method to assign volume in m^3,
    optionally converting from other units.

    Attributes:
        - volume : The volume in m^3.

    Methods:
        - set_volume: Assign the volume, converting units as needed.
    """

    def __init__(self):
        self.volume = None

    @validate_inputs({"volume": "nonnegative"})
    def set_volume(
        self,
        volume: Union[float, NDArray[np.float64]],
        volume_units: str,
    ):
        """
        Set the volume in m^3.

        Arguments:
            - volume : Volume value.
            - volume_units : Units of the provided volume. Default is "m^3".

        Returns:
            - self : The class instance for method chaining.

        Examples:
            ```py
            builder.set_volume(1.0, "L")
            # volume is now 0.001 m^3
            ```
        """
        if volume_units == "m^3":
            self.volume = volume
            return self
        self.volume = volume * get_unit_conversion(volume_units, "m^3")
        return self


class BuilderRadiusMixin:
    """
    Mixin class for setting a particle's radius in meters.

    This class provides a method to assign radius in meters,
    optionally converting from other units.

    Attributes:
        - radius : The radius in meters.

    Methods:
        - set_radius: Assign the radius, converting units as needed.
    """

    def __init__(self):
        self.radius = None

    @validate_inputs({"radius": "nonnegative"})
    def set_radius(
        self,
        radius: Union[float, NDArray[np.float64]],
        radius_units: str,
    ):
        """
        Set the radius of the particle in meters.

        Arguments:
            - radius : Numeric radius value.
            - radius_units : Units of the provided radius. Default is "m".

        Returns:
            - self : The class instance for method chaining.

        Examples:
            ```py
            builder.set_radius(1.0, "um")
            # radius is now 1e-6 m
            ```
        """
        if radius_units == "m":
            self.radius = radius
            return self
        self.radius = radius * get_unit_conversion(radius_units, "m")
        return self


class BuilderTemperatureMixin:
    """
    Mixin class for setting temperature in Kelvin.

    This class provides a method to assign temperature in Kelvin,
    optionally converting from specified units such as 'degC', 'degF',
    'degR', or 'K'.

    Attributes:
        - temperature : The temperature in Kelvin.

    Methods:
        - set_temperature: Assign the temperature, converting units as needed.
    """

    def __init__(self):
        self.temperature = None

    @validate_inputs({"temperature": "finite"})
    def set_temperature(
        self, temperature: float, temperature_units: str = "K"
    ):
        """
        Set the temperature of the atmosphere in Kelvin.

        Arguments:
            - temperature : Numeric temperature value.
            - temperature_units : Units of the given temperature.
                Defaults to "K". Accepts "degC", "degF", "degR", or "K".

        Returns:
            - self : The class instance for method chaining.

        Raises:
            - ValueError : If the converted temperature is below absolute zero.

        Examples:
            ```py title="Setting temperature"
            builder.set_temperature(25, "degC")
            # temperature is now 298.15 K
            ```
        """
        if temperature_units == "K":
            self.temperature = temperature
            return self
        self.temperature = get_unit_conversion(
            temperature_units, "K", temperature
        )
        return self


class BuilderPressureMixin:
    """
    Mixin class for setting total pressure in Pa.

    This class provides a method to assign the total gas mixture pressure
    in pascals, optionally converting from units like 'kPa', 'MPa', 'psi',
    'bar', or 'atm'.

    Attributes:
        - pressure : The total pressure in Pa.

    Methods:
        - set_pressure: Assign the pressure, converting units as needed.
    """

    def __init__(self):
        self.pressure = None

    @validate_inputs({"pressure": "nonnegative"})
    def set_pressure(
        self,
        pressure: Union[float, NDArray[np.float64]],
        pressure_units: str,
    ):
        """
        Set the total pressure of the atmosphere.

        Arguments:
            - pressure : Numeric pressure value.
            - pressure_units : Units of the given pressure. Default is "Pa".

        Returns:
            - self : The class instance for method chaining.

        Examples:
            ```py
            builder.set_pressure(1.0, "bar")
            # pressure is now 1e5 Pa
            ```
        """
        if pressure_units == "Pa":
            self.pressure = pressure
            return self
        self.pressure = pressure * get_unit_conversion(pressure_units, "Pa")
        return self


class BuilderLognormalMixin:
    """
    Mixin class for setting lognormal distribution parameters.

    This class provides methods to assign and manage lognormal distribution
    parameters for particle radius, including the mode, geometric standard
    deviation, and number concentration.

    Attributes:
        - mode : Array of modes in meters.
        - number_concentration : Number concentration in 1/m^3.
        - geometric_standard_deviation : The dimensionless geometric std. dev.

    Methods:
        - set_mode: Assign the modal radius.
        - set_geometric_standard_deviation: Assign the geometric std. dev.
            (ignored units).
        - set_number_concentration: Assign the number concentration in 1/m^3.
    """

    def __init__(self):
        self.mode = None
        self.number_concentration = None
        self.geometric_standard_deviation = None

    @validate_inputs({"mode": "positive"})
    def set_mode(
        self,
        mode: NDArray[np.float64],
        mode_units: str,
    ):
        """
        Set the mode for the lognormal distribution in meters.

        Arguments:
            - mode : Array of modal radius values.
            - mode_units : Units of the provided mode. Default is "m".

        Returns:
            - self : The class instance for method chaining.

        Examples:
            ```py
            builder.set_mode(np.array([1e-8, 2e-8]), "m")
            # modes are now [1e-8, 2e-8] m
            ```
        """
        if mode_units == "m":
            self.mode = mode
            return self
        self.mode = mode * get_unit_conversion(mode_units, "m")
        return self

    @validate_inputs({"geometric_standard_deviation": "positive"})
    def set_geometric_standard_deviation(
        self,
        geometric_standard_deviation: NDArray[np.float64],
        geometric_standard_deviation_units: Optional[str] = None,
    ):
        """
        Set the geometric standard deviation for the lognormal distribution.

        Arguments:
            - geometric_standard_deviation : Dimensionless geometric std. dev.
            - geometric_standard_deviation_units : Ignored
                (for interface consistency).

        Returns:
            - self : The class instance for method chaining.

        Examples:
            ```py
            builder.set_geometric_standard_deviation(np.array([1.5, 2.0]))
            # geometric std dev is now [1.5, 2.0]
            ```
        """
        if geometric_standard_deviation_units is not None:
            logger.warning("Ignoring units for surface strategy parameter.")
        self.geometric_standard_deviation = geometric_standard_deviation
        return self

    @validate_inputs({"number_concentration": "positive"})
    def set_number_concentration(
        self,
        number_concentration: NDArray[np.float64],
        number_concentration_units: str,
    ):
        """
        Set the number concentration for the lognormal distribution in 1/m^3.

        Arguments:
            - number_concentration : Array of number concentration values.
            - number_concentration_units : Units of the concentration,
                must be "1/m^3" or equivalent.

        Returns:
            - self : The class instance for method chaining.

        Examples:
            ```py
            builder.set_number_concentration(np.array([1e6, 5e5]), "m^-3")
            # stored as [1e6, 5e5] 1/m^3
            ```
        """
        if number_concentration_units in {"1/m^3", "m^-3"}:
            self.number_concentration = number_concentration
            return self
        self.number_concentration = number_concentration * get_unit_conversion(
            number_concentration_units, "1/m^3"
        )
        return self


class BuilderParticleResolvedCountMixin:
    """
    Mixin class for setting a particle-resolved count.

    This class provides a method to define how many individual particles
    should be resolved in a simulation or model.

    Attributes:
        - particle_resolved_count : The number of particles to resolve.

    Methods:
        - set_particle_resolved_count: Assign the particle-resolved count.
    """

    def __init__(self):
        self.particle_resolved_count = None

    @validate_inputs({"particle_resolved_count": "positive"})
    def set_particle_resolved_count(
        self,
        particle_resolved_count: int,
        particle_resolved_count_units: Optional[str] = None,
    ):
        """
        Set the number of particles to resolve.

        Arguments:
            - particle_resolved_count : Positive integer count of particles.
            - particle_resolved_count_units : Ignored, for interface
                consistency.

        Returns:
            - self : The class instance for method chaining.

        Examples:
            ```py
            builder.set_particle_resolved_count(1000)
            ```
        """
        if particle_resolved_count_units is not None:
            logger.warning("Ignoring units for particle resolved count.")
        self.particle_resolved_count = particle_resolved_count
        return self
