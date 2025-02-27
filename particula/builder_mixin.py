"""Mixin classes for Builder classes to set attributes and units."""

# pylint: disable=too-few-public-methods

import logging
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs
from particula.util.converting.convert_units import get_unit_conversion

logger = logging.getLogger("particula")


class BuilderDensityMixin:
    """Mixin class for Builder classes to set density and density_units.

    Methods:
        set_density: Set the density attribute and units.
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

        Args:
            density : Density of the particle.
            density_units : Units of the density. Default is *kg/m^3*
        """
        if density_units == "kg/m^3":
            self.density = density
            return self
        self.density = density * get_unit_conversion(density_units, "kg/m^3")
        return self


class BuilderSurfaceTensionMixin:
    """Mixin class for Builder classes to set surface_tension.

    Methods:
        set_surface_tension: Set the surface_tension attribute and units.
    """

    def __init__(self):
        self.surface_tension = None

    @validate_inputs({"surface_tension": "positive"})
    def set_surface_tension(
        self,
        surface_tension: Union[float, NDArray[np.float64]],
        surface_tension_units: str,
    ):
        """Set the surface tension of the particle in N/m.

        Args:
            surface_tension : Surface tension of the particle.
            surface_tension_units : Surface tension units. Default is *N/m*.
        """
        if surface_tension_units == "N/m":
            self.surface_tension = surface_tension
            return self
        self.surface_tension = surface_tension * get_unit_conversion(
            surface_tension_units, "N/m"
        )
        return self


class BuilderMolarMassMixin:
    """Mixin class for Builder classes to set molar_mass and molar_mass_units.

    Methods:
        set_molar_mass: Set the molar_mass attribute and units.
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

        Args:
            molar_mass : Molar mass of the particle.
            molar_mass_units : Units of the molar mass. Default is *kg/mol*.
        """
        if molar_mass_units == "kg/mol":
            self.molar_mass = molar_mass
            return self
        self.molar_mass = molar_mass * get_unit_conversion(
            molar_mass_units, "kg/mol"
        )
        return self


class BuilderConcentrationMixin:
    """Mixin class for Builder classes to set concentration and
    concentration_units.

    Args:
        default_units : Default units of concentration. Default is *kg/m^3*.

    Methods:
        set_concentration: Set the concentration attribute and units.
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
        """Set the concentration.

        Args:
            concentration : Concentration in the mixture.
            concentration_units : Units of the concentration.
                Default is *kg/m^3*.
        """
        if concentration_units == self.default_units:
            self.concentration = concentration
            return self
        self.concentration = concentration * get_unit_conversion(
            concentration_units, self.default_units
        )
        return self


class BuilderChargeMixin:
    """Mixin class for Builder classes to set charge and charge_units.

    Methods:
        set_charge: Set the charge attribute and units.
    """

    def __init__(self):
        self.charge = None

    def set_charge(
        self,
        charge: Union[float, NDArray[np.float64]],
        charge_units: Optional[str] = None,
    ):
        """Set the number of elemental charges on the particle.

        Args:
            charge : Charge of the particle [C].
            charge_units : Not used. (for interface consistency)
        """
        if charge_units is not None:
            logger.warning("Ignoring units for charge parameter.")
        self.charge = charge
        return self


class BuilderMassMixin:
    """Mixin class for Builder classes to set mass and mass_units.

    Methods:
        set_mass: Set the mass attribute and units.
    """

    def __init__(self):
        self.mass = None

    @validate_inputs({"mass": "nonnegative"})
    def set_mass(
        self,
        mass: Union[float, NDArray[np.float64]],
        mass_units: str,
    ):
        """Set the mass of the particle in kg.

        Args:
            mass : Mass of the particle.
            mass_units : Units of the mass. Default is *kg*.
        """
        if mass_units == "kg":
            self.mass = mass
            return self
        self.mass = mass * get_unit_conversion(mass_units, "kg")
        return self


class BuilderVolumeMixin:
    """Mixin class for Builder classes to set volume and volume_units.

    Methods:
        set_volume: Set the volume attribute and units.
    """

    def __init__(self):
        self.volume = None

    @validate_inputs({"volume": "nonnegative"})
    def set_volume(
        self,
        volume: Union[float, NDArray[np.float64]],
        volume_units: str,
    ):
        """Set the volume in m^3.

        Args:
            volume : Volume.
            volume_units : Units of the volume. Default is *m^3*.
        """
        if volume_units == "m^3":
            self.volume = volume
            return self
        self.volume = volume * get_unit_conversion(volume_units, "m^3")
        return self


class BuilderRadiusMixin:
    """Mixin class for Builder classes to set radius and radius_units.

    Methods:
        set_radius: Set the radius attribute and units.
    """

    def __init__(self):
        self.radius = None

    @validate_inputs({"radius": "nonnegative"})
    def set_radius(
        self,
        radius: Union[float, NDArray[np.float64]],
        radius_units: str,
    ):
        """Set the radius of the particle in meters.

        Args:
            radius : Radius of the particle.
            radius_units : Units of the radius. Default is *m*.
        """
        if radius_units == "m":
            self.radius = radius
            return self
        self.radius = radius * get_unit_conversion(radius_units, "m")
        return self


class BuilderTemperatureMixin:
    """Mixin class for AtmosphereBuilder to set temperature.

    Methods:
        set_temperature: Set the temperature attribute and units.
    """

    def __init__(self):
        self.temperature = None

    @validate_inputs({"temperature": "finite"})
    def set_temperature(
        self, temperature: float, temperature_units: str = "K"
    ):
        """Set the temperature of the atmosphere.

        Args:
            temperature : Temperature of the gas mixture.
            temperature_units : Units of the temperature.
                Options include 'degC', 'degF', 'degR', 'K'. Default is 'K'.

        Returns:
            AtmosphereBuilderMixin: This object instance with updated
                temperature.

        Raises:
            ValueError: If the converted temperature is below absolute zero.
        """
        if temperature_units == "K":
            self.temperature = temperature
            return self
        self.temperature = get_unit_conversion(
            temperature_units, "K", temperature
        )
        return self


class BuilderPressureMixin:
    """Mixin class for AtmosphereBuilder to set total pressure.

    Methods:
        set_pressure: Set the total pressure attribute and units.
    """

    def __init__(self):
        self.pressure = None

    @validate_inputs({"pressure": "nonnegative"})
    def set_pressure(
        self,
        pressure: Union[float, NDArray[np.float64]],
        pressure_units: str,
    ):
        """Set the total pressure of the atmosphere.

        Args:
            pressure : Total pressure of the gas mixture.
            pressure_units : Units of the pressure. Options include
                'Pa', 'kPa', 'MPa', 'psi', 'bar', 'atm'. Default is 'Pa'.

        Returns:
            AtmosphereBuilderMixin: This object instance with updated pressure.
        """
        if pressure_units == "Pa":
            self.pressure = pressure
            return self
        self.pressure = pressure * get_unit_conversion(pressure_units, "Pa")
        return self


class BuilderLognormalMixin:
    """Mixin class for Builder classes to set lognormal distributions.

    Methods:
        set_mode: Set the mode attribute and units.
        set_geometric_standard_deviation: Set the geometric standard deviation
            attribute and units.
        set_number_concentration: Set the number concentration attribute and
            units.
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
        """Set the mode for distribution.

        Args:
            mode : The modes for the radius.
            mode_units : The units for the modes, default is [m].
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
        """Set the geometric standard deviation for the distribution.

        Args:
            geometric_standard_deviation : The geometric standard deviation for
                the radius.
            geometric_standard_deviation_units : Optional, ignored units for
                geometric standard deviation [dimensionless].
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
        """Set the number concentration for the distribution.

        Args:
            number_concentration : The number concentration for the radius.
            number_concentration_units : The units for the number
                concentration, "1/m^3".
        """
        if number_concentration_units in ("1/m^3", "m^-3"):
            self.number_concentration = number_concentration
            return self
        self.number_concentration = number_concentration * get_unit_conversion(
            number_concentration_units, "1/m^3"
        )
        return self


class BuilderParticleResolvedCountMixin:
    """Mixin class for Builder classes to set particle_resolved_count.

    Methods:
        set_particle_resolved_count: Set the number of particles to resolve.
    """

    def __init__(self):
        self.particle_resolved_count = None

    @validate_inputs({"particle_resolved_count": "positive"})
    def set_particle_resolved_count(
        self,
        particle_resolved_count: int,
        particle_resolved_count_units: Optional[str] = None,
    ):
        """Set the number of particles to resolve.

        Args:
            particle_resolved_count : The number of particles to resolve.
            particle_resolved_count_units : Ignored units for particle
                resolved.
        """
        if particle_resolved_count_units is not None:
            logger.warning("Ignoring units for particle resolved count.")
        self.particle_resolved_count = particle_resolved_count
        return self
