"""Mixin classes for Builder classes to set attributes and units.
"""

# pylint: disable=too-few-public-methods

from typing import Optional, Union
import logging
from numpy.typing import NDArray
import numpy as np

from particula.util.input_handling import convert_units  # type: ignore

logger = logging.getLogger("particula")


class BuilderDensityMixin:
    """Mixin class for Builder classes to set density and density_units.

    Methods:
        set_density: Set the density attribute and units.
    """

    def __init__(self):
        self.density = None

    def set_density(
        self,
        density: Union[float, NDArray[np.float64]],
        density_units: Optional[str] = "kg/m^3",
    ):
        """Set the density of the particle in kg/m^3.

        Args:
            density: Density of the particle.
            density_units: Units of the density. Default is *kg/m^3*
        """
        if np.any(density < 0):
            error_message = "Density must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.density = density * convert_units(density_units, "kg/m^3")
        return self


class BuilderSurfaceTensionMixin:
    """Mixin class for Builder classes to set surface_tension.

    Methods:
    -------
        set_surface_tension: Set the surface_tension attribute and units.
    """

    def __init__(self):
        self.surface_tension = None

    def set_surface_tension(
        self,
        surface_tension: Union[float, NDArray[np.float64]],
        surface_tension_units: Optional[str] = "N/m",
    ):
        """Set the surface tension of the particle in N/m.

        Args:
            surface_tension: Surface tension of the particle.
            surface_tension_units: Surface tension units. Default is *N/m*.
        """
        if np.any(surface_tension < 0):
            error_message = "Surface tension must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.surface_tension = surface_tension * convert_units(
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

    def set_molar_mass(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        molar_mass_units: Optional[str] = "kg/mol",
    ):
        """Set the molar mass of the particle in kg/mol.

        Args:
        -----
        - molar_mass: Molar mass of the particle.
        - molar_mass_units: Units of the molar mass. Default is *kg/mol*.
        """
        if np.any(molar_mass < 0):
            error_message = "Molar mass must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.molar_mass = molar_mass * convert_units(
            molar_mass_units, "kg/mol"
        )
        return self


class BuilderConcentrationMixin:
    """Mixin class for Builder classes to set concentration and
    concentration_units.

    Args:
        default_units: Default units of concentration. Default is *kg/m^3*.

    Methods:
        set_concentration: Set the concentration attribute and units.
    """

    def __init__(self, default_units: Optional[str] = "kg/m^3"):
        self.concentration = None
        self.default_units = default_units if default_units else "kg/m^3"

    def set_concentration(
        self,
        concentration: Union[float, NDArray[np.float64]],
        concentration_units: Optional[str] = None,
    ):
        """Set the concentration.

        Args:
            concentration: Concentration in the mixture.
            concentration_units: Units of the concentration.
            Default is *kg/m^3*.
        """
        if concentration_units is None:
            concentration_units = self.default_units
        if np.any(concentration < 0):
            error_message = "Concentration must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.concentration = concentration * convert_units(
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
            charge: Charge of the particle [C].
            charge_units: Not used. (for interface consistency)
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

    def set_mass(
        self,
        mass: Union[float, NDArray[np.float64]],
        mass_units: Optional[str] = "kg",
    ):
        """Set the mass of the particle in kg.

        Args:
            mass: Mass of the particle.
            mass_units: Units of the mass. Default is *kg*.

        Raises:
            ValueError: If mass is negative
        """
        if np.any(mass < 0):
            error_message = "Mass must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.mass = mass * convert_units(mass_units, "kg")
        return self


class BuilderVolumeMixin:
    """Mixin class for Builder classes to set volume and volume_units.

    Methods:
        set_volume: Set the volume attribute and units.
    """

    def __init__(self):
        self.volume = None

    def set_volume(
        self,
        volume: Union[float, NDArray[np.float64]],
        volume_units: Optional[str] = "m^3",
    ):
        """Set the volume in m^3.

        Args:
            volume: Volume.
            volume_units: Units of the volume. Default is *m^3*.

        Raises:
            ValueError: If volume is negative
        """
        if np.any(volume < 0):
            error_message = "Volume must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.volume = volume * convert_units(volume_units, "m^3")
        return self


class BuilderRadiusMixin:
    """Mixin class for Builder classes to set radius and radius_units.

    Methods:
        set_radius: Set the radius attribute and units.
    """

    def __init__(self):
        self.radius = None

    def set_radius(
        self,
        radius: Union[float, NDArray[np.float64]],
        radius_units: Optional[str] = "m",
    ):
        """Set the radius of the particle in meters.

        Args:
            radius: Radius of the particle.
            radius_units: Units of the radius. Default is *m*.

        Raises:
            ValueError: If radius is negative
        """
        if np.any(radius < 0):
            error_message = "Radius must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.radius = radius * convert_units(radius_units, "m")
        return self


class BuilderTemperatureMixin:
    """Mixin class for AtmosphereBuilder to set temperature.

    Methods:
        set_temperature: Set the temperature attribute and units.
    """

    def __init__(self):
        self.temperature = None

    def set_temperature(
        self, temperature: float, temperature_units: str = "K"
    ):
        """Set the temperature of the atmosphere.

        Args:
            temperature (float): Temperature of the gas mixture.
            temperature_units (str): Units of the temperature.
                Options include 'degC', 'degF', 'degR', 'K'. Default is 'K'.

        Returns:
            AtmosphereBuilderMixin: This object instance with updated
                temperature.

        Raises:
            ValueError: If the converted temperature is below absolute zero.
        """
        self.temperature = convert_units(
            temperature_units, "K", value=temperature
        )  # temperature is a non-multiplicative conversion
        if self.temperature < 0:
            logger.error("Temperature must be above zero Kelvin.")
            raise ValueError("Temperature must be above zero Kelvin.")
        return self


class BuilderPressureMixin:
    """Mixin class for AtmosphereBuilder to set total pressure.

    Methods:
        set_pressure: Set the total pressure attribute and units.
    """

    def __init__(self):
        self.pressure = None

    def set_pressure(
        self,
        pressure: Union[float, NDArray[np.float64]],
        pressure_units: str = "Pa",
    ):
        """Set the total pressure of the atmosphere.

        Args:
            total_pressure: Total pressure of the gas mixture.
            pressure_units: Units of the pressure. Options include
                'Pa', 'kPa', 'MPa', 'psi', 'bar', 'atm'. Default is 'Pa'.

        Returns:
            AtmosphereBuilderMixin: This object instance with updated pressure.

        Raises:
            ValueError: If the total pressure is below zero.
        """
        if np.any(pressure < 0):
            logger.error("Pressure must be a positive value.")
            raise ValueError("Pressure must be a positive value.")
        self.pressure = pressure * convert_units(pressure_units, "Pa")
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

    def set_mode(
        self,
        mode: NDArray[np.float64],
        mode_units: str = "m",
    ):
        """Set the mode for distribution.

        Args:
            mode: The modes for the radius.
            mode_units: The units for the modes, default is 'm'.

        Raises:
            ValueError: If mode is negative.
        """
        if np.any(mode < 0):
            message = "The mode must be positive."
            logger.error(message)
            raise ValueError(message)
        self.mode = mode * convert_units(mode_units, "m")
        return self

    def set_geometric_standard_deviation(
        self,
        geometric_standard_deviation: NDArray[np.float64],
        geometric_standard_deviation_units: Optional[str] = None,
    ):
        """Set the geometric standard deviation for the distribution.

        Args:
            geometric_standard_deviation: The geometric standard deviation for
                the radius.
            geometric_standard_deviation_units: Optional, ignored units for
                geometric standard deviation [dimensionless].

        Raises:
            ValueError: If geometric standard deviation is negative.
        """
        if np.any(geometric_standard_deviation < 0):
            message = "The geometric standard deviation must be positive."
            logger.error(message)
            raise ValueError(message)
        if geometric_standard_deviation_units is not None:
            logger.warning("Ignoring units for surface strategy parameter.")
        self.geometric_standard_deviation = geometric_standard_deviation
        return self

    def set_number_concentration(
        self,
        number_concentration: NDArray[np.float64],
        number_concentration_units: str = "1/m^3",
    ):
        """Set the number concentration for the distribution.

        Args:
            number_concentration: The number concentration for the radius.
            number_concentration_units: The units for the number concentration,
                default is '1/m^3'.

        Raises:
            ValueError: If number concentration is negative.
        """
        if np.any(number_concentration < 0):
            message = "The number concentration must be positive."
            logger.error(message)
            raise ValueError(message)
        self.number_concentration = number_concentration * convert_units(
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

    def set_particle_resolved_count(
        self,
        particle_resolved_count: int,
        particle_resolved_count_units: Optional[str] = None,
    ):
        """Set the number of particles to resolve.

        Args:
            particle_resolved_count: The number of particles to resolve.
            particle_resolved_count_units: Ignored units for particle resolved.

        Raises:
            ValueError: If particle_resolved_count is negative.
        """
        if particle_resolved_count < 0:
            message = "The number of particles must be positive."
            logger.error(message)
            raise ValueError(message)
        if particle_resolved_count_units is not None:
            logger.warning("Ignoring units for particle resolved count.")
        self.particle_resolved_count = particle_resolved_count
        return self
