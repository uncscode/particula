"""Builder class for Surface strategies, for how to account for
surface tension in the calculation of the Kelvin effect.

This builds the strategy and checks that the required parameters are set,
and converts the units of the parameters if necessary.

We could add another layer for common methods between the three strategies,
but lets wait, as now it is very clear for the user to see what is required
for each strategy. And there is no coupling between the strategies.
"""

import logging
from typing import Optional, Union
from numpy.typing import NDArray
import numpy as np
from particula.next.abc_builder import BuilderABC
from particula.next.particles.surface_strategies import (
    SurfaceStrategyMass, SurfaceStrategyMolar, SurfaceStrategyVolume
)
from particula.util.input_handling import convert_units  # type: ignore

logger = logging.getLogger("particula")


class SurfaceStrategyMolarBuilder(BuilderABC):
    """Builder class for SurfaceStrategyMolar objects.

    Methods:
    --------
    - set_surface_tension(surface_tension, surface_tension_units): Set the
        surface tension of the particle in N/m. Default units are 'N/m'.
    - set_density(density, density_units): Set the density of the particle in
        kg/m^3. Default units are 'kg/m^3'.
    - set_molar_mass(molar_mass, molar_mass_units): Set the molar mass of the
        particle in kg/mol. Default units are 'kg/mol'.
    - set_parameters(params): Set the parameters of the SurfaceStrategyMolar
        object from a dictionary including optional units.
    - build(): Validate and return the SurfaceStrategyMolar object.
    """

    def __init__(self):
        required_parameters = ['surface_tension', 'density', 'molar_mass']
        super().__init__(required_parameters)
        self.surface_tension = None
        self.density = None
        self.molar_mass = None

    def set_surface_tension(
        self,
        surface_tension: Union[float, NDArray[np.float_]],
        surface_tension_units: Optional[str] = 'N/m'
    ):
        """Set the surface tension of the particle in N/m.

        Args:
        -----
        - surface_tension (float or NDArray[float]): Surface tension of the
            particle [N/m].
        - surface_tension_units (str, optional): Units of the surface tension.
            Default is 'N/m'.
        """
        if np.any(surface_tension < 0):
            error_message = "Surface tension must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.surface_tension = surface_tension \
            * convert_units(surface_tension_units, 'N/m')

    def set_density(
        self,
        density: Union[float, NDArray[np.float_]],
        density_units: Optional[str] = 'kg/m^3'
    ):
        """Set the density of the particle in kg/m^3.

        Args:
        -----
        - density (float or NDArray[float]): Density of the particle [kg/m^3].
        - density_units (str, optional): Units of the density. Default is
            'kg/m^3'.
        """
        if np.any(density < 0):
            error_message = "Density must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.density = density * convert_units(density_units, 'kg/m^3')

    def set_molar_mass(
        self,
        molar_mass: Union[float, NDArray[np.float_]],
        molar_mass_units: Optional[str] = 'kg/mol'
    ):
        """Set the molar mass of the particle in kg/mol.

        Args:
        -----
        - molar_mass (float or NDArray[float]): Molar mass of the particle
            [kg/mol].
        - molar_mass_units (str, optional): Units of the molar mass. Default is
            'kg/mol'.
        """
        if np.any(molar_mass < 0):
            error_message = "Molar mass must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.molar_mass = molar_mass \
            * convert_units(molar_mass_units, 'kg/mol')

    def build(self) -> SurfaceStrategyMolar:
        """Validate and return the SurfaceStrategyMass object.

        Returns:
        --------
        - SurfaceStrategyMolar: Instance of the SurfaceStrategyMolar object.
        """
        self.pre_build_check()
        return SurfaceStrategyMolar(
            surface_tension=self.surface_tension,  # type: ignore
            density=self.density,  # type: ignore
            molar_mass=self.molar_mass  # type: ignore
        )


class SurfaceStrategyMassBuilder(BuilderABC):
    """Builder class for SurfaceStrategyMass objects.

    Methods:
    --------
    - set_surface_tension(surface_tension, surface_tension_units): Set the
        surface tension of the particle in N/m. Default units are 'N/m'.
    - set_density(density, density_units): Set the density of the particle in
        kg/m^3. Default units are 'kg/m^3'.
    - set_parameters(params): Set the parameters of the SurfaceStrategyMass
        object from a dictionary including optional units.
    - build(): Validate and return the SurfaceStrategyMass object.
    """

    def __init__(self):
        required_parameters = ['surface_tension', 'density']
        super().__init__(required_parameters)
        self.surface_tension = None
        self.density = None

    def set_surface_tension(
        self,
        surface_tension: Union[float, NDArray[np.float_]],
        surface_tension_units: Optional[str] = 'N/m'
    ):
        """Set the surface tension of the particle in N/m.

        Args:
        -----
        - surface_tension (float or NDArray[float]): Surface tension of the
            particle [N/m].
        - surface_tension_units (str, optional): Units of the surface tension.
            Default is 'N/m'.
        """
        if np.any(surface_tension < 0):
            error_message = "Surface tension must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.surface_tension = surface_tension \
            * convert_units(surface_tension_units, 'N/m')

    def set_density(
        self,
        density: Union[float, NDArray[np.float_]],
        density_units: Optional[str] = 'kg/m^3'
    ):
        """Set the density of the particle in kg/m^3.

        Args:
        -----
        - density (float or NDArray[float]): Density of the particle [kg/m^3].
        - density_units (str, optional): Units of the density. Default is
            'kg/m^3'.
        """
        if np.any(density < 0):
            error_message = "Density must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.density = density * convert_units(density_units, 'kg/m^3')

    def build(self) -> SurfaceStrategyMass:
        """Validate and return the SurfaceStrategyMass object.

        Returns:
        --------
        - SurfaceStrategyMass: Instance of the SurfaceStrategyMass object.
        """
        self.pre_build_check()
        return SurfaceStrategyMass(
            surface_tension=self.surface_tension,  # type: ignore
            density=self.density,  # type: ignore
        )


class SurfaceStrategyVolumeBuilder(BuilderABC):
    """Builder class for SurfaceStrategyVolume objects.

    Methods:
    --------
    - set_surface_tension(surface_tension, surface_tension_units): Set the
        surface tension of the particle in N/m. Default units are 'N/m'.
    - set_density(density, density_units): Set the density of the particle in
        kg/m^3. Default units are 'kg/m^3'.
    - set_parameters(params): Set the parameters of the SurfaceStrategyVolume
        object from a dictionary including optional units.
    - build(): Validate and return the SurfaceStrategyVolume object.
    """

    def __init__(self):
        required_parameters = ['surface_tension', 'density']
        super().__init__(required_parameters)
        self.surface_tension = None
        self.density = None

    def set_surface_tension(
        self,
        surface_tension: Union[float, NDArray[np.float_]],
        surface_tension_units: Optional[str] = 'N/m'
    ):
        """Set the surface tension of the particle in N/m.

        Args:
        -----
        - surface_tension (float or NDArray[float]): Surface tension of the
            particle [N/m].
        - surface_tension_units (str, optional): Units of the surface tension.
            Default is 'N/m'.
        """
        if np.any(surface_tension < 0):
            error_message = "Surface tension must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.surface_tension = surface_tension \
            * convert_units(surface_tension_units, 'N/m')

    def set_density(
        self,
        density: Union[float, NDArray[np.float_]],
        density_units: Optional[str] = 'kg/m^3'
    ):
        """Set the density of the particle in kg/m^3.

        Args:
        -----
        - density (float or NDArray[float]): Density of the particle [kg/m^3].
        - density_units (str, optional): Units of the density. Default is
            'kg/m^3'.
        """
        if np.any(density < 0):
            error_message = "Density must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.density = density * convert_units(density_units, 'kg/m^3')

    def build(self) -> SurfaceStrategyVolume:
        """Validate and return the SurfaceStrategyVolume object.

        Returns:
        --------
        - SurfaceStrategyVolume: Instance of the SurfaceStrategyVolume object.
        """
        self.pre_build_check()
        return SurfaceStrategyVolume(
            surface_tension=self.surface_tension,  # type: ignore
            density=self.density  # type: ignore
        )
