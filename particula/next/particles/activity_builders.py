"""Builder class for Activity objects with validation and error handling.
"""

import logging
from typing import Optional, Union
from numpy.typing import NDArray
import numpy as np
from particula.next.abc_builder import BuilderABC
from particula.next.particles.activity_strategies import (
    IdealActivityMass, IdealActivityMolar, KappaParameterActivity
)
from particula.util.input_handling import convert_units  # type: ignore

logger = logging.getLogger("particula")


class IdealActivityMassBuilder(BuilderABC):
    """Builder class for IdealActivityMass objects. No parameters are required
    to be set.

    Methods:
    --------
    - build(): Validate and return the IdealActivityMass object.
    """

    def __init__(self):
        required_parameters = None
        super().__init__(required_parameters)

    def build(self) -> IdealActivityMass:
        """Validate and return the IdealActivityMass object.

        Returns:
        -------
        - IdealActivityMass: The validated IdealActivityMass object.
        """
        return IdealActivityMass()


class IdealActivityMolarBuilder(BuilderABC):
    """Builder class for IdealActivityMolar objects. 

    Methods:
    --------
    - set_molar_mass(molar_mass, molar_mass_units): Set the molar mass of the
        particle in kg/mol. Default units are 'kg/mol'.
    - set_parameters(params): Set the parameters of the IdealActivityMolar
        object from a dictionary including optional units.
    - build(): Validate and return the IdealActivityMolar object.
    """

    def __init__(self):
        required_parameters = ['molar_mass']
        super().__init__(required_parameters)
        self.molar_mass = None

    def set_molar_mass(
        self,
        molar_mass: Union[float, NDArray[np.float_]],
        molar_mass_units: Optional[str] = 'kg/mol'
    ):
        """Set the molar mass of the particle in kg/mol.

        Args:
        ----
        - molar_mass (float): The molar mass of the chemical species.
        - molar_mass_units (str): The units of the molar mass input.
        Default is 'kg/mol'.
        """
        if molar_mass < 0:
            error_message = "Molar mass must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.molar_mass = molar_mass \
            * convert_units(molar_mass_units, 'kg/mol')
        return self

    def build(self) -> IdealActivityMolar:
        """Validate and return the IdealActivityMolar object.

        Returns:
        -------
        - IdealActivityMolar: The validated IdealActivityMolar object.
        """
        self.pre_build_check()
        return IdealActivityMolar(molar_mass=self.molar_mass)  # type: ignore


class KappaParameterActivityBuilder(BuilderABC):
    """Builder class for KappaParameterActivity objects.

    Methods:
    --------
    - set_kappa(kappa): Set the kappa parameter for the activity calculation.
    - set_density(density, density_units): Set the density of the species in
        kg/m^3. Default units are 'kg/m^3'.
    - set_molar_mass(molar_mass, molar_mass_units): Set the molar mass of the
        species in kg/mol. Default units are 'kg/mol'.
    - set_water_index(water_index): Set the array index of the species.
    - set_parameters(dict): Set the parameters of the KappaParameterActivity
        object from a dictionary including optional units.
    - build(): Validate and return the KappaParameterActivity object.
    """

    def __init__(self):
        required_parameters = [
            'kappa', 'density', 'molar_mass', 'water_index']
        super().__init__(required_parameters)
        self.kappa = None
        self.density = None
        self.molar_mass = None
        self.water_index = None

    def set_kappa(self, kappa: Union[float, NDArray[np.float_]]):
        """Set the kappa parameter for the activity calculation.

        Args:
        ----
        - kappa: The kappa parameter for the activity calculation.
        """
        if kappa < 0:
            error_message = "Kappa parameter must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.kappa = kappa
        return self

    def set_density(
        self,
        density: Union[float, NDArray[np.float_]],
        density_units: str = 'kg/m^3'
    ):
        """Set the density of the species in kg/m^3.

        Args:
        ----
        - density (float): The density of the species.
        - density_units (str): The units of the density input. Default is
            'kg/m^3'.
        """
        if density < 0:
            error_message = "Density must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.density = density * convert_units(density_units, 'kg/m^3')
        return self

    def set_molar_mass(
        self,
        molar_mass: Union[float, NDArray[np.float_]],
        molar_mass_units: str = 'kg/mol'
    ):
        """Set the molar mass of the species in kg/mol.

        Args:
        ----
        - molar_mass (float): The molar mass of the species.
        - molar_mass_units (str): The units of the molar mass input. Default is
            'kg/mol'.
        """
        if molar_mass < 0:
            error_message = "Molar mass must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        self.molar_mass = molar_mass \
            * convert_units(molar_mass_units, 'kg/mol')
        return self

    def set_water_index(self, water_index: int):
        """Set the array index of the species.

        Args:
        ----
        - water_index (int): The array index of the species."""
        self.water_index = water_index
        return self

    def build(self) -> KappaParameterActivity:
        """Validate and return the KappaParameterActivity object.

        Returns:
        -------
        - KappaParameterActivity: The validated KappaParameterActivity object.
        """
        self.pre_build_check()
        return KappaParameterActivity(
            kappa=self.kappa,
            density=self.density,
            molar_mass=self.molar_mass,
            water_index=self.water_index
        )
