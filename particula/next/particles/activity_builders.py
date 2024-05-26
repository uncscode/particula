"""Builder class for Activity objects with validation and error handling.
"""

import logging
from typing import Optional
from numpy.typing import NDArray
import numpy as np
from particula.next.abc_builder import BuilderABC
from particula.next.particles.activity_strategies import (
    ActivityStrategy, IdealActivityMass, IdealActivityMolar,
    KappaParameterActivity
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
        molar_mass: float,
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
