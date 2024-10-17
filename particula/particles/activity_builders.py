"""Builder class for Activity objects with validation and error handling.

Change to MixinMolar classes, after PR integration.
"""

import logging
from typing import Optional, Union
from numpy.typing import NDArray
import numpy as np
from particula.abc_builder import (
    BuilderABC,
)
from particula.builder_mixin import (
    BuilderMolarMassMixin,
    BuilderDensityMixin,
)
from particula.particles.activity_strategies import (
    ActivityIdealMass,
    ActivityIdealMolar,
    ActivityKappaParameter,
    ActivityStrategy,
)

logger = logging.getLogger("particula")


class ActivityIdealMassBuilder(BuilderABC):
    """Builder class for IdealActivityMass objects. No additional parameters.

    Methods:
        build(): Validate and return the IdealActivityMass object.
    """

    def __init__(self):
        required_parameters = None
        BuilderABC.__init__(self, required_parameters)

    def build(self) -> ActivityStrategy:
        """Validate and return the IdealActivityMass object.

        Returns:
            IdealActivityMass: The validated IdealActivityMass object.
        """
        return ActivityIdealMass()


class ActivityIdealMolarBuilder(BuilderABC, BuilderMolarMassMixin):
    """Builder class for IdealActivityMolar objects.

    Methods:
        set_molar_mass(molar_mass, molar_mass_units): Set the molar mass of the
            particle in kg/mol. Default units are 'kg/mol'.
        set_parameters(params): Set the parameters of the IdealActivityMolar
            object from a dictionary including optional units.
        build(): Validate and return the IdealActivityMolar object.
    """

    def __init__(self):
        required_parameters = ["molar_mass"]
        BuilderABC.__init__(self, required_parameters)
        BuilderMolarMassMixin.__init__(self)

    def build(self) -> ActivityStrategy:
        """Validate and return the IdealActivityMolar object.

        Returns:
            IdealActivityMolar: The validated IdealActivityMolar object.
        """
        self.pre_build_check()
        return ActivityIdealMolar(molar_mass=self.molar_mass)  # type: ignore


class ActivityKappaParameterBuilder(
    BuilderABC, BuilderDensityMixin, BuilderMolarMassMixin
):
    """Builder class for KappaParameterActivity objects.

    Methods:
        set_kappa(kappa): Set the kappa parameter for the activity calculation.
        set_density(density,density_units): Set the density of the species in
            kg/m^3. Default units are 'kg/m^3'.
        set_molar_mass(molar_mass,molar_mass_units): Set the molar mass of the
            species in kg/mol. Default units are 'kg/mol'.
        set_water_index(water_index): Set the array index of the species.
        set_parameters(dict): Set the parameters of the KappaParameterActivity
            object from a dictionary including optional units.
        build(): Validate and return the KappaParameterActivity object.
    """

    def __init__(self):
        required_parameters = ["kappa", "density", "molar_mass", "water_index"]
        BuilderABC.__init__(self, required_parameters)
        BuilderDensityMixin.__init__(self)
        BuilderMolarMassMixin.__init__(self)
        self.kappa = None
        self.water_index = None

    def set_kappa(
        self,
        kappa: Union[float, NDArray[np.float64]],
        kappa_units: Optional[str] = None,
    ):
        """Set the kappa parameter for the activity calculation.

        Args:
            kappa: The kappa parameter for the activity calculation.
            kappa_units: Not used. (for interface consistency)
        """
        if np.any(kappa < 0):
            error_message = "Kappa parameter must be a positive value."
            logger.error(error_message)
            raise ValueError(error_message)
        if kappa_units is not None:
            logger.warning("Ignoring units for kappa parameter.")
        self.kappa = kappa
        return self

    def set_water_index(
        self, water_index: int, water_index_units: Optional[str] = None
    ):
        """Set the array index of the species.

        Args:
            water_index: The array index of the species.
            water_index_units: Not used. (for interface consistency)
        """
        if not isinstance(water_index, int):  # type: ignore
            error_message = "Water index must be an integer."
            logger.error(error_message)
            raise TypeError(error_message)
        if water_index_units is not None:
            logger.warning("Ignoring units for water index.")
        self.water_index = water_index
        return self

    def build(self) -> ActivityStrategy:
        """Validate and return the KappaParameterActivity object.

        Returns:
            KappaParameterActivity: The validated KappaParameterActivity
                object.
        """
        self.pre_build_check()
        return ActivityKappaParameter(
            kappa=self.kappa,  # type: ignore
            density=self.density,  # type: ignore
            molar_mass=self.molar_mass,  # type: ignore
            water_index=self.water_index,  # type: ignore
        )
