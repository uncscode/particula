""" defining the environment class

    kwargs:
        temperature             (float) [K]      (default: 298.15)
        pressure                (float) [Pa]     (default: 101325)
      # dynamic_viscosity       (float) [Pa*s]   (default: util)
      $ molecular_weight        (float) [kg/mol] (default: constants)
      $ reference_viscosity     (float) [Pa*s]   (default: constants)
      $ reference_temperature   (float) [K]      (default: constants)
      $ sutherland_constant     (float) [K]      (default: constants)
      $ gas_constant            (float) [J/mol/K](default: constants)

      Using particula.util:
        dynamic_viscosity       (float) [Pa*s]

      Using particula.constants:
        GAS_CONSTANT            (float) [J/mol/K]
        MOLECULAR_WEIGHT_AIR    (float) [kg/mol]
        REF_VISCOSITY_AIR_STP   (float) [Pa*s]
        REF_TEMPERATURE_STP     (float) [K]
        SUTHERLAND_CONSTANT     (float) [K]

    Notes:
      # stands for an optional util override
      $ stands for an optional constants override

"""

from particula.constants import (GAS_CONSTANT, MOLECULAR_WEIGHT_AIR,
                                 REF_TEMPERATURE_STP, REF_VISCOSITY_AIR_STP,
                                 SUTHERLAND_CONSTANT)
from particula.util.dynamic_viscosity import dyn_vis
from particula.util.input_handling import (in_gas_constant,
                                           in_molecular_weight, in_pressure,
                                           in_temperature, in_viscosity)
from particula.util.mean_free_path import mfp


class SharedProperties:  # pylint: disable=too-few-public-methods
    """ a hidden class for sharing properties like
        coagulation_approximation
    """

    def __init__(self, **kwargs):
        """ initiate
        """

        self.coagulation_approximation = str(
            kwargs.get("coagulation_approximation", "hardsphere")
        )


class Environment(SharedProperties):  # pylint: disable=too-many-instance-attributes
    """ creating the environment class

        For now, the environment class takes properties such as
        temperature and pressure to calculate derived properties
        such as viscosity and mean free path.
    """

    def __init__(self, **kwargs):
        """ Initiate the environment class with base attrs.
        """

        super().__init__(**kwargs)

        self.temperature = in_temperature(
            kwargs.get("temperature", 298.15)
        )
        self.reference_viscosity = in_viscosity(
            kwargs.get("reference_viscosity", REF_VISCOSITY_AIR_STP)
        )
        self.reference_temperature = in_temperature(
            kwargs.get("reference_temperature", REF_TEMPERATURE_STP)
        )
        self.pressure = in_pressure(
            kwargs.get("pressure", 101325)
        )
        self.molecular_weight = in_molecular_weight(
            kwargs.get("molecular_weight", MOLECULAR_WEIGHT_AIR)
        )
        self.sutherland_constant = in_temperature(
            kwargs.get("sutherland_constant", SUTHERLAND_CONSTANT)
        )
        self.gas_constant = in_gas_constant(
            kwargs.get("gas_constant", GAS_CONSTANT)
        )

        self.kwargs = kwargs

    def dynamic_viscosity(self):
        """ Returns the dynamic viscosity in Pa*s.
        """
        return dyn_vis(
            temperature=self.temperature,
            reference_viscosity=self.reference_viscosity,
            reference_temperature=self.reference_temperature,
            sutherland_constant=self.sutherland_constant,
        )

    def mean_free_path(self):
        """ Returns the mean free path in m.
        """
        return mfp(
            temperature=self.temperature,
            pressure=self.pressure,
            molecular_weight=self.molecular_weight,
            dynamic_viscosity=self.dynamic_viscosity(),
            gas_constant=self.gas_constant,
        )
