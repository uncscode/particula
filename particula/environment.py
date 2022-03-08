""" defining the environment class
"""

from particula.util.dynamic_viscosity import dyn_vis
from particula.util.input_handling import in_pressure, in_temperature
from particula.util.mean_free_path import mfp


class Environment:
    """ creating the environment class

        For now, the environment class takes properties such as
        temperature and pressure to calculate derived properties
        such as viscosity.
    """

    def __init__(self, **kwargs):
        """ Function calls for enviornment class.
        """
        self.kwargs = kwargs

    def temperature(self):
        """ Returns the temperature in the environment.
        """
        return in_temperature(self.kwargs.get("temperature", 298.15))

    def pressure(self):
        """ Returns the pressure in the environment.
        """
        return in_pressure(self.kwargs.get("pressure", 101325))

    def dynamic_viscosity(self):
        """ Returns the dynamic viscosity of air.
        """
        return dyn_vis(**self.kwargs)

    def mean_free_path(self):
        """ Returns the mean free path of this environment.
        """
        return mfp(**self.kwargs)
