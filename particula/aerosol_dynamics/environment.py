""" defining the environment class.

    The environment is defined by:
        - the temperature of the environment in K
        - the pressure inside the environment in Pa

    The enviroment class is used to derivate properties:
        - the mean free path in air
        - the dynamic viscosity in air
"""

from particula import u
from particula.utils.environment_ import(
    mean_free_path_air as mfp_air,
    dynamic_viscosity_air as dyn_vis_air,
)


class Environment:
    """ sets the environment class.
    """
    def __init__(self, temperature, pressure):
        """ initiates the environment class.

            Inputs:
                - temperature
                - pressure
        """
        self._temperature = temperature
        self._pressure = pressure

    @u.wraps(u.K, [None])
    def temperature(self) -> float:
        """ Returns the temperature: [K]
        """

        return self._temperature

    @u.wraps(u.Pa, [None])
    def pressure(self) -> float:
        """ Returns the pressure: [Pa]
        """

        return self._pressure

    @u.wraps(u.kg / u.m / u.s, [None])
    def dynamic_viscosity_air(self) -> float:
        """ Returns the dynamic viscosity of air: [kg/m/s]

            uses: utils.environment_.dynamic_viscosity_air
        """

        return dyn_vis_air(self.temperature())

    @u.wraps(u.m, [None])
    def mean_free_path_air(self) -> float:
        """ Returns the mean free path in air: [m]

            uses: utils.environment_.mean_free_path_air
        """

        return mfp_air(self.temperature(), self.pressure())
