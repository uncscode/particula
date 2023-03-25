""" calculate the species properties, saturation vapor pressures,
    latent heats, etc.

    TODO: add a more general class, with a dictionary that can be added from
    a file or inputs.
"""

import numpy as np
from particula import u
from particula.environment import Environment
from input_handling import in_latent_heat


class water(Environment):
    """ based on the Environment class
    """

    def __init__(self, **kwargs):
        """ initiating the vapor class
        """
        super().__init__(**kwargs)

        self.kwargs = kwargs

    def saturation_pressure0(self,):
        """Calculate the saturation vapour pressure for a given temperature.

        Parameters
        ----------
        temperature : K
            Temperature of the air.
        """

        # # Latent heat
        # L = self.latent_heat()
        # # Saturation vapour pressure # not clear where this comes from.
        # es = es0*np.exp((L/Rv)*(1/283 - 1/self.temperature))

        saturation_pressure = self.buck_wvpsat(self.temperature)

        return saturation_pressure

    def buck_wvpsat(self):
        """ Buck equation for water vapor pressure
            https://en.wikipedia.org/wiki/Arden_Buck_equation
        """

        temp = self.temperature.m_as("degC")

        return 6.1115 * np.exp(
            (23.036-temp/333.7)*(temp/(279.82+temp))
        )*u.hPa * (temp < 0.0) + 6.1121 * np.exp(
            (18.678-temp/234.5)*(temp/(257.14+temp))
        )*u.hPa * (temp >= 0.0)

    def latent_heat(self):
        """Calculate the latent heat of condensation for a given temperature.
        Polynomial for the latent heat of condensation from Rogers and Yau
        Table 2.1. https://en.wikipedia.org/wiki/Latent_heat

        range: -25 °C to 40 °C

        Parameters
        ----------
        temperature : K
            Temperature of the air.

        Returns
        -------
        latent_heat : J/kg"""

        # Convert Kelvin to Celsius
        temperatureC = self.temperature.to('u.C').m

        # Latent heat of condensation
        latent_heat = (
                2500.8 - 2.36*temperatureC
                + 0.0016*temperatureC**2
                - 0.00006*temperatureC**3
            ) * u.J/u.g

        return in_latent_heat(latent_heat)
