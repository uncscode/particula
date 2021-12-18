"""
Environment Class
"""

import numpy as np
from aerosol_dynamics import physical_parameters as pp
from . import u

class Environment:
    """
    Sets the environment class for creating air parcels, with properties such as
    temperature and pressure, and derived properties such as air viscosity.
    """
    def __init__(self, temperature, pressure):
        # assert temperature == 298 * u.K , "Currently 300 K is the only temperature supported"
        # assert pressure == 1 * u.atm , "Currently 1 atm is the only pressure supported"
        self._temperature = temperature
        self._pressure = pressure

    @u.wraps(u.K, [None])
    def temperature(self) -> float:
        """Returns the temperature of the environment."""
        return self._temperature

    @u.wraps(u.Pa, [None])
    def pressure(self) -> float:
        """Returns the pressure of the environment."""
        return self._pressure

    @u.wraps(u.kg / u.m / u.s, [None])
    def dynamic_viscosity_air(self) -> float:
        """Returns the dynamic viscosity of air. [kg/m/s]

        The dynamic viscosity is calculated using the 3 parameter
        Sutherland Viscosity Law.
        """
        mu_ref = 1.716e-5 * u.Pa * u.s # Viscosity at T_REF
        t_ref = 273.15 * u.K
        suth_const = 110.4 * u.K # Sutherland constant

        return (
            mu_ref *
            (self.temperature()/t_ref)**(3/2) *
            (t_ref + suth_const) / (self.temperature() + suth_const)
        )

    # mean free path of air in m
    @u.wraps(u.m, [None])
    def mean_free_path_air(self) -> float:
        """Returns the mean free path of this environment. [m]

        The mean free path is the average distance traveled by a molecule
        between collisions with other molecules.
        """
        molecular_weight = (28.9644 * u.g / u.mol).to_base_units() # Molecular weight of air

        return (
            (
                2 * self.dynamic_viscosity_air() /
                self.pressure() /
                (8*molecular_weight / (np.pi*pp.GAS_CONSTANT*self.temperature()))**0.5
            ).to_base_units()
        )
