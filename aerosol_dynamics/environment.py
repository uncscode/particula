from aerosol_dynamics import physical_parameters as pp
from . import u
import math

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
        MU_REF = 1.716e-5 * u.Pa * u.s # Viscosity at T_REF
        T_REF = 273.15 * u.K
        S = 110.4 * u.K # Sutherland constant

        return (
            MU_REF *
            (self.temperature()/T_REF)**(3/2) *
            (T_REF+S) / (self.temperature()+S)
        )

    # mean free path of air in m
    # @u.wraps(u.m, [None])
    def mean_free_path_air(self) -> float:
        """Returns the mean free path of this environment. [m]
        
        The mean free path is the average distance traveled by a molecule
        between collisions with other molecules.
        """
        MOLECULAR_WEIGHT = (28.9644 * u.g / u.mol).to_base_units() # Molecular weight of air
        
        return (
            (
                2 * self.dynamic_viscosity_air() /
                self.pressure() /
                (8*MOLECULAR_WEIGHT / (math.pi*pp.GAS_CONSTANT*self.temperature()))**0.5
            ).to_base_units()
        )
