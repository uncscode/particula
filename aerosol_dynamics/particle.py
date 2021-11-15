import numpy as np
from aerosol_dynamics import physical_parameters as pp

class Particle:
    def __init__(self, name: str, radius: float, density: float, charge: float):
        self.__name = name
        self.__radius = radius
        # self.__density = density
        self.__charge = charge
        self.__mass = density * (4*np.pi/3) * (radius**3)

    def name(self):
        return self.__name

    def mass(self):
        return self.__mass
 
    def radius(self):
        return self.__radius

    def charge(self):
        return self.__charge

    def knudsen_number(self) -> float:
        return pp.MEAN_FREE_PATH_AIR / self.radius()

    def slip_correction_factor(self) -> float:
        knudsen_number: float = self.knudsen_number()
        return 1 + knudsen_number * (1.257 + 0.4*np.exp(-1.1/knudsen_number))

    def friction_factor(self) -> float:
        slip_correction_factor: float = self.slip_correction_factor()
        return (6 * np.pi * pp.MEDIUM_VISCOSITY * self.radius() / slip_correction_factor)

    def reduced_mass(self, other) -> float:
        return self.mass() * other.mass() / (self.mass() + other.mass())

    def reduced_friction_factor(self, other) -> float:
        return (self.friction_factor() * other.friction_factor()
            / (self.friction_factor() + other.friction_factor()))

    def coulomb_potential_ratio(self, other) -> float:
        numerator = -1 * self.charge() * other.charge() * (pp.ELEMENTARY_CHARGE_VALUE ** 2)
        denominator = 4 * np.pi * pp.ELECTRIC_PERMITTIVITY * (self.radius() + other.radius())
        return numerator / (denominator * pp.BOLTZMANN_CONSTANT * pp.TEMPERATURE)

    def coulomb_enhancement_kinetic_limit(self, other) -> float:
        coulomb_potential_ratio = self.coulomb_potential_ratio(other)
        return 1 + coulomb_potential_ratio if coulomb_potential_ratio >=0 else np.exp(coulomb_potential_ratio)

    def coulomb_enhancement_continuum_limit(self, other) -> float:
        coulomb_potential_ratio = self.coulomb_potential_ratio(other)
        return coulomb_potential_ratio / (1 - np.exp(-1*coulomb_potential_ratio)) if coulomb_potential_ratio != 0 else 1

    def diffusive_knudsen_number(self, other) -> float:
        reduced_mass = self.reduced_mass(other)
        coulomb_enhancement_continuum_limit = self.coulomb_enhancement_continuum_limit(other)
        reduced_friction_factor = self.reduced_friction_factor(other)
        coulomb_enhancement_kinetic_limit = self.coulomb_enhancement_kinetic_limit(other)
        return ((pp.TEMPERATURE * pp.BOLTZMANN_CONSTANT * reduced_mass**0.5) * coulomb_enhancement_continuum_limit/ 
            (reduced_friction_factor * (self.radius() + other.radius()) * coulomb_enhancement_kinetic_limit))

    def dimensionless_coagulation_kernel_hard_sphere(self, other) -> float:
        # Constants for the chargeless hard-sphere limit
        HSC1 = 25.836
        HSC2 = 11.211
        HSC3 = 3.502
        HSC4 = 7.211

        diffusive_knudsen_number = self.diffusive_knudsen_number(other)

        numerator = ((4 * np.pi * diffusive_knudsen_number**2) + (HSC1 * diffusive_knudsen_number**3) + 
                    ((8 * np.pi)**(1/2) * HSC2 * diffusive_knudsen_number**4))
        denominator = (1 + HSC3 * diffusive_knudsen_number + HSC4 * diffusive_knudsen_number**2 + HSC2 * diffusive_knudsen_number**3)

        return numerator / denominator

    def collision_kernel_continuum_limit(self, other) -> float:
        diffusive_knudsen_number = self.diffusive_knudsen_number(other)
        return 4 * np.pi * (diffusive_knudsen_number**2)

    def collision_kernel_kinetic_limit(self, other) -> float:
        diffusive_knudsen_number = self.diffusive_knudsen_number(other)
        return np.sqrt(8 * np.pi) * diffusive_knudsen_number

    # # Gopalkrishnan and Hogan, 2012
    # # doi: https://doi.org/10.1103/PhysRevE.85.026410
    # # Equation 18
    # def dimensionless_coagulation_kernel_GopalkrishnanHogan2012(self, other):
    #     diffusive_knudsen_number = self.diffusive_knudsen_number(other)
    #     continuum_limit = self.collision_kernel_continuum_limit(other)
    #     coulomb_potential_ratio = self.coulomb_potential_ratio(other)
    #     # kinetic_limit = self.collision_kernel_kinetic_limit(other)

    #     return continuum_limit / (
    #         1 + 1.598 * np.minimum(diffusive_knudsen_number,
    #         3*diffusive_knudsen_number / (2*coulomb_potential_ratio)
    #         )**1.1709
    #     )

    # # Gatti and Kortshagen 2008
    # # doi: https://doi.org/10.1103/PhysRevE.78.046402
    # # Retrieved from Gopalkrishnan and Hogan, 2012,
    # # 10.1103/PhysRevE.85.026410,
    # # Equation 13
    # def dimensionless_coagulation_kernel_GattiKortshagen2008(self, other):
    #     kernel_hard_sphere = self.dimensionless_coagulation_kernel_hard_sphere(other)