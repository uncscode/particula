"""
Class for creating particles.
"""

import numpy as np

from aerosol_dynamics import physical_parameters as pp
from aerosol_dynamics.environment import Environment
from . import u


class Particle:
    """
    Class for creating particles:
    Framework for particle--particle and gas--particle interactions.

    Attributes:
        name    (str)   [no units]
        radius  (float) [m]
        density (float) [kg/m**3]
        charge  (int)   [dimensionless]
        mass    (float) [kg]
    """

    def __init__(self, name: str, radius, density, charge):
        """
        Constructs the particle object.

        Parameters:
            name    (str)   [no units]
            radius  (float) [m]
            density (float) [kg/m**3]
            charge  (int)   [dimensionless]
        """
        self._name = name
        self._radius = radius
        self._density = density
        self._charge = charge
        self._mass = density * (4*np.pi/3) * (radius**3)

    def name(self) -> str:
        """Returns the name of particle."""
        return self._name

    @u.wraps(u.kg, [None])
    def mass(self) -> float:
        """Returns mass of particle.
        Checks units: [kg]"""
        return self._mass

    @u.wraps(u.m, [None])
    def radius(self) -> float:
        """Returns radius of particle.
        Checks units: [m]"""
        return self._radius

    @u.wraps(u.kg / u.m**3, [None])
    def density(self) -> int:
        """Returns density of particle.
        Checks units: [kg/m**3]"""
        return self._density

    @u.wraps(u.dimensionless, [None])
    def charge(self) -> int:
        """Returns charge of particle.
        Checks units: [dimensionless]"""
        return self._charge

    @u.wraps(u.dimensionless, [None, None])
    def knudsen_number(self, environment: Environment) -> float:
        """Returns particle's Knudsen number.
        Checks units: [dimensionless]

        The Knudsen number reflects the relative length scales of
        the particle and the suspending fluid (air, water, etc.).
        This is calculated by the mean free path of the medium
        divided by the particle radius."""
        return environment.mean_free_path_air() / self.radius()

    @u.wraps(u.dimensionless, [None, None])
    def slip_correction_factor(self, environment: Environment) -> float:
        """Returns particle's Cunningham slip correction factor.
        Checks units: [dimensionless]

        Dimensionless quantity accounting for non-continuum effects
        on small particles. It is a deviation from Stokes' Law.
        Stokes assumes a no-slip condition that is not correct at
        high Knudsen numbers. The slip correction factor is used to
        calculate the friction factor.

        See Eq 9.34 in Atmos. Chem. & Phys. (2016) for more informatiom."""
        knudsen_number: float = self.knudsen_number(environment)
        return 1 + knudsen_number * (
            1.257 + 0.4*np.exp(-1.1/knudsen_number)
        )

    @u.wraps(u.kg / u.s, [None, None])
    def friction_factor(self, environment: Environment) -> float:
        """Returns a particle's friction factor.
        Checks units: [N*s/m]

        Property of the particle's size and surrounding medium.
        Multiplying the friction factor by the fluid velocity
        yields the drag force on the particle."""
        slip_correction_factor: float = self.slip_correction_factor(environment)
        return (
            6 * np.pi * environment.dynamic_viscosity_air() * self.radius() /
            slip_correction_factor
        )


    @u.wraps(u.kg, [None, None])
    def reduced_mass(self, other) -> float:
        """Returns the reduced mass of two particles.
        Checks units: [kg]

        The reduced mass is an "effective inertial" mass.
        Allows a two-body problem to be solved as a one-body problem."""
        return self.mass() * other.mass() / (self.mass() + other.mass())

    @u.wraps(u.kg / u.s, [None, None, None])
    def reduced_friction_factor(self, other, environment: Environment) -> float:
        """Returns the reduced friction factor between two particles.
        Checks units: [N*s/m]

        Similar to the reduced mass.
        The reduced friction factor allows a two-body problem
        to be solved as a one-body problem."""
        return (
            self.friction_factor(environment) * other.friction_factor(environment)
            / (self.friction_factor(environment) + other.friction_factor(environment))
        )

    @u.wraps(u.dimensionless, [None, None, None])
    def coulomb_potential_ratio(self, other, environment: Environment) -> float:
        """Calculates the Coulomb potential ratio.
        Checks units: [dimensionless]"""
        numerator = -1 * self.charge() * other.charge() * (
            pp.ELEMENTARY_CHARGE_VALUE ** 2
        )
        denominator = 4 * np.pi * pp.ELECTRIC_PERMITTIVITY * (
            self.radius() + other.radius()
        )
        return (
            numerator /
            (denominator * pp.BOLTZMANN_CONSTANT * environment.temperature())
        )

    @u.wraps(u.dimensionless, [None, None, None])
    def coulomb_enhancement_kinetic_limit(self, other, environment: Environment) -> float:
        """Kinetic limit of Coulomb enhancement for particle--particle cooagulation.
        Checks units: [dimensionless]"""
        coulomb_potential_ratio = self.coulomb_potential_ratio(other, environment)
        return (
            1 + coulomb_potential_ratio if coulomb_potential_ratio >= 0

            else np.exp(coulomb_potential_ratio)
        )

    @u.wraps(u.dimensionless, [None, None, None])
    def coulomb_enhancement_continuum_limit(self, other, environment: Environment) -> float:
        """Continuum limit of Coulomb enhancement for particle--particle coagulation.
        Checks units: [dimensionless]"""

        coulomb_potential_ratio = self.coulomb_potential_ratio(other, environment)
        return coulomb_potential_ratio / (
            1 - np.exp(-1*coulomb_potential_ratio)
        ) if coulomb_potential_ratio != 0 else 1

    @u.wraps(u.dimensionless, [None, None, None])
    def diffusive_knudsen_number(self, other, environment: Environment) -> float:
        """Diffusive Knudsen number.
        Checks units: [dimensionless]

        The *diffusive* Knudsen number is different from Knudsen number.
        Ratio of:
            - numerator: mean persistence of one particle
            - denominator: effective length scale of
                particle--particle Coulombic interaction"""

        numerator = (
            (
                environment.temperature() * pp.BOLTZMANN_CONSTANT
                * self.reduced_mass(other)
            )**0.5
            / self.reduced_friction_factor(other, environment)
        )
        denominator = (
            (self.radius() + other.radius())
            * self.coulomb_enhancement_kinetic_limit(other, environment)
            / self.coulomb_enhancement_continuum_limit(other, environment)
        )
        return numerator / denominator


    @u.wraps(u.dimensionless, [None, None, None])
    def dimensionless_coagulation_kernel_hard_sphere(
        self, other, environment: Environment
    ) -> float:
        """Dimensionless particle--particle coagulation kernel.
        Checks units: [dimensionless]"""

        # Constants for the chargeless hard-sphere limit
        # see doi:
        hsc1 = 25.836
        hsc2 = 11.211
        hsc3 = 3.502
        hsc4 = 7.211
        diffusive_knudsen_number = self.diffusive_knudsen_number(other, environment)

        numerator = (
            (4 * np.pi * diffusive_knudsen_number**2)
            + (hsc1 * diffusive_knudsen_number**3)
            + ((8 * np.pi)**(1/2) * hsc2 * diffusive_knudsen_number**4)
        )
        denominator = (
            1
            + hsc3 * diffusive_knudsen_number
            + hsc4 * diffusive_knudsen_number**2
            + hsc2 * diffusive_knudsen_number**3
        )
        return numerator / denominator

    @u.wraps(u.dimensionless, [None, None, None])
    def collision_kernel_continuum_limit(self, other, environment: Environment) -> float:
        """Continuum limit of collision kernel.
        Checks units: [dimensionless]"""
        diffusive_knudsen_number = self.diffusive_knudsen_number(other, environment)
        return 4 * np.pi * (diffusive_knudsen_number**2)

    @u.wraps(u.dimensionless, [None, None, None])
    def collision_kernel_kinetic_limit(self, other, environment: Environment) -> float:
        """Kinetic limit of collision kernel.
        Checks units: [dimensionless]"""
        diffusive_knudsen_number = self.diffusive_knudsen_number(other, environment)
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
    #     kernel_hard_sphere =
    # self.dimensionless_coagulation_kernel_hard_sphere(other)
