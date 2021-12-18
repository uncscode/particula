"""
Class for creating particles.
"""

import numpy as np

from aerosol_dynamics import physical_parameters as pp

from . import u


class Particle:
    """
    Class for creating particles:
    Framework for particle--particle and gas--particle interactions.

    Attributes:
        name    (str)
        radius  (float)
        density (float)
        charge  (int)
        mass    (float)
    """

    def __init__(self, name: str, radius, density, charge):
        """
        Constructs the particle object.

        Parameters:
            name    (str)
            radius  (float)
            density (float)
            charge  (int)
        """
        self._name      = name
        self._radius    = radius
        self._density   = density
        self._charge    = charge
        self._mass      = density * (4*np.pi/3) * (radius**3)

    def name(self) -> str:
        """Returns the name of particle."""
        return self._name

    @u.wraps(u.kg, [None])
    def mass(self) -> float:
        """Returns mass of particle.
        Checks units. [kg]"""
        return self._mass

    @u.wraps(u.m, [None])
    def radius(self) -> float:
        """Returns radius of particle.
        Checks units. [m]"""
        return self._radius

    @u.wraps(u.dimensionless, [None])
    def charge(self) -> int:
        """Returns charge of particle.
        Checks units. [unitless]"""
        return self._charge

    @u.wraps(u.dimensionless, [None])
    def knudsen_number(self) -> float:
        """Returns particle's Knudsen number.
        Checks units. [unitless]

        The Knudsen number reflects the relative length scales of the
        particle and the suspending fluid (air, water, etc.).
        This is calculated by the mean free path of the medium divided by the particle radius."""
        return pp.MEAN_FREE_PATH_AIR / self.radius()

    @u.wraps(u.dimensionless, [None])
    def slip_correction_factor(self) -> float:
        """Returns particle's Cunningham slip correction factor.
        Checks units. [unitless]

        Dimensionless quantity accounting for non-continuum effects on small particles.
        It is a deviation from Stokes' Law.
        Stokes assumes a no-slip condition that is not correct at high Knudsen numbers.
        The slip correction factor is used to calculate the friction factor.
        See Eq 9.34 in Atmos. Chem. & Phys. (2016) for more informatiom."""
        knudsen_number: float = self.knudsen_number()
        return 1 + knudsen_number * (
            1.257 + 0.4*np.exp(-1.1/knudsen_number)
        )

    @u.wraps(u.kg / u.s, [None])
    def friction_factor(self) -> float:
        """Returns a particle's friction factor.
        Checks units. [N*s/m]

        Property of the particle's size and surrounding medium.
        Multiplying the friction factor by the fluid velocity is the drag force on the particle."""
        slip_correction_factor: float = self.slip_correction_factor()
        return (
            6 * np.pi * pp.MEDIUM_VISCOSITY * self.radius() /
            slip_correction_factor
        )

    @u.wraps(u.kg, [None, None])
    def reduced_mass(self, other) -> float:
        """Returns the reduced mass of two particles.
        Checks units. [kg]

        The reduced mass is an "effective inertial" mass.
        Allows a two-body problem to be solved as a one-body problem."""
        return self.mass() * other.mass() / (self.mass() + other.mass())

    def reduced_friction_factor(self, other) -> float:
        """Returns the reduced friction factor between two particles. Checks units. [N*s/m]

        Similar to the reduced mass.
        The reduced friction factor allows a two-body problem to be solved as a one-body problem."""
        return (
            self.friction_factor() * other.friction_factor()
            / (self.friction_factor() + other.friction_factor())
        )

    def coulomb_potential_ratio(self, other) -> float:
        """Calculates the Coulomb potential ratio."""
        numerator = -1 * self.charge() * other.charge() * (
            pp.ELEMENTARY_CHARGE_VALUE ** 2
        )
        denominator = 4 * np.pi * pp.ELECTRIC_PERMITTIVITY * (
            self.radius() + other.radius()
        )
        return (
            numerator /
            (denominator * pp.BOLTZMANN_CONSTANT * pp.TEMPERATURE)
        )

    def coulomb_enhancement_kinetic_limit(self, other) -> float:
        """Kinetic limit of Coulomb enhancement for particle--particle cooagulation."""
        coulomb_potential_ratio = self.coulomb_potential_ratio(other)
        return (
            1 + coulomb_potential_ratio if coulomb_potential_ratio >= 0
            else np.exp(coulomb_potential_ratio)
        )

    def coulomb_enhancement_continuum_limit(self, other) -> float:
        """Continuum limit of Coulomb enhancement for particle--particle coagulation."""
        coulomb_potential_ratio = self.coulomb_potential_ratio(other)
        return (
            coulomb_potential_ratio / (
                1 - np.exp(-1*coulomb_potential_ratio)
            ) if coulomb_potential_ratio != 0 
            else 1
        )
    def diffusive_knudsen_number(self, other) -> float:
        """
        Calculates the diffusive Knudsen number for a particle-particle interaction
        TODO:
        - do return statements in steps -> easier to debug
        - assess if multi-step --> higher resources
        """

        reduced_mass = self.reduced_mass(other)
        coulomb_enhancement_continuum_limit = self.coulomb_enhancement_continuum_limit(other)
        reduced_friction_factor = self.reduced_friction_factor(other)
        coulomb_enhancement_kinetic_limit = self.coulomb_enhancement_kinetic_limit(other)
        return (
            (pp.TEMPERATURE * pp.BOLTZMANN_CONSTANT * reduced_mass**0.5)
            * coulomb_enhancement_continuum_limit /
            (reduced_friction_factor * (self.radius() + other.radius())
            * coulomb_enhancement_kinetic_limit)
        )

    def dimensionless_coagulation_kernel_hard_sphere(self, other) -> float:
        """
        Calculates the dimensionless coagulation kernel for a particle-particle interaction
        """

        # Constants for the chargeless hard-sphere limit
        hsc1 = 25.836
        hsc2 = 11.211
        hsc3 = 3.502
        hsc4 = 7.211

        diffusive_knudsen_number = self.diffusive_knudsen_number(other)

        numerator = (
            (4 * np.pi * diffusive_knudsen_number**2) +
            (hsc1 * diffusive_knudsen_number**3) +
            ((8 * np.pi)**(1/2) * hsc2 * diffusive_knudsen_number**4)
        )

        denominator = (
            1
            + hsc3 * diffusive_knudsen_number
            + hsc4 * diffusive_knudsen_number**2
            + hsc2 * diffusive_knudsen_number**3
        )

        return numerator / denominator

    def collision_kernel_continuum_limit(self, other) -> float:
        """
        Calculates the collision kernel for a particle-particle interaction
        """

        diffusive_knudsen_number = self.diffusive_knudsen_number(other)
        return 4 * np.pi * (diffusive_knudsen_number**2)

    def collision_kernel_kinetic_limit(self, other) -> float:
        """
        Calculates the collision kernel for a particle-particle interaction
        """

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
