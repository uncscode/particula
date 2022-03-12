""" to instantiate particles and calculate their properties.

    This module contains the Particle class, which is used to
    instantiate the particles and calculate their properties.
    Particles are introduced and defined by calling Particle
    class, for example:

    >>> from particula import Particle
    >>> p1 = Particle(name='my_particle', radius=1e-9, density=1e3, charge=1)

    Then, it is possible to return the properties of the particle p1:

    >>> p1.mass()

    The environment is defined by the following parameters:
    >>> from particula.aerosol_dynamics import environment
    >>> env = environment.Environment(temperature=300, pressure=1e5)

    If another particle is introduced, it is possible to calculate
    the binary coagulation coefficient:

    >>> p2 = Particle(name='my_particle2', radius=1e-9, density=1e3, charge=1)
    >>> p1.dimensioned_coagulation_kernel(p2, env)

    For more details, see below. More information to follow.
"""

import numpy as np

from particula import u
from particula.constants import (BOLTZMANN_CONSTANT, ELECTRIC_PERMITTIVITY,
                                 ELEMENTARY_CHARGE_VALUE)
from particula.environment import Environment


class Particle:
    """Class to instantiate particles and calculate their properties.

    This class represents the underlying framework for both
    particle--particle and gas--particle interactions. See detailed
    methods and functions below.

    Attributes:

        name    (str)   [no units]
        radius  (float) [m]
        density (float) [kg/m**3]
        charge  (int)   [dimensionless]
        mass    (float) [kg]
    """

    def __init__(self, name: str, radius, density, charge):
        """Constructs particle objects.

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
        """Returns the name of particle.
        """

        return self._name

    @u.wraps(u.kg, [None])
    def mass(self) -> float:
        """Returns mass of particle.

        Checks units: [kg]
        """

        return self._mass

    @u.wraps(u.m, [None])
    def radius(self) -> float:
        """Returns radius of particle.

        Checks units: [m]
        """

        return self._radius

    @u.wraps(u.kg / u.m**3, [None])
    def density(self) -> int:
        """Returns density of particle.

        Checks units: [kg/m**3]
        """

        return self._density

    @u.wraps(u.dimensionless, [None])
    def charge(self) -> int:
        """Returns number of charges on particle.

        Checks units: [dimensionless]
        """

        return self._charge

    @u.wraps(u.dimensionless, [None, None])
    def knudsen_number(self, environment: Environment) -> float:
        """Returns particle's Knudsen number.

        Checks units: [dimensionless]

        The Knudsen number reflects the relative length scales of
        the particle and the suspending fluid (air, water, etc.).
        This is calculated by the mean free path of the medium
        divided by the particle radius.
        """

        return environment.mean_free_path() / self.radius()

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
        yields the drag force on the particle.
        """

        slip_correction_factor: float = self.slip_correction_factor(
            environment
        )
        return (
            6 * np.pi * environment.dynamic_viscosity() * self.radius() /
            slip_correction_factor
        )

    @u.wraps(u.kg, [None, None])
    def reduced_mass(self, other) -> float:
        """Returns the reduced mass of two particles.

        Checks units: [kg]

        The reduced mass is an "effective inertial" mass.
        Allows a two-body problem to be solved as a one-body problem.
        """

        return self.mass() * other.mass() / (self.mass() + other.mass())

    @u.wraps(u.kg / u.s, [None, None, None])
    def reduced_friction_factor(
        self, other, environment: Environment
    ) -> float:
        """Returns the reduced friction factor between two particles.

        Checks units: [N*s/m]

        Similar to the reduced mass.
        The reduced friction factor allows a two-body problem
        to be solved as a one-body problem.
        """

        return (
            self.friction_factor(environment)
            * other.friction_factor(environment)
            / (
                self.friction_factor(environment)
                + other.friction_factor(environment)
            )
        )

    @u.wraps(u.dimensionless, [None, None, None])
    def coulomb_potential_ratio(
        self, other, environment: Environment
    ) -> float:
        """Calculates the Coulomb potential ratio.

        Checks units: [dimensionless]
        """

        numerator = -1 * self.charge() * other.charge() * (
            ELEMENTARY_CHARGE_VALUE ** 2
        )
        denominator = 4 * np.pi * ELECTRIC_PERMITTIVITY * (
            self.radius() + other.radius()
        )
        return (
            numerator /
            (denominator * BOLTZMANN_CONSTANT * environment.temperature)
        )

    @u.wraps(u.dimensionless, [None, None, None])
    def coulomb_enhancement_kinetic_limit(
        self, other, environment: Environment
    ) -> float:
        """Kinetic limit of Coulomb enhancement for particle--particle cooagulation.

        Checks units: [dimensionless]
        """

        coulomb_potential_ratio = self.coulomb_potential_ratio(
            other, environment
        )
        return (
            1 + coulomb_potential_ratio if coulomb_potential_ratio >= 0

            else np.exp(coulomb_potential_ratio)
        )

    @u.wraps(u.dimensionless, [None, None, None])
    def coulomb_enhancement_continuum_limit(
        self, other, environment: Environment
    ) -> float:
        """Continuum limit of Coulomb enhancement for particle--particle coagulation.

        Checks units: [dimensionless]
        """

        coulomb_potential_ratio = self.coulomb_potential_ratio(
            other, environment
        )
        return coulomb_potential_ratio / (
            1 - np.exp(-1*coulomb_potential_ratio)
        ) if coulomb_potential_ratio != 0 else 1

    @u.wraps(u.dimensionless, [None, None, None])
    def diffusive_knudsen_number(
        self, other, environment: Environment
    ) -> float:
        """Diffusive Knudsen number.

        Checks units: [dimensionless]

        The *diffusive* Knudsen number is different from Knudsen number.
        Ratio of:

            - numerator: mean persistence of one particle
            - denominator: effective length scale of
                particle--particle Coulombic interaction
        """

        numerator = (
            (
                environment.temperature * BOLTZMANN_CONSTANT
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

        Checks units: [dimensionless]
        """

        # Constants for the chargeless hard-sphere limit
        # see doi:
        hsc1 = 25.836
        hsc2 = 11.211
        hsc3 = 3.502
        hsc4 = 7.211
        diffusive_knudsen_number = self.diffusive_knudsen_number(
            other, environment
        )

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
    def collision_kernel_continuum_limit(
        self, other, environment: Environment
    ) -> float:
        """Continuum limit of collision kernel.

        Checks units: [dimensionless]
        """

        diffusive_knudsen_number = self.diffusive_knudsen_number(
            other, environment
        )
        return 4 * np.pi * (diffusive_knudsen_number**2)

    @u.wraps(u.dimensionless, [None, None, None])
    def collision_kernel_kinetic_limit(
        self, other, environment: Environment
    ) -> float:
        """Kinetic limit of collision kernel.

        Checks units: [dimensionless]
        """

        diffusive_knudsen_number = self.diffusive_knudsen_number(
            other, environment
        )
        return np.sqrt(8 * np.pi) * diffusive_knudsen_number

    @u.wraps(u.dimensionless, [None, None, None, None])
    def dimensionless_coagulation_kernel_parameterized(
        self,
        other,
        environment: Environment,
        authors: str = "cg2019",
    ) -> float:
        """Dimensionless particle--particle coagulation kernel.

        Checks units: [dimensionless]

        Paramaters:

            self:           particle 1
            other:          particle 2
            environment:    environment conditions
            authors:        authors of the parameterization
                - gh2012    doi.org:10.1103/PhysRevE.78.046402
                - cg2019    doi:10.1080/02786826.2019.1614522
                - hard_sphere
                (default: cg2019)
        """

        if authors == "cg2019":
            # some parameters
            corra = 2.5
            corrb = (
                4.528*np.exp(-1.088*self.coulomb_potential_ratio(
                    other, environment
                ))
            ) + (
                0.7091*np.log(1 + 1.527*self.coulomb_potential_ratio(
                    other, environment
                ))
            )

            corrc = (11.36)*(self.coulomb_potential_ratio(
                other, environment
            )**0.272) - 10.33
            corrk = - 0.003533*self.coulomb_potential_ratio(
                other, environment
            ) + 0.05971

            # mu for the parameterization
            corr_mu = (corrc/corra)*(
                (1 + corrk*((np.log(
                    self.diffusive_knudsen_number(other, environment)
                ) - corrb)/corra))**((-1/corrk) - 1)
            ) * (
                np.exp(-(1 + corrk*(np.log(
                    self.diffusive_knudsen_number(other, environment)
                ) - corrb)/corra)**(- 1/corrk))
            )

            answer = (
                # self.dimensionless_coagulation_kernel_hard_sphere(
                #     other, environment
                # ) if self.coulomb_potential_ratio(
                #     other, environment
                # ) <= 0 else
                self.dimensionless_coagulation_kernel_hard_sphere(
                    other, environment
                )*np.exp(corr_mu)
            )

        elif authors == "gh2012":
            numerator = self.coulomb_enhancement_continuum_limit(
                other, environment
            )

            denominator = 1 + 1.598*(np.minimum(
                self.diffusive_knudsen_number(other, environment),
                3*self.diffusive_knudsen_number(other, environment)/2
                / self.coulomb_potential_ratio(other, environment)
            ))**1.1709

            answer = (
                self.dimensionless_coagulation_kernel_hard_sphere(
                    other, environment
                ) if self.coulomb_potential_ratio(
                    other, environment
                ) <= 0.5 else
                numerator / denominator
            )

        elif authors == "hard_sphere":
            answer = self.dimensionless_coagulation_kernel_hard_sphere(
                other, environment
            )

        if authors not in ["gh2012", "hard_sphere", "cg2019"]:
            raise ValueError("We don't have this parameterization.")

        return answer

    @u.wraps(u.m**3 / u.s, [None, None, None, None])
    def dimensioned_coagulation_kernel(
        self,
        other,
        environment: Environment,
        authors: str = "cg2019",
    ) -> float:
        """Dimensioned particle--particle coagulation kernel.

        Checks units: [m**3/s]

        Paramaters:

            self:           particle 1
            other:          particle 2
            environment:    environment conditions
            authors:        authors of the parameterization
                - gh2012    https://doi.org/10.1103/PhysRevE.78.046402
                - cg2020    https://doi.org/XXXXXXXXXXXXXXXXXXXXXXXXXX
                - hard_sphere (from above)
                (default: cg2019)
        """

        return (
            self.dimensionless_coagulation_kernel_parameterized(
                other, environment, authors
            ) * self.reduced_friction_factor(
                other, environment
            ) * (
                self.radius() + other.radius()
            )**3 * self.coulomb_enhancement_kinetic_limit(
                other, environment
            )**2 / self.reduced_mass(
                other
            ) / self.coulomb_enhancement_continuum_limit(
                other, environment
            )
        )
