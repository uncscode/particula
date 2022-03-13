""" the particule class
"""

import numpy as np

from particula import u
from particula.constants import (BOLTZMANN_CONSTANT, ELECTRIC_PERMITTIVITY,
                                 ELEMENTARY_CHARGE_VALUE)
from particula.environment import Environment
from particula.util.friction_factor import frifac
from particula.util.input_handling import in_density, in_radius, in_scalar
from particula.util.knudsen_number import knu
from particula.util.particle_mass import mass
from particula.util.slip_correction import scf
from particula.vapor import Vapor


class Particle(Vapor):
    """ based on the Vapor(Environment) class
    """

    def __init__(self, **kwargs):
        """  particle objects.
        """
        super().__init__(**kwargs)

        self.particle_radius = in_radius(
            kwargs.get('particle_radius', None)
        )
        self.particle_density = in_density(
            kwargs.get('particle_density', 1000)
        )
        self.shape_factor = in_scalar(
            kwargs.get('shape_factor', 1)
        )
        self.volume_void = in_scalar(
            kwargs.get('volume_void', 0)
        )
        self.particle_charge = in_scalar(
            kwargs.get('particle_charge', 0)
        )
        self.kwargs = kwargs

    def mass(self):
        """ Returns mass of particle.
        """

        return mass(
            radius=self.particle_radius,
            density=self.particle_density,
            shape_factor=self.shape_factor,
            volume_void=self.volume_void,
        )

    def knudsen_number(self):
        """ Returns particle's Knudsen number.
        """

        return knu(
            radius=self.particle_radius,
            mfp=self.mean_free_path(),
        )

    def slip_correction_factor(self):
        """ Returns particle's Cunningham slip correction factor.
        """

        return scf(
            radius=self.particle_radius,
            knu=self.knudsen_number(),
        )

    def friction_factor(self):
        """ Returns a particle's friction factor.
        """

        return frifac(
            radius=self.particle_radius,
            dynamic_viscosity=self.dynamic_viscosity(),
            scf=self.slip_correction_factor(),
        )

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
