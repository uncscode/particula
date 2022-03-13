""" the particule class
"""

from particula import u
from particula.constants import (BOLTZMANN_CONSTANT, ELECTRIC_PERMITTIVITY,
                                 ELEMENTARY_CHARGE_VALUE)
from particula.util.diffusive_knudsen import DiffusiveKnudsen
from particula.util.dimensionless_coagulation import DimensionlessCoagulation
from particula.util.friction_factor import frifac
from particula.util.input_handling import (in_density, in_handling, in_radius,
                                           in_scalar)
from particula.util.knudsen_number import knu
from particula.util.particle_mass import mass
from particula.util.slip_correction import scf
from particula.vapor import Vapor


class BaseParticle(Vapor):
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


class Particle(BaseParticle):
    """ expanding on BaseParticle
    """

    def __init__(self, **kwargs):
        """ particle objects.
        """

        super().__init__(**kwargs)

        self.elementary_charge_value = in_handling(
            kwargs.get('elementary_charge_value', ELEMENTARY_CHARGE_VALUE),
            u.C
        )
        self.electric_permittivity = in_handling(
            kwargs.get('electric_permittivity', ELECTRIC_PERMITTIVITY),
            u.F/u.m
        )
        self.boltzmann_constant = in_handling(
            kwargs.get('boltzmann_constant', BOLTZMANN_CONSTANT),
            u.m**2*u.kg/u.s**2/u.K
        )
        self.kwargs = kwargs

    def _coag_prep(self, other: 'Particle'):
        """ get all related quantities to coulomb enhancement
        """

        return DiffusiveKnudsen(
            radius=self.particle_radius,
            other_radius=other.particle_radius,
            density=self.particle_density,
            other_density=other.particle_density,
            charge=self.particle_charge,
            other_charge=other.particle_charge,
            temperature=self.temperature,
            elementary_charge_value=self.elementary_charge_value,
            electric_permittivity=self.electric_permittivity,
            boltzmann_constant=self.boltzmann_constant,
        )

    def reduced_mass(self, other: 'Particle'):
        """ Returns the reduced mass.
        """

        return self._coag_prep(other).get_red_mass()

    def reduced_friction_factor(self, other: 'Particle'):
        """ Returns the reduced friction factor between two particles.
        """

        return self._coag_prep(other).get_red_frifac()

    def coulomb_potential_ratio(self, other: 'Particle'):
        """ Calculates the Coulomb potential ratio.
        """
        return self._coag_prep(other).get_ces()[0]

    def coulomb_enhancement_kinetic_limit(self, other: 'Particle'):
        """ Kinetic limit of Coulomb enhancement for particle--particle cooagulation.
        """
        return self._coag_prep(other).get_ces()[1]

    def coulomb_enhancement_continuum_limit(self, other: 'Particle'):
        """ Continuum limit of Coulomb enhancement for particle--particle coagulation.
        """
        return self._coag_prep(other).get_ces()[2]

    def diffusive_knudsen_number(self, other: 'Particle'):
        """ Diffusive Knudsen number.
        """
        return self._coag_prep(other).get_diff_knu()

    def dimensionless_coagulation_kernel_hard_sphere(
        self, other,
    ):
        """Dimensionless particle--particle coagulation kernel.
        """
        return DimensionlessCoagulation(
            dkn_val=self.diffusive_knudsen_number(other),
            authors="hardshpere",
        )

    # TO BE UPDATED LATER IN UTIL FIRST:
    # THEN WILL BE UPDATED HERE.
    # def collision_kernel_continuum_limit(
    #     self, other,
    # ) -> float:
    #     """Continuum limit of collision kernel.

    #     Checks units: [dimensionless]
    #     """

    #     diffusive_knudsen_number = self.diffusive_knudsen_number(
    #         other,
    #     )
    #     return 4 * np.pi * (diffusive_knudsen_number**2)

    # def collision_kernel_kinetic_limit(
    #     self, other,
    # ) -> float:
    #     """Kinetic limit of collision kernel.

    #     Checks units: [dimensionless]
    #     """

    #     diffusive_knudsen_number = self.diffusive_knudsen_number(
    #         other,
    #     )
    #     return np.sqrt(8 * np.pi) * diffusive_knudsen_number

    # def dimensionless_coagulation_kernel_parameterized(
    #     self,
    #     other,
    #     authors: str = "cg2019",
    # ) -> float:
    #     """Dimensionless particle--particle coagulation kernel.

    #     Checks units: [dimensionless]

    #     Paramaters:

    #         self:           particle 1
    #         other:          particle 2
    #         environment:    environment conditions
    #         authors:        authors of the parameterization
    #             - gh2012    doi.org:10.1103/PhysRevE.78.046402
    #             - cg2019    doi:10.1080/02786826.2019.1614522
    #             - hard_sphere
    #             (default: cg2019)
    #     """

    #     if authors == "cg2019":
    #         # some parameters
    #         corra = 2.5
    #         corrb = (
    #             4.528*np.exp(-1.088*self.coulomb_potential_ratio(
    #                 other,
    #             ))
    #         ) + (
    #             0.7091*np.log(1 + 1.527*self.coulomb_potential_ratio(
    #                 other,
    #             ))
    #         )

    #         corrc = (11.36)*(self.coulomb_potential_ratio(
    #             other,
    #         )**0.272) - 10.33
    #         corrk = - 0.003533*self.coulomb_potential_ratio(
    #             other,
    #         ) + 0.05971

    #         # mu for the parameterization
    #         corr_mu = (corrc/corra)*(
    #             (1 + corrk*((np.log(
    #                 self.diffusive_knudsen_number(other,)
    #             ) - corrb)/corra))**((-1/corrk) - 1)
    #         ) * (
    #             np.exp(-(1 + corrk*(np.log(
    #                 self.diffusive_knudsen_number(other,)
    #             ) - corrb)/corra)**(- 1/corrk))
    #         )

    #         answer = (
    #             # self.dimensionless_coagulation_kernel_hard_sphere(
    #             #     other, environment
    #             # ) if self.coulomb_potential_ratio(
    #             #     other, environment
    #             # ) <= 0 else
    #             self.dimensionless_coagulation_kernel_hard_sphere(
    #                 other,
    #             )*np.exp(corr_mu)
    #         )

    #     elif authors == "gh2012":
    #         numerator = self.coulomb_enhancement_continuum_limit(
    #             other,
    #         )

    #         denominator = 1 + 1.598*(np.minimum(
    #             self.diffusive_knudsen_number(other,),
    #             3*self.diffusive_knudsen_number(other,)/2
    #             / self.coulomb_potential_ratio(other,)
    #         ))**1.1709

    #         answer = (
    #             self.dimensionless_coagulation_kernel_hard_sphere(
    #                 other,
    #             ) if self.coulomb_potential_ratio(
    #                 other,
    #             ) <= 0.5 else
    #             numerator / denominator
    #         )

    #     elif authors == "hard_sphere":
    #         answer = self.dimensionless_coagulation_kernel_hard_sphere(
    #             other,
    #         )

    #     if authors not in ["gh2012", "hard_sphere", "cg2019"]:
    #         raise ValueError("We don't have this parameterization.")

    #     return answer

    # def dimensioned_coagulation_kernel(
    #     self,
    #     other,
    #     authors: str = "cg2019",
    # ) -> float:
    #     """Dimensioned particle--particle coagulation kernel.

    #     Checks units: [m**3/s]

    #     Paramaters:

    #         self:           particle 1
    #         other:          particle 2
    #         environment:    environment conditions
    #         authors:        authors of the parameterization
    #             - gh2012    https://doi.org/10.1103/PhysRevE.78.046402
    #             - cg2020    https://doi.org/XXXXXXXXXXXXXXXXXXXXXXXXXX
    #             - hard_sphere (from above)
    #             (default: cg2019)
    #     """

    #     return (
    #         self.dimensionless_coagulation_kernel_parameterized(
    #             other, authors
    #         ) * self.reduced_friction_factor(
    #             other,
    #         ) * (
    #             self.particle_radius + other.particle_radius
    #         )**3 * self.coulomb_enhancement_kinetic_limit(
    #             other,
    #         )**2 / self.reduced_mass(
    #             other
    #         ) / self.coulomb_enhancement_continuum_limit(
    #             other,
    #         )
    #     )
