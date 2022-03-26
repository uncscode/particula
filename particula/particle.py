""" the particule class
"""

import numpy as np

from particula import u
from particula.constants import (BOLTZMANN_CONSTANT, ELECTRIC_PERMITTIVITY,
                                 ELEMENTARY_CHARGE_VALUE)
from particula.util.dimensionless_coagulation import DimensionlessCoagulation
from particula.util.distribution_discretization import discretize
from particula.util.friction_factor import frifac
from particula.util.input_handling import (in_density, in_handling, in_radius,
                                           in_scalar, in_volume)
from particula.util.knudsen_number import knu
from particula.util.particle_mass import mass
from particula.util.radius_cutoff import cut_rad
from particula.util.slip_correction import scf
from particula.vapor import Vapor


class BasePreParticle(Vapor):  # pylint: disable=too-many-instance-attributes
    """ the pre-particle class
    """

    def __init__(self, **kwargs):
        """  particle distribution objects.
        """
        super().__init__(**kwargs)

        self.spacing = kwargs.get("spacing", "linspace")
        self.nbins = in_scalar(kwargs.get("nbins", 1000)).m
        self.nparticles = in_scalar(kwargs.get("nparticles", 1e5))
        self.volume = in_volume(kwargs.get("volume", 1e-6))
        self.cutoff = in_scalar(kwargs.get("cutoff", 0.9999)).m
        self.gsigma = in_scalar(kwargs.get("gsigma", 1.25)).m
        self.mode = in_radius(kwargs.get("mode", 100e-9)).m
        self.kwargs = kwargs

    def pre_radius(self):
        """ Returns the radius space of the particles

            Utilizing the utility cut_rad to get 99.99% of the distribution.
            From this interval, radius is made on a linspace with nbins points.
            Note: linspace is used here to practical purposes --- often, the
            logspace treatment will return errors in the discretization due
            to the asymmetry across the interval (finer resolution for smaller
            particles, but much coarser resolution for larger particles).
        """

        (rad_start, rad_end) = cut_rad(
            cutoff=self.cutoff,
            gsigma=self.gsigma,
            mode=self.mode,
        )

        if self.spacing == "logspace":
            radius = np.logspace(
                np.log10(rad_start),
                np.log10(rad_end),
                self.nbins
            )
        elif self.spacing == "linspace":
            radius = np.linspace(
                rad_start,
                rad_end,
                self.nbins
            )
        else:
            raise ValueError("Spacing must be 'logspace' or 'linspace'!")

        return radius*u.m

    def pre_discretize(self):
        """ Returns a distribution pdf of the particles

            Utilizing the utility discretize to get make a lognorm distribution
            via scipy.stats.lognorm.pdf:
                interval: the size interval of the distribution
                gsigma  : geometric standard deviation of distribution
                mode    : geometric mean radius of the particles
        """

        return discretize(
            interval=self.pre_radius(),
            disttype="lognormal",
            gsigma=self.gsigma,
            mode=self.mode
        )

    def pre_distribution(self):
        """ Returns a distribution pdf of the particles

            Utilizing the utility discretize to get make a lognorm distribution
            via scipy.stats.lognorm.pdf:
                interval: the size interval of the distribution
                gsigma  : geometric standard deviation of distribution
                mode    : geometric mean radius of the particles
        """

        return self.nparticles*self.pre_discretize()/self.volume


class BaseParticle(BasePreParticle):
    """ based on the Vapor(Environment) class
    """

    def __init__(self, **kwargs):
        """  particle objects.
        """
        super().__init__(**kwargs)

        self.particle_radius = self.pre_radius() if kwargs.get(
            "particle_radius", None
        ) is None else in_radius(
            kwargs.get("particle_radius", None)
        )

        self.particle_density = in_density(
            kwargs.get("particle_density", 1000)
        )
        self.shape_factor = in_scalar(
            kwargs.get("shape_factor", 1)
        )
        self.volume_void = in_scalar(
            kwargs.get("volume_void", 0)
        )
        self.particle_charge = in_scalar(
            kwargs.get("particle_charge", 0)
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
            kwargs.get("elementary_charge_value", ELEMENTARY_CHARGE_VALUE),
            u.C
        )
        self.electric_permittivity = in_handling(
            kwargs.get("electric_permittivity", ELECTRIC_PERMITTIVITY),
            u.F/u.m
        )
        self.boltzmann_constant = in_handling(
            kwargs.get("boltzmann_constant", BOLTZMANN_CONSTANT),
            u.m**2*u.kg/u.s**2/u.K
        )
        self.coagulation_approximation = str(
            kwargs.get("coagulation_approximation", "hardsphere")
        )

        self.kwargs = kwargs

    def _coag_prep(self, other: "Particle"):
        """ get all related quantities to coulomb enhancement
        """
        return DimensionlessCoagulation(
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

    def reduced_mass(self, other: "Particle"):
        """ Returns the reduced mass.
        """
        return self._coag_prep(other).get_red_mass()

    def reduced_friction_factor(self, other: "Particle"):
        """ Returns the reduced friction factor between two particles.
        """
        return self._coag_prep(other).get_red_frifac()

    def coulomb_potential_ratio(self, other: "Particle"):
        """ Calculates the Coulomb potential ratio.
        """
        return self._coag_prep(other).get_ces()[0]

    def coulomb_enhancement_kinetic_limit(self, other: "Particle"):
        """ Kinetic limit of Coulomb enhancement for particle--particle cooagulation.
        """
        return self._coag_prep(other).get_ces()[1]

    def coulomb_enhancement_continuum_limit(self, other: "Particle"):
        """ Continuum limit of Coulomb enhancement for particle--particle coagulation.
        """
        return self._coag_prep(other).get_ces()[2]

    def diffusive_knudsen_number(self, other: "Particle"):
        """ Diffusive Knudsen number.
        """
        return self._coag_prep(other).get_diff_knu()

    def dimensionless_coagulation(self, other: "Particle"):
        """ Dimensionless particle--particle coagulation kernel.
        """
        return self._coag_prep(other).coag_less()

    def coagulation(self, other: "Particle"):
        """ Dimensioned particle--particle coagulation kernel
        """
        return self._coag_prep(other).coag_full()
