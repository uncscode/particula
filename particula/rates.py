""" statics -> dynamics
"""

import numpy as np
from scipy.interpolate import interp1d

from hypersolver.derivative import ord1_acc4

from particula.particle import Particle
from particula.util.coagulation_rate import CoagulationRate


class Rates:
    """ The class to calculate the rates

        Parameters:
            particle (Particle): The particle object
            lazy     (bool)   : Whether to use lazy evaluation
            sparse_coagulation (bool): Whether to use sparse coagulation
            sparse_factor (int): The factor to use for sparse coagulation sample
                draws. samples = sparse_factor + sparse_factor * log(radi span).
                There are two samples, a) spaced linearly (on indexes) and 
                b) based on particle concentration (on values).
            sparse_error_adjust (bool): Whether to adjust the sparse coagulation
                error in large particles, that can be more pronounce in sparse 
                coagulation. The same error appears in in the non-sparse case, 
                but to a lesser degree. 
    """

    def __init__(
        self,
        particle=None,
        lazy=True,
        sparse_coagulation=False,
        sparse_factor: int=8,
        sparse_error_adjust: bool=True,
    ):
        """ setting up the class
        """

        if particle is None or not isinstance(particle, Particle):
            raise ValueError("You must provide a baseline Particle object!")

        self.particle = particle
        self.lazy = lazy
        self.particle_distribution = self.particle.particle_distribution()
        self.particle_radius = self.particle.particle_radius
        self.particle_formation_rate = self.particle.particle_formation_rate
        self.sparse_coagulation = sparse_coagulation
        self.sparse_factor = sparse_factor
        self.sparse_error_adjust = sparse_error_adjust

        if not self.lazy:
            self.eager_coags = CoagulationRate(
                distribution=self.particle_distribution,
                radius=self.particle_radius,
                kernel=self.particle.coagulation(),
                lazy=self.lazy
            ).eager_coags

        if self.sparse_coagulation:
            # span of the distribution
            span_of_rads = np.log10(self.particle_radius[-1].m) \
                -np.log10(self.particle_radius[0].m)
            sample_number = self.sparse_factor\
                + np.ceil(span_of_rads).astype(int) * self.sparse_factor

            number_weight = self.particle.particle_distribution()\
                / np.sum(self.particle.particle_distribution())
            mass_weight = self.particle.particle_mass()\
                / np.sum(self.particle.particle_mass())
            # Draw samples, uniform, number and mass weighted
            array = np.arange(self.particle_radius.m.size)
            sample_index = np.unique(np.sort(np.concatenate((
                np.linspace(array[0], array[-1], sample_number).astype(int),
                np.random.choice(
                    array, sample_number,
                    p=number_weight.m
                    ),
                np.random.choice(
                    array, sample_number,
                    p=mass_weight.m
                    )
                ),axis=0)))
            
            # create sparse Particle object
            sparse_radius = self.particle_radius[sample_index]
            sparse_number = self.particle.particle_distribution()[sample_index]\
                * sparse_radius * self.particle.volume

            print(f'size of radius array: {sparse_radius.m.size}')
            self.sparse_particle = Particle(
                particle_radius=sparse_radius,
                particle_number=sparse_number.m)

    def _coag_loss_gain(self):
        """ get both loss and gain rates, sparse or full
        """
        if self.sparse_coagulation:
            return CoagulationRate(
                distribution=self.sparse_particle.particle_distribution(),
                radius=self.sparse_particle.particle_radius,
                kernel=self.sparse_particle.coagulation(),
            )
        else:
            return CoagulationRate(
                distribution=self.particle_distribution,
                radius=self.particle_radius,
                kernel=self.particle.coagulation(),
            )

    def coagulation_loss(self):
        """ get the coagulation loss rate. If sparse, use scipy interpolation
        """
        if self.sparse_coagulation:
            sparse_loss = interp1d(
                    self._coag_loss_gain().radius.m,
                    self._coag_loss_gain().coag_loss().m,
                    kind="cubic"
                )
            return sparse_loss(self.particle_radius.m) * self.particle_radius.u
        else:
            return self._coag_loss_gain().coag_loss() if self.lazy \
                else self.eager_coags[0]

    def coagulation_gain(self):
        """ get coagulation gain rate. If sparse, use scipy interpolation
        """
        if self.sparse_coagulation:
            sparse_gain = interp1d(
                    self._coag_loss_gain().radius.m,
                    self._coag_loss_gain().coag_gain().m,
                    kind="cubic"
                )
            return sparse_gain(self.particle_radius.m) * self.particle_radius.u
        else:
            return self._coag_loss_gain().coag_gain() if self.lazy \
                else self.eager_coags[1]

    def coagulation_rate(self):
        """ get the coagulation rate by summing the loss and gain rates
        """
        if self.sparse_error_adjust and self.sparse_coagulation:
            # there is an error in gain for large particles
            loss = self.coagulation_loss()
            gain = self.coagulation_gain()

            if self.particle_radius[-1].m > 600e-9:
                # this is the quick option, I need ot add a shift to it too
                index_start = np.argmin(np.abs(self.particle_radius.m-600e-9))
                gain[index_start:] = loss[index_start:]

            return gain - loss
        else:
            return self.coagulation_gain() - self.coagulation_loss()

    def condensation_growth_speed(self):
        """ condensation speed
        """

        return self.particle.particle_growth()

    def condensation_growth_rate(self):
        """ condensation rate
        """
        return ord1_acc4(
            - self.condensation_growth_speed().m * self.particle_distribution.m,
            self.particle_radius.m
        ) * (
            self.condensation_growth_speed().u * self.particle_distribution.u /
            self.particle_radius.u
        )

    def nucleation_rate(self):
        """ nucleation rate
        """
        result = np.zeros(
            self.condensation_growth_rate().m.shape
        )*self.particle_formation_rate.u
        result[0] = self.particle_formation_rate
        return result

    def dilution_rate(self):
        """ dilution rate
        """
        return (
            - self.particle.dilution_rate_coefficient() *
            self.particle_distribution
        )

    def wall_loss_rate(self):
        """ wall loss rate
        """
        return (
            - self.particle.wall_loss_coefficient() *
            self.particle_distribution
        )

    def sum_rates(
        self,
        coagulation=1,
        condensation=1,
        nucleation=1,
    ):
        """ sum rates
        """

        return (
            self.coagulation_rate() * coagulation +
            self.condensation_growth_rate() * condensation +
            self.nucleation_rate() * nucleation
        )
