""" statics -> dynamics
"""

import numpy as np
from scipy.interpolate import interp1d
from functools import lru_cache

from hypersolver.derivative import ord1_acc4

from particula.particle import Particle
from particula.util.coagulation_rate import CoagulationRate


def round_to_nearest_order_of_magnitude(x, log10_interval):
    power = np.log10(abs(x))
    rounded_power = round(power / log10_interval) * log10_interval
    rounded_number = 10 ** rounded_power
    return rounded_number


class Rates:
    """ The class to calculate the rates

        Parameters:
            particle (Particle): The particle object
            lazy     (bool)   : Whether to use lazy evaluation
            cache_coagulation (bool): Whether to use cached coagulation rate
            cache_percent_step (float): The fraction of the step size to use
                                        for the cache to update. 
    """

    def __init__(
        self,
        particle=None,
        lazy: bool = True,
        **kwargs
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
        self.cache_coagulation = kwargs.get("cache_coagulation", False)
        self.cache_log10_round_maxloss_conc = kwargs.get(
                "cache_log10_round_maxloss_conc",
                0.1
            )
        self.cache_log10_round_maxloss_radius = kwargs.get(
                "cache_log10_round_maxloss_radius",
                0.1
            )

        if not self.lazy:
            self.eager_coags = CoagulationRate(
                distribution=self.particle_distribution,
                radius=self.particle_radius,
                kernel=self.particle.coagulation(),
                lazy=self.lazy
            ).eager_coags

    def _coag_loss_gain(self):
        """ get both loss and gain rates
        """
        return CoagulationRate(
            distribution=self.particle_distribution,
            radius=self.particle_radius,
            kernel=self.particle.coagulation(),
        )

    def coagulation_loss(self):
        """ get the coagulation loss rate.
        """
        return self._coag_loss_gain().coag_loss() if self.lazy \
            else self.eager_coags[0]

    def coagulation_gain(self):
        """ get coagulation gain rate.
        """
        return self._coag_loss_gain().coag_gain() if self.lazy \
            else self.eager_coags[1]

    @lru_cache(maxsize=10)
    def coagulation_cached(self, conc_interval, radius_interval):
        """ get coagulation gain rate. Using a cache that refreshes if the
            interval changes by conc_interval or radius_interval changes.
        
            Parameters:
                conc_interval (float): The interval to use for the concentration
                radius_interval (float): The interval to use for the radius
        """
        return self.coagulation_gain() - self.coagulation_loss()

    def coagulation_rate(self):
        """ get the coagulation rate by summing the loss and gain rates
        """
        if self.cache_coagulation:
            cache_reference = self.particle_distribution.m

            max_index = np.argmax(cache_reference)
            max_value = round_to_nearest_order_of_magnitude(
                    cache_reference[max_index],
                    log10_interval=self.cache_log10_round_maxloss_conc
                )
            max_radius = round_to_nearest_order_of_magnitude(
                    self.particle_radius.m[max_index],
                    log10_interval=self.cache_log10_round_maxloss_radius
                )
            return self.coagulation_cached(
                    conc_interval=max_value,
                    radius_interval=max_radius
                )
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
            - self.condensation_growth_speed().m
            * self.particle_distribution.m,
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
