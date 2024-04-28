"""Runable process that are particle centric.
Includes, condensation (and evaporation), coagulation, and deposition.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

# Particula imports
from particula.next.runnable import Runnable
from particula.next.aerosol import Aerosol
from particula.next.particle import Particle
from particula.next.gas import Gas
from particula.next.condensation import CondensationStrategy


class MassCondensation(Runnable):
    """
    A class for running a mass condensation process.

    Parameters:
    - condensation_strategy (CondensationStrategy): The condensation strategy
    to use.
    """

    def __init__(self, condensation_strategy: CondensationStrategy):
        self.condensation_strategy = condensation_strategy

    def execute(self, aerosol: Aerosol, time_step: float) -> Aerosol:
        """
        Execute the mass condensation process.

        Parameters:
        - aerosol (Aerosol): The aerosol instance to modify.
        """
        # loop over gas species
        for gas_species in aerosol.iterate_gas():

            # check if gas species is condensable
            if not gas_species.condensable:
                continue
            # loop over particles to apply condensation
            for particle in aerosol.iterate_particle():
                # calculate the rate of condensation
                mass_rate = self.condensation_strategy.mass_transfer_rate(
                    particle=particle,
                    gas_species=gas_species,
                    temperature=aerosol.gas.temperature,
                    pressure=aerosol.gas.total_pressure,
                )
                mass_gain_per_bin = mass_rate * time_step * particle.concentration()
                particle.concentration = particle.add_mass(mass_gain_per_bin)