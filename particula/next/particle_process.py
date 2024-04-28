"""Runnable process that are particle centric.
Includes, condensation (and evaporation), coagulation, and deposition.
"""

from typing import Any
import numpy as np

# Particula imports
from particula.next.runnable import Runnable
from particula.next.aerosol import Aerosol
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
                # add check for same species somewhere to ensure this works
                # calculate the rate of condensation
                mass_rate = self.condensation_strategy.mass_transfer_rate(
                    particle=particle,
                    gas_species=gas_species,
                    temperature=aerosol.gas.temperature,
                    pressure=aerosol.gas.total_pressure,
                )
                # mass rate per particle * time step * particle concentration
                mass_gain_per_bin = (
                    mass_rate * time_step * particle.concentration
                )
                # apply the mass change
                particle.add_mass(added_mass=mass_gain_per_bin)
                # remove mass from gas phase
                gas_species.add_concentration(
                    added_concentration=-mass_gain_per_bin)
        return aerosol

    def rate(self, aerosol: Aerosol) -> Any:
        """
        Calculate the rate of mass condensation for each particle due to
        each condensable gas species.

        Parameters:
        - aerosol (Aerosol): The aerosol instance to modify.

        Returns:
        - np.ndarray: An array of condensation rates for each particle.
        """
        rates = np.array([], dtype=np.float_)
        # Loop over gas species in the aerosol
        for gas_species in aerosol.iterate_gas():
            # Check if the gas species is condensable
            if not gas_species.condensable:
                continue
            # Loop over particles to apply condensation
            for particle in aerosol.iterate_particle():
                # Calculate the rate of condensation
                mass_rate = self.condensation_strategy.mass_transfer_rate(
                    particle=particle,
                    gas_species=gas_species,
                    temperature=aerosol.gas.temperature,
                    pressure=aerosol.gas.total_pressure
                )
                # Multiply mass rate by particle concentration
                rates = np.append(rates, mass_rate * particle.concentration)
        return rates
