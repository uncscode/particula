"""Runnable process that are particle centric.
Includes, condensation (and evaporation), coagulation, and deposition.
"""

from typing import Any
import numpy as np

# Particula imports
from particula.next.runnable import Runnable
from particula.next.aerosol import Aerosol
from particula.next.dynamics.condensation import CondensationStrategy
from particula.next.dynamics.coagulation.strategy import CoagulationStrategy


class MassCondensation(Runnable):
    """
    A class for running a mass condensation process.

    Args:
    -----
    - condensation_strategy (CondensationStrategy): The condensation strategy
    to use.

    Methods:
    --------
    - execute: Execute the mass condensation process.
    - rate: Calculate the rate of mass condensation for each particle due to
    each condensable gas species.
    """

    def __init__(self, condensation_strategy: CondensationStrategy):
        self.condensation_strategy = condensation_strategy

    def execute(self, aerosol: Aerosol, time_step: float) -> Aerosol:
        """
        Execute the mass condensation process.

        Args:
        -----
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
                    temperature=aerosol.atmosphere.temperature,
                    pressure=aerosol.atmosphere.total_pressure,
                )

                # Multiply mass rate by particle concentration
                if mass_rate.ndim == 2:
                    concentration = particle.concentration[:, np.newaxis]
                else:
                    concentration = particle.concentration
                # mass rate per particle * time step * particle concentration
                mass_gain_per_bin = mass_rate * time_step * concentration
                # apply the mass change
                particle.add_mass(added_mass=mass_gain_per_bin)
                # remove mass from gas phase concentration
                if mass_rate.ndim == 2:
                    mass_gain_per_bin = np.sum(mass_gain_per_bin, axis=0)
                gas_species.add_concentration(
                    added_concentration=-mass_gain_per_bin
                )
        return aerosol

    def rate(self, aerosol: Aerosol) -> Any:
        """
        Calculate the rate of mass condensation for each particle due to
        each condensable gas species.

        Args:
        -----
        - aerosol (Aerosol): The aerosol instance to modify.

        Returns:
        --------
        - np.ndarray: An array of condensation rates for each particle.
        """
        rates = np.array([], dtype=np.float64)
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
                    temperature=aerosol.atmosphere.temperature,
                    pressure=aerosol.atmosphere.total_pressure,
                )
                # Multiply mass rate by particle concentration
                if mass_rate.ndim == 2:
                    concentration = particle.concentration[:, np.newaxis]
                else:
                    concentration = particle.concentration
                rates = np.append(rates, mass_rate * concentration)
        return rates


class Coagulation(Runnable):
    """
    A class for running a coagulation strategy.

    Args:
    -----
    - coagulation_strategy (CoagulationStrategy): The coagulation strategy to
    use.

    Methods:
    --------
    - execute: Execute the coagulation process.
    - rate: Calculate the rate of coagulation for each particle.
    """

    def __init__(self, coagulation_strategy: CoagulationStrategy):
        self.coagulation_strategy = coagulation_strategy

    def execute(self, aerosol: Aerosol, time_step: float) -> Aerosol:
        """
        Execute the coagulation process.

        Args:
        -----
        - aerosol (Aerosol): The aerosol instance to modify.
        """
        # Loop over particles
        for particle in aerosol.iterate_particle():
            # Calculate the net coagulation rate for the particle
            net_rate = self.coagulation_strategy.net_rate(
                particle=particle,
                temperature=aerosol.atmosphere.temperature,
                pressure=aerosol.atmosphere.total_pressure,
            )
            # Apply the change in distribution
            particle.add_concentration(net_rate * time_step)  # type: ignore
        return aerosol

    def rate(self, aerosol: Aerosol) -> Any:
        """
        Calculate the rate of coagulation for each particle.

        Args:
        -----
        - aerosol (Aerosol): The aerosol instance to modify.

        Returns:
        --------
        - np.ndarray: An array of coagulation rates for each particle.
        """
        rates = np.array([], dtype=np.float64)
        # Loop over particles
        for particle in aerosol.iterate_particle():
            # Calculate the net coagulation rate for the particle
            net_rate = self.coagulation_strategy.net_rate(
                particle=particle,
                temperature=aerosol.atmosphere.temperature,
                pressure=aerosol.atmosphere.total_pressure,
            )
            rates = np.append(rates, net_rate)
        return rates
