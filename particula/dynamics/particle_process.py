"""Runnable process that are particle centric.
Includes, condensation (and evaporation), coagulation, and deposition.
"""

from typing import Any
import numpy as np

# Particula imports
from particula.runnable import Runnable
from particula.aerosol import Aerosol
from particula.dynamics.condensation.condensation_strategies import (
    CondensationStrategy,
)
from particula.dynamics.coagulation.strategy import CoagulationStrategy


class MassCondensation(Runnable):
    """
    A class for running a mass condensation process.

    Args:
        condensation_strategy (CondensationStrategy): The condensation strategy
            to use.

    Methods:
        execute: Execute the mass condensation process.
        rate: Calculate the rate of mass condensation for each particle due to
            each condensable gas species.
    """

    def __init__(self, condensation_strategy: CondensationStrategy):
        self.condensation_strategy = condensation_strategy

    def execute(
        self, aerosol: Aerosol, time_step: float, sub_steps: int = 1
    ) -> Aerosol:
        """
        Execute the mass condensation process.

        Args:
            aerosol (Aerosol): The aerosol instance to modify.
        """
        # loop over gas species
        for gas_species in aerosol.iterate_gas():
            # check if gas species is condensable
            if not gas_species.condensable:
                continue
            # loop over particles to apply condensation
            for particle in aerosol.iterate_particle():
                for _ in range(sub_steps):
                    # calculate the condensation step for strategy
                    particle, gas_species = self.condensation_strategy.step(
                        particle=particle,
                        gas_species=gas_species,
                        temperature=aerosol.atmosphere.temperature,
                        pressure=aerosol.atmosphere.total_pressure,
                        time_step=time_step / sub_steps,
                    )
        return aerosol

    def rate(self, aerosol: Aerosol) -> Any:
        """
        Calculate the rate of mass condensation for each particle due to
        each condensable gas species.

        Args:
            aerosol (Aerosol): The aerosol instance to modify.

        Returns:
            np.ndarray: An array of condensation rates for each particle.
        """
        rates = np.array([], dtype=np.float64)
        # Loop over gas species in the aerosol
        for gas_species in aerosol.iterate_gas():
            # Check if the gas species is condensable
            if not gas_species.condensable:
                continue
            # Loop over particles to apply condensation
            for particle in aerosol.iterate_particle():
                # print(gas_species.name)
                # Calculate the rate of condensation
                mass_rate = self.condensation_strategy.rate(
                    particle=particle,
                    gas_species=gas_species,
                    temperature=aerosol.atmosphere.temperature,
                    pressure=aerosol.atmosphere.total_pressure,
                )
                rates = np.append(rates, mass_rate)
        return rates


class Coagulation(Runnable):
    """
    A class for running a coagulation strategy.

    Args:
        coagulation_strategy (CoagulationStrategy): The coagulation strategy to
            use.

    Methods:
        execute: Execute the coagulation process.
        rate: Calculate the rate of coagulation for each particle.
    """

    def __init__(self, coagulation_strategy: CoagulationStrategy):
        self.coagulation_strategy = coagulation_strategy

    def execute(
        self, aerosol: Aerosol, time_step: float, sub_steps: int = 1
    ) -> Aerosol:
        """
        Execute the coagulation process.

        Args:
            aerosol (Aerosol): The aerosol instance to modify.
        """
        # Loop over particles
        for particle in aerosol.iterate_particle():
            for _ in range(sub_steps):
                # Calculate the coagulation step for the particle
                particle = self.coagulation_strategy.step(
                    particle=particle,
                    temperature=aerosol.atmosphere.temperature,
                    pressure=aerosol.atmosphere.total_pressure,
                    time_step=time_step / sub_steps,
                )
        return aerosol

    def rate(self, aerosol: Aerosol) -> Any:
        """
        Calculate the rate of coagulation for each particle.

        Args:
            aerosol (Aerosol): The aerosol instance to modify.

        Returns:
            np.ndarray: An array of coagulation rates for each particle.
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
