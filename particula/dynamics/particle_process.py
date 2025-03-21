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
from particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)


class MassCondensation(Runnable):
    """
    Handles the mass condensation process for aerosols.

    This class applies a specified condensation strategy to each particle
    in an Aerosol, updating particle mass and reducing gas concentration
    accordingly. It is designed to work with any CondensationStrategy
    subclass.

    Attributes:
        - condensation_strategy : The condensation strategy used for mass
          transfer calculations.

    Methods:
    - execute : Perform the mass condensation over a specified time step.
    - rate : Calculate the mass condensation rate for each particle.

    Examples:
        ```py title="Example Mass Condensation"
        import particula as par
        condensation = par.dyanmics.MassCondensation(
            condensation_strategy=my_strategy
        )
        updated_aerosol = condensation.execute(aerosol, time_step=1.0)
        # updated_aerosol now reflects condensed mass
        ```

    References:
    - [Aerosol Wikipedia](https://en.wikipedia.org/wiki/Aerosol)
    - Seinfeld, J. H. and Pandis, S. N., "Atmospheric Chemistry and Physics:
      From Air Pollution to Climate Change," Wiley, 2016.
    """

    def __init__(self, condensation_strategy: CondensationStrategy):
        """
        Initialize the MassCondensation process.

        Arguments:
            - condensation_strategy : The condensation strategy to use,
              responsible for calculating mass transfer.

        Returns:
            - None
        """
        self.condensation_strategy = condensation_strategy

    def execute(
        self, aerosol: Aerosol, time_step: float, sub_steps: int = 1
    ) -> Aerosol:
        """
        Perform the mass condensation process over a given time step.

        Arguments:
            - aerosol : The Aerosol instance to modify.
            - time_step : The total time interval for condensation.
            - sub_steps : Number of subdivisions for iterative calculation.

        Returns:
            - The updated aerosol object after condensation.

        Examples:
            ```py title="Example Condensation Execution"
            updated_aerosol = condensation.execute(
                aerosol, time_step=1.0, sub_steps=2
            )
            # The aerosol now has reduced/increased particle/gas mass
            ```
        """
        # loop over gas species
        for gas_species in aerosol.iterate_gas():
            # check if gas species is condensable
            if not gas_species.condensable:
                continue
            for _ in range(sub_steps):
                # calculate the condensation step for strategy
                aerosol.particles, gas_species = (
                    self.condensation_strategy.step(
                        particle=aerosol.particles,
                        gas_species=gas_species,
                        temperature=aerosol.atmosphere.temperature,
                        pressure=aerosol.atmosphere.total_pressure,
                        time_step=time_step / sub_steps,
                    )
                )
        return aerosol

    def rate(self, aerosol: Aerosol) -> Any:
        """
        Compute mass condensation rates for each particle.

        Arguments:
            - aerosol : The Aerosol instance containing particles and gases.

        Returns:
            - An array of condensation rates for each particle,
              in units of mass per unit time.

        Examples:
            ```py title="Rate Calculation Example"
            rates = condensation.rate(aerosol)
            # rates may look like array([1.2e-12, 4.5e-12, ...])
            ```
        """
        rates = np.array([], dtype=np.float64)
        # Loop over gas species in the aerosol
        for gas_species in aerosol.iterate_gas():
            # Check if the gas species is condensable
            if not gas_species.condensable:
                continue
            # print(gas_species.name)
            # Calculate the rate of condensation
            mass_rate = self.condensation_strategy.rate(
                particle=aerosol.particles,
                gas_species=gas_species,
                temperature=aerosol.atmosphere.temperature,
                pressure=aerosol.atmosphere.total_pressure,
            )
            rates = np.append(rates, mass_rate)
        return rates


class Coagulation(Runnable):
    """
    Implements a coagulation process for aerosol particles.

    This class applies a specified coagulation strategy to each particle
    in an Aerosol, merging or aggregating particles as needed, based on
    the chosen physical model.

    Attributes:
        - coagulation_strategy : The coagulation strategy used for particle
          collision calculations.

    Methods:
    - execute : Perform the coagulation step over a given time interval.
    - rate : Calculate the coagulation rate for each particle.

    Examples:
        ```py title="Example Usage"
        import particula as par
        coagulation = par.dynamics.Coagulation(
            coagulation_strategy=my_strategy
        )
        updated_aerosol = coagulation.execute(aerosol, time_step=0.5)
        # updated_aerosol now reflects coalesced or aggregated particles
        ```

    References:
        - [Aerosol Wikipedia](https://en.wikipedia.org/wiki/Aerosol)
        - Seinfeld, J. H. and Pandis, S. N., "Atmospheric Chemistry and
          Physics: From Air Pollution to Climate Change," Wiley, 2016.
    """

    def __init__(self, coagulation_strategy: CoagulationStrategyABC):
        """
        Initialize the Coagulation process.

        Arguments:
            - coagulation_strategy : The coagulation strategy to use,
              describing how particles collide and combine.
        """
        self.coagulation_strategy = coagulation_strategy

    def execute(
        self, aerosol: Aerosol, time_step: float, sub_steps: int = 1
    ) -> Aerosol:
        """
        Perform the coagulation process over a given time step.

        Arguments:
            - aerosol : The Aerosol instance to modify.
            - time_step : The total time interval for coagulation.
            - sub_steps : Number of internal subdivisions for iterative
              calculation.

        Returns:
            - Aerosol : The updated aerosol object after the coagulation step.

        Examples:
            ```py title="Example Coagulation Execution"
            updated_aerosol = coagulation.execute(
                aerosol, time_step=0.5, sub_steps=2
            )
            # The aerosol now reflects changes from particle collisions
            ```
        """
        # Loop over particles
        for _ in range(sub_steps):
            # Calculate the coagulation step for the particle
            aerosol.particles = self.coagulation_strategy.step(
                particle=aerosol.particles,
                temperature=aerosol.atmosphere.temperature,
                pressure=aerosol.atmosphere.total_pressure,
                time_step=time_step / sub_steps,
            )
        return aerosol

    def rate(self, aerosol: Aerosol) -> Any:
        """
        Compute the coagulation rate for each particle in the aerosol.

        Arguments:
            - aerosol : The Aerosol instance containing particles.

        Returns:
            - np.ndarray : An array of coagulation rates for each particle,
              in units related to particle collisions per unit time.

        Examples:
            ```py title="Coagulation Rate Calculation Example"
            rates = coagulation.rate(aerosol)
            # rates might look like array([0.1, 0.05, ...])
            ```
        """
        rates = np.array([], dtype=np.float64)
        # Calculate the net coagulation rate for the particle
        net_rate = self.coagulation_strategy.net_rate(
            particle=aerosol.particles,
            temperature=aerosol.atmosphere.temperature,
            pressure=aerosol.atmosphere.total_pressure,
        )
        rates = np.append(rates, net_rate)
        return rates
