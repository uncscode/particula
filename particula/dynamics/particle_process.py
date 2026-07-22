"""Runnable particle-centric aerosol processes.

Includes condensation and evaporation, coagulation, wall loss, and dilution.
"""

from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from particula.aerosol import Aerosol
from particula.dynamics.dilution import (
    DilutionStrategy,
    _validate_nonnegative_scalar,
)
from particula.gas.species import GasSpecies
from particula.particles.representation import ParticleRepresentation

# Particula imports
from particula.runnable import RunnableABC

from .coagulation.coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from .condensation.condensation_strategies import (
    CondensationStrategy,
)
from .wall_loss.wall_loss_strategies import WallLossStrategy


class DilutionStrategyProtocol(Protocol):
    """Structural contract for strategies compatible with ``Dilution``."""

    def rate(self, aerosol: Aerosol) -> float | NDArray[np.float64]:
        """Return the particle-number concentration rate."""

    def step(self, aerosol: Aerosol, time_step: float) -> Aerosol:
        """Apply one dilution step to an aerosol."""


class MassCondensation(RunnableABC):
    """Handles the mass condensation process for aerosols.

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
        """Initialize the MassCondensation process.

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
        """Perform the mass condensation process over a given time step.

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
        for _ in range(sub_steps):
            # calculate the condensation step for strategy
            particles_out, gas_out = self.condensation_strategy.step(
                particle=aerosol.particles,
                gas_species=aerosol.atmosphere.partitioning_species,
                temperature=aerosol.atmosphere.temperature,
                pressure=aerosol.atmosphere.total_pressure,
                time_step=time_step / sub_steps,
            )
            aerosol.particles = cast(ParticleRepresentation, particles_out)
            aerosol.atmosphere.partitioning_species = cast(GasSpecies, gas_out)
        return aerosol

    def rate(self, aerosol: Aerosol) -> Any:
        """Compute mass condensation rates for each particle.

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
        return self.condensation_strategy.rate(
            particle=aerosol.particles,
            gas_species=aerosol.atmosphere.partitioning_species,
            temperature=aerosol.atmosphere.temperature,
            pressure=aerosol.atmosphere.total_pressure,
        )


class Coagulation(RunnableABC):
    """Implements a coagulation process for aerosol particles.

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
        """Initialize the Coagulation process.

        Arguments:
            - coagulation_strategy : The coagulation strategy to use,
              describing how particles collide and combine.
        """
        self.coagulation_strategy = coagulation_strategy

    def execute(
        self, aerosol: Aerosol, time_step: float, sub_steps: int = 1
    ) -> Aerosol:
        """Perform the coagulation process over a given time step.

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
            )  # type: ignore[assignment]
        return aerosol

    def rate(self, aerosol: Aerosol) -> Any:
        """Compute the coagulation rate for each particle in the aerosol.

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


class WallLoss(RunnableABC):
    """Apply wall loss strategy to aerosol particles.

    Supports discrete, continuous PDF, and particle-resolved distributions via
    the configured wall loss strategy. The total ``time_step`` is split across
    ``sub_steps`` and concentrations are clamped to non-negative values after
    each sub-step to avoid negative counts from aggressive steps.

    Example:
        >>> import particula as par
        >>> strategy = par.dynamics.SphericalWallLossStrategy(
        ...     wall_eddy_diffusivity=0.001,
        ...     chamber_radius=0.5,
        ...     distribution_type="discrete",
        ... )
        >>> wall_loss = par.dynamics.WallLoss(
        ...     wall_loss_strategy=strategy,
        ... )
        >>> _ = wall_loss.execute(aerosol, time_step=1.0, sub_steps=2)
    """

    def __init__(self, wall_loss_strategy: WallLossStrategy):
        """Create a wall loss runnable.

        Args:
            wall_loss_strategy: Strategy that provides wall loss rates and
                updates particle concentrations for the configured
                distribution type.
        """
        self.wall_loss_strategy = wall_loss_strategy

    def _clamp_non_negative(self, particle: Any) -> None:
        """Clamp particle concentrations to non-negative values.

        Args:
            particle: Particle object whose concentration is clipped in place.
        """
        concentration = particle.get_concentration()
        clipped_concentration = np.clip(concentration, 0.0, None)
        if not np.array_equal(clipped_concentration, concentration):
            particle.concentration = (
                clipped_concentration * particle.get_volume()
            )

    def execute(
        self, aerosol: Aerosol, time_step: float, sub_steps: int = 1
    ) -> Aerosol:
        """Apply wall loss over the provided time step.

        Concentrations are clamped to remain non-negative after each
        sub-step.

        Args:
            aerosol: Aerosol instance to update.
            time_step: Total simulation interval in seconds.
            sub_steps: Number of internal steps used to split ``time_step``.

        Returns:
            Aerosol with updated particle concentrations.
        """
        for _ in range(sub_steps):
            aerosol.particles = self.wall_loss_strategy.step(
                particle=aerosol.particles,
                temperature=aerosol.atmosphere.temperature,
                pressure=aerosol.atmosphere.total_pressure,
                time_step=time_step / sub_steps,
            )
            self._clamp_non_negative(aerosol.particles)
        return aerosol

    def rate(self, aerosol: Aerosol) -> Any:
        """Return the wall loss rate for the aerosol particles.

        Args:
            aerosol: Aerosol instance containing particles to evaluate.

        Returns:
            Array of wall loss rates matching the particle representation.
        """
        return self.wall_loss_strategy.rate(
            particle=aerosol.particles,
            temperature=aerosol.atmosphere.temperature,
            pressure=aerosol.atmosphere.total_pressure,
        )


class Dilution(RunnableABC):
    """Apply a dilution strategy over one or more equal substeps.

    Each substep delegates aerosol mutation to the configured strategy. The
    runnable preserves the input aerosol identity and ignores a strategy's
    return value. When configured with the public ``DilutionStrategy``, it
    validates concrete aerosol state before its first delegated substep, so
    malformed state cannot permit a partial multi-substep update. Compatible
    custom strategies retain generic equal-substep delegation and own their
    validation and atomicity.

    Args:
        dilution_strategy: Public ``DilutionStrategy`` or a compatible custom
            strategy that reports particle-number rates and applies aerosol
            dilution steps.
    """

    def __init__(self, dilution_strategy: DilutionStrategyProtocol):
        """Initialize the dilution runnable.

        Args:
        dilution_strategy: Public ``DilutionStrategy`` or compatible custom
            strategy that applies dilution to an aerosol.
        """
        self.dilution_strategy = dilution_strategy

    def rate(self, aerosol: Aerosol) -> float | NDArray[np.float64]:
        """Delegate particle-number dilution-rate evaluation to the strategy.

        Args:
            aerosol: Aerosol whose particle concentration is evaluated.

        Returns:
            Particle-number concentration rate [1/(m³ s)] with the scalar or
            array shape returned by the strategy.
        """
        return self.dilution_strategy.rate(aerosol)

    def execute(
        self,
        aerosol: Aerosol,
        time_step: float | np.number,
        sub_steps: int | np.integer = 1,
    ) -> Aerosol:
        """Apply dilution as equal, sequential substeps over a total duration.

        The runnable validates the total duration and substep count before
        calling the strategy. A ``DilutionStrategy`` validates and executes one
        total-duration exact step, making the concrete path whole-call atomic.
        Custom strategies are not inspected and retain equal-substep delegation
        with responsibility for aerosol validation, mutation, and atomicity.

        Args:
            aerosol: Aerosol to mutate in place.
            time_step: Total elapsed time [s], finite and nonnegative.
            sub_steps: Positive count of equal internal dilution steps.

        Returns:
            The identical, mutated aerosol instance.

        Raises:
            ValueError: If ``sub_steps`` is not a positive integer,
                ``time_step`` is nonfinite, negative, or nonscalar, or
                supported concrete aerosol state is invalid.
            TypeError: If ``time_step`` is not numeric or a required supported
                concrete aerosol value is not numeric.
        """
        if (
            isinstance(sub_steps, bool)
            or not isinstance(sub_steps, (int, np.integer))
            or sub_steps <= 0
        ):
            raise ValueError("sub_steps must be a positive integer.")

        validated_time_step = _validate_nonnegative_scalar(
            time_step,
            "time_step",
        )
        if isinstance(self.dilution_strategy, DilutionStrategy):
            self.dilution_strategy._preflight(aerosol, validated_time_step)
            self.dilution_strategy.step(aerosol, validated_time_step)
            return aerosol
        sub_step_time_step = validated_time_step / sub_steps
        for _ in range(sub_steps):
            self.dilution_strategy.step(aerosol, sub_step_time_step)
        return aerosol
