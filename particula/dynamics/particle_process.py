"""Runnable particle-centric aerosol processes.

Includes condensation and evaporation, coagulation, wall loss, and dilution.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from particula.aerosol import Aerosol
from particula.dynamics.dilution import (
    DilutionStrategy,
    _validate_nonnegative_scalar,
)
from particula.dynamics.nucleation.nucleation_configuration import (
    NucleationSourceConfig,
)
from particula.dynamics.nucleation.nucleation_strategies import (
    ActivationNucleationStrategy,
    KineticNucleationStrategy,
)
from particula.dynamics.nucleation.particle_source import (
    ParticleSourceCommitConfig,
    PotentialEventData,
    commit_particle_source,
    finalize_particle_source,
)
from particula.gas.environment_data import EnvironmentData
from particula.gas.gas_data import GasData
from particula.gas.species import GasSpecies
from particula.particles.exhaustion import ExhaustionControls
from particula.particles.particle_data import ParticleData
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


def _readonly_float64_vector(
    values: object,
    name: str,
) -> NDArray[np.float64]:
    """Return an owned, read-only, rank-one float64 array.

    Args:
        values: Array-compatible sidecar values.
        name: Field name used in validation errors.

    Returns:
        Fresh immutable float64 vector.

    Raises:
        TypeError: If the values cannot be converted to float64.
        ValueError: If the converted values are not rank one.
    """
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be float64-compatible") from error
    if raw.dtype.kind in "bOSUc":
        raise TypeError(f"{name} must be float64-compatible")
    try:
        result = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be float64-compatible") from error
    if result.ndim != 1:
        raise ValueError(f"{name} must be rank 1")
    result.setflags(write=False)
    return result


def _is_real_scalar(value: object) -> bool:
    """Return whether ``value`` is a non-Boolean real scalar."""
    return (
        isinstance(value, (Real, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and not isinstance(value, np.ndarray)
    )


@dataclass(frozen=True)
class NucleationCommitConfig:
    """Immutable one-box controls for CPU nucleation source commits.

    The record mirrors the P3 transaction controls while retaining public P5
    ownership. Its rank-one sidecars are copied into read-only float64 arrays;
    their required one-box length is checked immediately before each commit.

    Args:
        maximum_slot_weight: Positive maximum represented events per slot.
        source_charge: Finite charge assigned to activated source slots.
        exhaustion_controls: Immutable fixed-capacity policy controls.
        requested_scale: Requested representative-volume scale per box.
        minimum_scale: Minimum allowed representative-volume scale per box.
        minimum_volume: Minimum representative volume per box [m³].
        radius_cubed_relative_error: Allowed resampling radius-cubed error.
        mean_radius_relative_error: Allowed resampling mean-radius error.
        surface_relative_error: Allowed resampling surface-area error.
        diversity_absolute_error: Allowed resampling diversity error.
    """

    maximum_slot_weight: float
    source_charge: float
    exhaustion_controls: ExhaustionControls
    requested_scale: NDArray[np.float64]
    minimum_scale: NDArray[np.float64]
    minimum_volume: NDArray[np.float64]
    radius_cubed_relative_error: float = 1.0
    mean_radius_relative_error: float = 1.0
    surface_relative_error: float = 1.0
    diversity_absolute_error: float = 1.0

    def __post_init__(self) -> None:
        """Validate scalar controls and defensively own sidecars."""
        for name, positive in (
            ("maximum_slot_weight", True),
            ("source_charge", False),
        ):
            value = getattr(self, name)
            if not _is_real_scalar(value):
                raise TypeError(f"{name} must be a real scalar")
            normalized = float(value)
            if not np.isfinite(normalized) or (positive and normalized <= 0.0):
                qualifier = "positive" if positive else "finite"
                raise ValueError(f"{name} must be finite and {qualifier}")
            object.__setattr__(self, name, normalized)
        if not isinstance(self.exhaustion_controls, ExhaustionControls):
            raise TypeError("exhaustion_controls must be ExhaustionControls")
        for name in (
            "radius_cubed_relative_error",
            "mean_radius_relative_error",
            "surface_relative_error",
            "diversity_absolute_error",
        ):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be an exact Python float")
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("requested_scale", "minimum_scale", "minimum_volume"):
            object.__setattr__(
                self,
                name,
                _readonly_float64_vector(getattr(self, name), name),
            )


class Nucleation(RunnableABC):
    """Apply CPU-only single-box nucleation as equal sequential substeps.

    The runnable adapts the legacy aerosol facades to their existing backing
    containers and returns the identical aerosol. Each successful substep is a
    separate P3 transaction, so prior substeps persist when a later one fails;
    the full call has no rollback boundary.

    Args:
        source_config: P4 potential-rate strategy and precursor selection.
        commit_config: Immutable P5 source-commit controls.
        environment: Single-box thermodynamic state used for every substep.
    """

    def __init__(
        self,
        source_config: NucleationSourceConfig,
        commit_config: NucleationCommitConfig,
        environment: EnvironmentData,
    ) -> None:
        """Initialize a validated CPU nucleation runnable."""
        if not isinstance(source_config, NucleationSourceConfig):
            raise TypeError("source_config must be NucleationSourceConfig")
        if not isinstance(commit_config, NucleationCommitConfig):
            raise TypeError("commit_config must be NucleationCommitConfig")
        if not isinstance(environment, EnvironmentData):
            raise TypeError("environment must be EnvironmentData")
        self.source_config = source_config
        self.commit_config = commit_config
        self.environment = environment

    def _topology(  # noqa: C901
        self, aerosol: Aerosol
    ) -> tuple[ParticleData, GasData]:
        """Validate and return the supported facade backing containers."""
        try:
            particle_facade = aerosol.particles
            gas_facade = aerosol.atmosphere.partitioning_species
        except AttributeError as error:
            raise TypeError(
                "aerosol must provide particle and gas facades"
            ) from error
        if not isinstance(particle_facade, ParticleRepresentation):
            raise TypeError("aerosol particles must be ParticleRepresentation")
        if not isinstance(gas_facade, GasSpecies):
            raise TypeError("partitioning_species must be GasSpecies")
        try:
            particles = particle_facade.data
            gas = gas_facade.data
        except AttributeError as error:
            raise TypeError(
                "aerosol facades must provide backing data"
            ) from error
        if not isinstance(particles, ParticleData):
            raise TypeError("particle backing data must be ParticleData")
        if not isinstance(gas, GasData):
            raise TypeError("gas backing data must be GasData")
        if not isinstance(particles.masses, np.ndarray):
            raise TypeError("particle masses must be a numpy array")
        if not isinstance(gas.concentration, np.ndarray):
            raise TypeError("gas concentration must be a numpy array")
        if not isinstance(gas.molar_mass, np.ndarray):
            raise TypeError("gas molar_mass must be a numpy array")
        if particles.masses.ndim != 3 or particles.masses.shape[0] != 1:
            raise ValueError("particle data must have exactly one box")
        if gas.concentration.shape != (1, gas.n_species):
            raise ValueError("gas data must have exactly one box")
        if gas.molar_mass.shape != (gas.n_species,):
            raise ValueError("gas molar_mass shape must match gas species")
        if particles.masses.shape[2] != gas.n_species:
            raise ValueError("particle and gas species widths must match")
        try:
            temperature = self.environment.temperature
            pressure = self.environment.pressure
            saturation_ratio = self.environment.saturation_ratio
        except AttributeError as error:
            raise TypeError(
                "environment must provide thermodynamic arrays"
            ) from error
        if not isinstance(temperature, np.ndarray):
            raise TypeError("environment temperature must be a numpy array")
        if not isinstance(pressure, np.ndarray):
            raise TypeError("environment pressure must be a numpy array")
        if not isinstance(saturation_ratio, np.ndarray):
            raise TypeError(
                "environment saturation_ratio must be a numpy array"
            )
        if temperature.shape != (1,):
            raise ValueError("environment must have exactly one box")
        if pressure.shape != (1,):
            raise ValueError("environment must have exactly one box")
        if saturation_ratio.shape != (1, gas.n_species):
            raise ValueError(
                "environment saturation_ratio shape must match gas"
            )
        if self.source_config.precursor_index >= gas.n_species:
            raise ValueError("precursor_index is out of range for gas species")
        return particles, gas

    def _commit_config(self) -> ParticleSourceCommitConfig:
        """Create a private P3 configuration for one attempted substep."""
        return ParticleSourceCommitConfig(
            maximum_slot_weight=self.commit_config.maximum_slot_weight,
            source_charge=self.commit_config.source_charge,
            exhaustion_controls=self.commit_config.exhaustion_controls,
            requested_scale=self.commit_config.requested_scale,
            minimum_scale=self.commit_config.minimum_scale,
            minimum_volume=self.commit_config.minimum_volume,
            radius_cubed_relative_error=(
                self.commit_config.radius_cubed_relative_error
            ),
            mean_radius_relative_error=(
                self.commit_config.mean_radius_relative_error
            ),
            surface_relative_error=self.commit_config.surface_relative_error,
            diversity_absolute_error=self.commit_config.diversity_absolute_error,
        )

    def _rate_from_gas(self, gas: GasData) -> float:
        """Calculate a potential rate from validated gas backing data."""
        precursor = self.source_config.precursor_index
        strategy = cast(
            ActivationNucleationStrategy | KineticNucleationStrategy,
            self.source_config.strategy,
        )
        saturation = None
        if strategy.validity_domain.saturation is not None:
            saturation = float(self.environment.saturation_ratio[0, precursor])
        return strategy.potential_rate(
            precursor_mass_concentration=float(gas.concentration[0, precursor]),
            precursor_molar_mass=float(gas.molar_mass[precursor]),
            temperature=float(self.environment.temperature[0]),
            saturation=saturation,
        )

    def rate(self, aerosol: Aerosol) -> float:
        """Calculate the current single-box potential event rate [#/m³/s]."""
        _, gas = self._topology(aerosol)
        return self._rate_from_gas(gas)

    def execute(
        self,
        aerosol: Aerosol,
        time_step: float | np.number,
        sub_steps: int | np.integer = 1,
    ) -> Aerosol:
        """Apply nucleation in equal sequential substeps.

        Args:
            aerosol: Legacy aerosol whose backing containers are mutated.
            time_step: Finite nonnegative total duration [s].
            sub_steps: Positive number of equal source transactions.

        Returns:
            The identical aerosol instance.
        """
        if (
            isinstance(sub_steps, bool)
            or not isinstance(sub_steps, (int, np.integer))
            or sub_steps <= 0
        ):
            raise ValueError("sub_steps must be a positive integer.")
        validated_time_step = _validate_nonnegative_scalar(
            time_step, "time_step"
        )
        particles, gas = self._topology(aerosol)
        if validated_time_step == 0.0:
            return aerosol
        duration = validated_time_step / sub_steps
        strategy = cast(
            ActivationNucleationStrategy | KineticNucleationStrategy,
            self.source_config.strategy,
        )
        for _ in range(sub_steps):
            rate = self._rate_from_gas(gas)
            if rate == 0.0:
                continue
            potential_events = PotentialEventData(
                potential_rate=np.array([rate], dtype=np.float64),
                duration=duration,
            )
            demand, diagnostics = finalize_particle_source(
                potential_events,
                strategy.injection_composition,
                gas,
            )
            commit_particle_source(
                demand,
                diagnostics,
                particles,
                gas,
                self._commit_config(),
            )
        return aerosol


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
