"""Runnable particle-centric aerosol processes.

Includes condensation and evaporation, coagulation, wall loss, dilution, and
the CPU-only nucleation process.
"""

from copy import copy, deepcopy
from dataclasses import dataclass, fields, replace
from numbers import Integral, Real
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from particula.aerosol import Aerosol
from particula.aerosol_validation import validate_aerosol_structure
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
    ParticleSourceCommitConfig as _ParticleSourceCommitConfig,
)
from particula.dynamics.nucleation.particle_source import (
    PotentialEventData as _PotentialEventData,
)
from particula.dynamics.nucleation.particle_source import (
    commit_particle_source as _commit_particle_source,
)
from particula.dynamics.nucleation.particle_source import (
    finalize_particle_source as _finalize_particle_source,
)
from particula.dynamics.nucleation.species_mapping import (
    NucleationSpeciesMap,
    validate_nucleation_mapping,
)
from particula.gas.environment_data import EnvironmentData
from particula.gas.gas_data import GasData
from particula.particles.exhaustion import ExhaustionControls
from particula.particles.particle_data import ParticleData

# Particula imports
from particula.runnable import RunnableABC

from .coagulation.coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from .condensation.condensation_strategies import (
    CondensationStrategy,
)
from .wall_loss.wall_loss_strategies import WallLossStrategy

_MAX_NUCLEATION_SUB_STEPS = 10_000


def _native_process_state(
    aerosol: Aerosol,
) -> tuple[ParticleData, GasData, float, float]:
    """Read and validate the currently held single-box native process state.

    This boundary never creates a facade or caches aggregate references.
    Scientific strategies additionally validate their own physical inputs.
    """
    particles, gas, environment = (
        aerosol.particles,
        aerosol.gas,
        aerosol.environment,
    )
    validate_aerosol_structure(particles, gas, environment)
    particles.validate_representation()
    if particles.n_boxes != 1:
        raise ValueError("CPU runnables require exactly one box.")
    temperature = float(environment.temperature[0])
    pressure = float(environment.pressure[0])
    if not all(
        np.isfinite(value) and value > 0 for value in (temperature, pressure)
    ):
        raise ValueError(
            "temperature and pressure must be finite and positive."
        )
    return particles, gas, temperature, pressure


def _native_duration(time_step: object, sub_steps: object) -> float:
    """Validate equal-substep controls before any native process mutation."""
    if (
        isinstance(sub_steps, (bool, np.bool_))
        or not isinstance(sub_steps, Integral)
        or sub_steps <= 0
    ):
        raise ValueError("sub_steps must be a positive integer.")
    return _validate_nonnegative_scalar(time_step, "time_step") / int(sub_steps)


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
    """Immutable per-box controls for CPU nucleation source commits.

    The record mirrors the source transaction transaction controls while
    retaining public runnable execution
    ownership. Its rank-one sidecars are copied into read-only float64 arrays;
    their required batch length is checked immediately before every nonzero
    substep commit. The configuration is not a source transaction configuration
    and does not
    create a transaction spanning a complete ``Nucleation.execute`` call.

    Args:
        maximum_slot_weight: Positive maximum represented events per slot:
            represented counts for native Nucleation.
        source_charge: Finite dimensionless elementary-charge count assigned to
            activated source slots.
        exhaustion_controls: Immutable fixed-capacity policy controls.
        requested_scale: Requested representative-volume scale per box.
        minimum_scale: Minimum allowed representative-volume scale per box.
        minimum_volume: Minimum representative volume per box [m³].
        radius_cubed_relative_error: Allowed resampling radius-cubed error.
        mean_radius_relative_error: Allowed resampling mean-radius error.
        surface_relative_error: Allowed resampling surface-area error.
        diversity_absolute_error: Allowed resampling diversity error.

    Raises:
        TypeError: If a scalar control, tolerance, exhaustion control, or
            sidecar type is invalid.
        ValueError: If a scalar or tolerance is outside its allowed domain, or
            a sidecar is not rank one.
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


class _NativeBatchRunnable(RunnableABC):
    """Bind native single-box strategies to the shared batch transaction."""

    _strategy_attribute: str | None = None
    _namespace = ""

    def _writer_fields(self, aerosol):
        writers = ["particles.concentration"]
        resolved = aerosol.particles.distribution_type == "particle_resolved"
        if self._namespace in ("condensation", "nucleation") or (
            resolved and self._namespace in ("coagulation", "wall_loss")
        ):
            writers.extend(["particles.masses", "particles.charge"])
        if self._namespace in ("condensation", "dilution", "nucleation"):
            if self._namespace != "condensation" or getattr(
                self.condensation_strategy, "update_gases", True
            ):
                writers.append("gas.concentration")
        if self._namespace == "nucleation":
            writers.append("particles.volume")
        return tuple(writers)

    def _validate_batch(self, aerosol):
        if self._strategy_attribute is not None:
            strategy = getattr(self, self._strategy_attribute)
            expected = getattr(strategy, "distribution_type", None)
            if expected is not None and (
                expected != aerosol.particles.distribution_type
            ):
                raise ValueError(
                    "strategy distribution_type must match particle "
                    "distribution_type"
                )
        if self._namespace == "nucleation":
            if aerosol.particles.distribution_type != "particle_resolved":
                raise ValueError("Nucleation requires particle_resolved data")
            for name in ("requested_scale", "minimum_scale", "minimum_volume"):
                if getattr(self.commit_config, name).shape != (
                    aerosol.particles.n_boxes,
                ):
                    raise ValueError(f"{name} must have one entry per box")

    def _is_stochastic(self, aerosol):
        if self._namespace in ("coagulation", "wall_loss"):
            return aerosol.particles.distribution_type == "particle_resolved"
        if self._namespace == "condensation":
            strategy = self.condensation_strategy
            return getattr(strategy, "theta_mode", None) == "random" or getattr(
                strategy, "shuffle_each_step", False
            )
        return False

    def _box_worker(self, box, boxes, transaction):
        worker = copy(self)
        if self._strategy_attribute is not None:
            original = getattr(self, self._strategy_attribute)
            strategy = copy(original)
            for name in ("random_generator", "random_state"):
                if hasattr(original, name):
                    value = getattr(original, name)
                    setattr(strategy, name, deepcopy(value))
                    if transaction is not None:
                        setattr(
                            strategy,
                            name,
                            transaction.generator(self._namespace, box),
                        )
            setattr(worker, self._strategy_attribute, strategy)
        if self._namespace == "nucleation":
            worker.commit_config = replace(
                self.commit_config,
                **{
                    name: getattr(self.commit_config, name)[box : box + 1]
                    for name in (
                        "requested_scale",
                        "minimum_scale",
                        "minimum_volume",
                    )
                },
            )
        return worker

    def _prepare_commit(self, workers, uses_context):  # noqa: C901
        if self._strategy_attribute is None:
            return lambda: None
        random_updates = []
        energies = None
        original = getattr(self, self._strategy_attribute)
        if len(workers) == 1 and not uses_context:
            staged = getattr(workers[0], self._strategy_attribute)
            for name in ("random_generator", "random_state"):
                value = getattr(original, name, None)
                updated = getattr(staged, name, None)
                if isinstance(value, np.random.Generator):
                    random_updates.append(
                        (value.bit_generator, updated.bit_generator.state)
                    )
                elif isinstance(value, np.random.RandomState):
                    random_updates.append((value, updated.get_state()))
        if hasattr(original, "last_latent_heat_energy"):
            energies = np.asarray(
                [
                    getattr(
                        worker, self._strategy_attribute
                    ).last_latent_heat_energy
                    for worker in workers
                ],
                dtype=np.float64,
            )
        scalar_energy = (
            float(energies[0])
            if energies is not None and len(workers) == 1
            else None
        )

        def commit():
            for target, state in random_updates:
                if isinstance(target, np.random.RandomState):
                    target.set_state(state)
                else:
                    target.state = state
            if energies is not None:
                self.last_latent_heat_energy = energies
                original.last_latent_heat_energy = scalar_energy

        return commit


class Nucleation(_NativeBatchRunnable):
    """Apply mapped nucleation independently to native resolved boxes.

    Source rates and demand planning finalization use densities. The commit
    converts event
    density to represented counts using the current physical volume. Each
    attempted substep is atomic; earlier successful substeps remain applied.
    Gas and environment are read from the currently held aerosol every time.
    """

    _strategy_attribute = None
    _namespace = "nucleation"
    _maximum_sub_steps = _MAX_NUCLEATION_SUB_STEPS

    def __init__(
        self,
        source_config: NucleationSourceConfig,
        commit_config: NucleationCommitConfig,
    ) -> None:
        """Bind immutable source selection and count-based commit controls."""
        if not isinstance(source_config, NucleationSourceConfig):
            raise TypeError("source_config must be NucleationSourceConfig")
        if not isinstance(commit_config, NucleationCommitConfig):
            raise TypeError("commit_config must be NucleationCommitConfig")
        if not isinstance(source_config.species_map, NucleationSpeciesMap):
            raise TypeError("native nucleation requires NucleationSpeciesMap")
        self.source_config = source_config
        self.commit_config = commit_config

    def _state(
        self, aerosol: Aerosol
    ) -> tuple[ParticleData, GasData, EnvironmentData]:
        """Validate current topology and configured injection alignment."""
        if not isinstance(aerosol, Aerosol):
            raise TypeError("Nucleation requires a native Aerosol")
        particles, gas, _, _ = _native_process_state(aerosol)
        strategy = cast(
            ActivationNucleationStrategy | KineticNucleationStrategy,
            self.source_config.strategy,
        )
        validate_nucleation_mapping(
            particles,
            gas,
            self.source_config.species_map,
            np.asarray(strategy.injection_composition.molecule_counts),
        )
        if self.source_config.precursor_index >= gas.n_species:
            raise ValueError("precursor_index is out of range for gas species")
        writers = (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.volume,
            gas.concentration,
        )
        protected = (
            aerosol.environment.temperature,
            aerosol.environment.pressure,
            aerosol.environment.saturation_ratio,
        )
        if any(
            np.shares_memory(writer, field)
            for writer in writers
            for field in protected
        ):
            raise ValueError(
                "source writers must not alias environment storage"
            )
        return particles, gas, aerosol.environment

    def _rate(self, gas: GasData, environment: EnvironmentData) -> float:
        """Evaluate the current precursor and optional saturation gate."""
        precursor = self.source_config.precursor_index
        strategy = cast(
            ActivationNucleationStrategy | KineticNucleationStrategy,
            self.source_config.strategy,
        )
        saturation = None
        if strategy.validity_domain.saturation is not None:
            saturation = float(environment.saturation_ratio[0, precursor])
        return strategy.potential_rate(
            precursor_mass_concentration=float(gas.concentration[0, precursor]),
            precursor_molar_mass=float(gas.molar_mass[precursor]),
            temperature=float(environment.temperature[0]),
            saturation=saturation,
        )

    def _rate_box(self, aerosol: Aerosol) -> float:
        """Return the current potential event density rate [m^-3 s^-1]."""
        _, gas, environment = self._state(aerosol)
        return self._rate(gas, environment)

    def _execute_box(
        self,
        aerosol: Aerosol,
        time_step: float | np.number,
        sub_steps: int | np.integer = 1,
    ) -> Aerosol:
        """Commit equal sequential source substeps and retain all identities."""
        duration = _native_duration(time_step, sub_steps)
        if sub_steps > _MAX_NUCLEATION_SUB_STEPS:
            raise ValueError(
                f"sub_steps must not exceed {_MAX_NUCLEATION_SUB_STEPS}."
            )
        self._state(aerosol)
        if duration == 0:
            return aerosol
        config = _ParticleSourceCommitConfig(
            **{
                field.name: getattr(self.commit_config, field.name)
                for field in fields(self.commit_config)
            },
            species_map=self.source_config.species_map,
        )
        for _ in range(sub_steps):
            particles, gas, environment = self._state(aerosol)
            rate = self._rate(gas, environment)
            if rate == 0:
                continue
            demand, diagnostics = _finalize_particle_source(
                _PotentialEventData(
                    np.array([rate], dtype=np.float64), duration
                ),
                cast(
                    ActivationNucleationStrategy | KineticNucleationStrategy,
                    self.source_config.strategy,
                ).injection_composition,
                gas,
            )
            if np.all(diagnostics.gas_admitted_event_count == 0):
                break
            _commit_particle_source(demand, diagnostics, particles, gas, config)
        return aerosol


class MassCondensation(_NativeBatchRunnable):
    """Handles the mass condensation process for aerosols.

    Apply a strategy to the currently held native particle and gas containers.
    Each equal substep reads the current environment and preserves container
    identity. Process-owned physics and species maps define the mass transfer;
    non-native aggregates reject before strategy dispatch.

    Attributes:
        condensation_strategy: The condensation strategy used for mass
            transfer calculations.

    Methods:
    - execute : Perform the mass condensation over a specified time step.
    - rate : Calculate the mass condensation rate for each particle.

    Examples:
        ```py title="Example Mass Condensation"
        import particula as par
        condensation = par.dynamics.MassCondensation(
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

    _strategy_attribute = "condensation_strategy"
    _namespace = "condensation"

    def __init__(self, condensation_strategy: CondensationStrategy):
        """Initialize the MassCondensation process.

        Args:
            condensation_strategy: The condensation strategy to use,
                responsible for calculating mass transfer.

        Returns:
            result (None): No return value.
        """
        self.condensation_strategy = condensation_strategy

    def _execute_box(
        self,
        aerosol: Aerosol,
        time_step: float,
        sub_steps: int = 1,
    ) -> Aerosol:
        """Perform the mass condensation process over a given time step.

        Args:
            aerosol: The Aerosol instance to modify.
            time_step: The total time interval for condensation.
            sub_steps: Number of subdivisions for iterative calculation.

        Returns:
            The updated aerosol object after condensation.

        Examples:
            ```py title="Example Condensation Execution"
            updated_aerosol = condensation.execute(
                aerosol, time_step=1.0, sub_steps=2
            )
            # The aerosol now has reduced/increased particle/gas mass
            ```
        """
        if not isinstance(aerosol, Aerosol):
            raise TypeError("MassCondensation requires a native Aerosol")
        duration = _native_duration(time_step, sub_steps)
        for _ in range(sub_steps):
            particles, gas, temperature, pressure = _native_process_state(
                aerosol
            )
            result = self.condensation_strategy.step(
                particle=particles,
                gas_species=gas,
                temperature=temperature,
                pressure=pressure,
                time_step=duration,
            )
            if result[0] is not particles or result[1] is not gas:
                raise ValueError(
                    "Native condensation must retain its containers."
                )
        return aerosol

    def _rate_box(self, aerosol: Aerosol) -> Any:
        """Compute mass condensation rates for each particle.

        Args:
            aerosol: The Aerosol instance containing particles and gases.

        Returns:
            An array of condensation rates for each particle,
                in units of mass per unit time.

        Examples:
            ```py title="Rate Calculation Example"
            rates = condensation.rate(aerosol)
            # rates may look like array([1.2e-12, 4.5e-12, ...])
            ```
        """
        if not isinstance(aerosol, Aerosol):
            raise TypeError("MassCondensation requires a native Aerosol")
        particles, gas, temperature, pressure = _native_process_state(aerosol)
        return self.condensation_strategy.rate(
            particle=particles,
            gas_species=gas,
            temperature=temperature,
            pressure=pressure,
        )


class Coagulation(_NativeBatchRunnable):
    """Implements a coagulation process for aerosol particles.

    Apply a strategy to the held single-box native particle population.
    Each equal substep reads the current environment and retains the particle
    container. Non-native aggregates reject before strategy dispatch.

    Attributes:
        coagulation_strategy: The coagulation strategy used for particle
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

    _strategy_attribute = "coagulation_strategy"
    _namespace = "coagulation"

    def __init__(self, coagulation_strategy: CoagulationStrategyABC):
        """Initialize the Coagulation process.

        Args:
            coagulation_strategy: The coagulation strategy to use,
                describing how particles collide and combine.
        """
        self.coagulation_strategy = coagulation_strategy

    def _execute_box(
        self,
        aerosol: Aerosol,
        time_step: float,
        sub_steps: int = 1,
    ) -> Aerosol:
        """Perform the coagulation process over a given time step.

        Args:
            aerosol: The Aerosol instance to modify.
            time_step: The total time interval for coagulation.
            sub_steps: Number of internal subdivisions for iterative
                calculation.

        Returns:
            Aerosol: The updated aerosol object after the coagulation step.

        Examples:
            ```py title="Example Coagulation Execution"
            updated_aerosol = coagulation.execute(
                aerosol, time_step=0.5, sub_steps=2
            )
            # The aerosol now reflects changes from particle collisions
            ```
        """
        if not isinstance(aerosol, Aerosol):
            raise TypeError("Coagulation requires a native Aerosol")
        duration = _native_duration(time_step, sub_steps)
        for _ in range(sub_steps):
            particles, _, temperature, pressure = _native_process_state(aerosol)
            result = self.coagulation_strategy.step(
                particle=particles,
                temperature=temperature,
                pressure=pressure,
                time_step=duration,
            )
            if result is not particles:
                raise ValueError(
                    "Native coagulation must retain its container."
                )
        return aerosol

    def _rate_box(self, aerosol: Aerosol) -> Any:
        """Compute the coagulation rate for each particle in the aerosol.

        Args:
            aerosol: The Aerosol instance containing particles.

        Returns:
            np.ndarray: An array of coagulation rates for each particle,
                in units related to particle collisions per unit time.

        Examples:
            ```py title="Coagulation Rate Calculation Example"
            rates = coagulation.rate(aerosol)
            # rates might look like array([0.1, 0.05, ...])
            ```
        """
        if not isinstance(aerosol, Aerosol):
            raise TypeError("Coagulation requires a native Aerosol")
        particles, _, temperature, pressure = _native_process_state(aerosol)
        return self.coagulation_strategy.net_rate(
            particle=particles,
            temperature=temperature,
            pressure=pressure,
        )


class WallLoss(_NativeBatchRunnable):
    """Apply wall loss strategy to aerosol particles.

    Supports discrete, continuous PDF, and particle-resolved distributions via
    the configured wall loss strategy. The total ``time_step`` is split across
    ``sub_steps``. Native strategies reject oversized steps before writing;
    the runnable clamps concentrations after successful strategy delegation.

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

    _strategy_attribute = "wall_loss_strategy"
    _namespace = "wall_loss"

    def __init__(self, wall_loss_strategy: WallLossStrategy):
        """Create a wall loss runnable.

        Args:
            wall_loss_strategy: Strategy that provides wall loss rates and
                updates particle concentrations for the configured
                distribution type.
        """
        self.wall_loss_strategy = wall_loss_strategy

    def _clamp_non_negative(self, particle: ParticleData) -> None:
        """Clamp particle concentrations to non-negative values.

        Args:
            particle: Particle object whose concentration is clipped in place.
        """
        np.maximum(particle.concentration, 0.0, out=particle.concentration)

    def _execute_box(
        self,
        aerosol: Aerosol,
        time_step: float,
        sub_steps: int = 1,
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
        if not isinstance(aerosol, Aerosol):
            raise TypeError("WallLoss requires a native Aerosol")
        duration = _native_duration(time_step, sub_steps)
        for _ in range(sub_steps):
            particles, _, temperature, pressure = _native_process_state(aerosol)
            result = self.wall_loss_strategy.step(
                particle=particles,
                temperature=temperature,
                pressure=pressure,
                time_step=duration,
            )
            if result is not particles:
                raise ValueError("Native wall loss must retain its container.")
            self._clamp_non_negative(particles)
        return aerosol

    def _rate_box(self, aerosol: Aerosol) -> Any:
        """Return the wall loss rate for the aerosol particles.

        Args:
            aerosol: Aerosol instance containing particles to evaluate.

        Returns:
            Array of wall loss rates matching the particle representation.
        """
        if not isinstance(aerosol, Aerosol):
            raise TypeError("WallLoss requires a native Aerosol")
        particles, _, temperature, pressure = _native_process_state(aerosol)
        return self.wall_loss_strategy.rate(
            particle=particles,
            temperature=temperature,
            pressure=pressure,
        )


class Dilution(_NativeBatchRunnable):
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

    _strategy_attribute = "dilution_strategy"
    _namespace = "dilution"

    def execute(self, aerosol, time_step, sub_steps=1, *, context=None):
        """Batch native dilution; retain legacy custom strategy delegation."""
        if isinstance(self.dilution_strategy, DilutionStrategy):
            return super().execute(
                aerosol, time_step, sub_steps, context=context
            )
        if context is not None:
            raise TypeError("custom dilution strategies do not support context")
        return self._execute_box(aerosol, time_step, sub_steps)

    def rate(self, aerosol):
        """Return native batched rates or the custom strategy's own result."""
        if isinstance(self.dilution_strategy, DilutionStrategy):
            return super().rate(aerosol)
        return self._rate_box(aerosol)

    def __init__(self, dilution_strategy: DilutionStrategyProtocol):
        """Initialize the dilution runnable.

        Args:
        dilution_strategy: Public ``DilutionStrategy`` or compatible custom
            strategy that applies dilution to an aerosol.
        """
        self.dilution_strategy = dilution_strategy

    def _rate_box(self, aerosol: Aerosol) -> float | NDArray[np.float64]:
        """Delegate particle-number dilution-rate evaluation to the strategy.

        Args:
            aerosol: Aerosol whose particle concentration is evaluated.

        Returns:
            Particle-number concentration rate [1/(m³ s)] with the scalar or
            array shape returned by the strategy.
        """
        return self.dilution_strategy.rate(aerosol)

    def _execute_box(
        self,
        aerosol: Aerosol,
        time_step: float | np.number,
        sub_steps: int | np.integer = 1,
    ) -> Aerosol:
        """Apply dilution as equal, sequential substeps over a total duration.

        The runnable validates the total duration and substep count before
        calling the strategy. A ``DilutionStrategy`` validates and executes one
        preflight before the first equal native substep. Each substep is atomic;
        prior successful steps persist if a later step fails.
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
        sub_step_time_step = validated_time_step / sub_steps
        for _ in range(sub_steps):
            self.dilution_strategy.step(aerosol, sub_step_time_step)
        return aerosol
