"""Native NumPy fixed-step ownership, observation and exact CPU continuation.

This owner executes native CPU runnables with CPUExecutionContext, never Warp
CPU. Each complete cycle stages state, process diagnostics and RNG together.
Restart compares signatures of explicitly supported native configurations.
It does not infer custom code identity or impose GPU capability restrictions.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from particula import _cpu_batch
from particula.aerosol import Aerosol
from particula.execution.cpu_configuration import cpu_process_identity
from particula.execution.cpu_context import CPUExecutionContext
from particula.runnable import RunnableSequence


def pack_native(state):
    """Detach native carriers, auxiliary field schemas, values and budgets."""
    copy = _cpu_batch.detached(state)
    return (
        copy.particles,
        copy.gas,
        copy.environment,
        tuple(
            (spec, copy.fields[spec.name].copy())
            for spec in copy.fields.specs.values()
        ),
        tuple(
            (name, value.copy())
            for name, value in copy.fields.transaction_arrays()
        ),
    )


def unpack_native(saved):
    """Validate and reconstruct native fields with explicit names and shapes."""
    particles, gas, environment, fields, ledgers = saved
    result = Aerosol(
        particles=particles.copy(),
        gas=gas.copy(),
        environment=environment.copy(),
    )
    for spec, values in fields:
        result.fields.add(spec, values)
    targets = dict(result.fields.transaction_arrays())
    supplied = dict(ledgers)
    if len(supplied) != len(ledgers) or set(supplied) != set(targets):
        raise ValueError("CPU restart field inventory schema differs")
    for name, value in supplied.items():
        target = targets[name]
        if (
            value.shape != target.shape
            or value.dtype != target.dtype
            or not np.all(np.isfinite(value))
        ):
            raise ValueError("CPU restart field inventory payload differs")
        target[:] = value
    _cpu_batch.validate(result)
    return result


@dataclass(frozen=True)
class CPUCheckpoint:
    """Retain native carriers and canonical CPU random continuation."""

    schema_version: int
    state: tuple
    processes: tuple
    time: float
    time_step: float
    completed_steps: int
    root_seed: int
    box_ids: tuple[str, ...]
    rng: bytes
    auxiliary: tuple


class CPUSimulation:
    """Own a detached CPU simulation with explicit duration and RNG identity.

    Supported built-in configuration is compiled for identity only; numerical
    execution always calls native CPU runnables. snapshot() is detached. Output
    times may shorten cycles; exact continuation assumes the same subsequent
    schedule and forcing. Native arrays/configuration supplied by the caller
    may subsequently change without mutating this owner.
    """

    def __init__(
        self,
        aerosol,
        process,
        *,
        time_step,
        time=0.0,
        root_seed=0,
        box_ids=None,
    ):
        """Validate built-in processes and detach state/configuration."""
        _cpu_batch.validate(aerosol)
        leaves = (
            tuple(process._leaf_processes())
            if isinstance(process, RunnableSequence)
            else (process,)
        )
        self._plans = tuple(cpu_process_identity(p) for p in leaves)
        if not leaves or any(not hasattr(p, "_namespace") for p in leaves):
            raise ValueError(
                "CPU restart profile requires native built-in runnables"
            )
        self._processes = deepcopy(leaves)
        self._state = _cpu_batch.detached(aerosol)
        self._time = _cpu_batch.duration(time, 1)
        self.time_step = _cpu_batch.duration(time_step, 1)
        if self.time_step <= 0:
            raise ValueError("time_step must be positive")
        ids = (
            tuple(str(i) for i in range(aerosol.particles.n_boxes))
            if box_ids is None
            else box_ids
        )
        self._context = CPUExecutionContext(root_seed, ids)
        self._context._begin(aerosol.particles.n_boxes)
        self.completed_steps = 0

    @property
    def time(self):
        """Return accepted absolute time [s]."""
        return self._time

    def step(self, bound):
        """Accept a complete CPU cycle; retain state/RNG/clock on failure."""
        bound = _cpu_batch.duration(bound, 1)
        if bound < self.time:
            raise ValueError("bound precedes accepted time")
        if bound == self.time:
            return self.time
        duration = min(self.time_step, bound - self.time)
        if self.time + duration == self.time:
            raise RuntimeError("cycle cannot advance floating-point time")
        candidate = _cpu_batch.detached(self._state)
        context = deepcopy(self._context)
        processes = deepcopy(self._processes)
        for process in processes:
            result = process.execute(candidate, duration, context=context)
            if result is not candidate:
                raise ValueError(
                    "CPU process replaced native state unexpectedly"
                )
        _cpu_batch.validate(candidate)
        self._state, self._context, self._processes = (
            candidate,
            context,
            processes,
        )
        self._time = (
            bound if duration == bound - self.time else self.time + duration
        )
        self.completed_steps += 1
        return self.time

    def snapshot(self):
        """Return detached native accepted state with auxiliary fields."""
        return _cpu_batch.detached(self._state)

    def checkpoint(self):
        """Detach native state and RNG bytes between accepted cycles."""
        return CPUCheckpoint(
            1,
            pack_native(self._state),
            self._plans,
            self.time,
            self.time_step,
            self.completed_steps,
            self._context.root_seed,
            self._context.box_ids,
            self._context._checkpoint(),
            self._auxiliary(),
        )

    def _auxiliary(self):
        """Detach last-step latent energy as a diagnostic."""
        return tuple(
            (
                None
                if getattr(process, "last_latent_heat_energy", None) is None
                else process.last_latent_heat_energy.copy(),
                getattr(
                    getattr(process, "condensation_strategy", None),
                    "last_latent_heat_energy",
                    None,
                ),
            )
            for process in self._processes
        )

    def _restore_auxiliary(self, auxiliary):
        """Validate per-process diagnostic state before restore."""
        if type(auxiliary) is not tuple or len(auxiliary) != len(
            self._processes
        ):
            raise ValueError("CPU auxiliary schema differs")
        for process, entry in zip(self._processes, auxiliary, strict=True):
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("CPU auxiliary entry differs")
            energy, scalar = entry
            strategy = getattr(process, "condensation_strategy", None)
            if strategy is None or not hasattr(
                strategy, "last_latent_heat_energy"
            ):
                if energy is not None or scalar is not None:
                    raise ValueError("unexpected CPU auxiliary energy")
                continue
            if energy is not None:
                if (
                    type(energy) is not np.ndarray
                    or energy.shape != (self._state.particles.n_boxes,)
                    or energy.dtype != np.float64
                    or not np.all(np.isfinite(energy))
                ):
                    raise ValueError("invalid CPU auxiliary energy")
                process.last_latent_heat_energy = energy.copy()
            if scalar is not None and (
                type(scalar) not in (int, float) or not np.isfinite(scalar)
            ):
                raise ValueError("invalid CPU auxiliary scalar energy")
            strategy.last_latent_heat_energy = scalar

    @classmethod
    def resume(cls, checkpoint, process):
        """Restore identical declared process configurations and CPU RNG."""
        if (
            type(checkpoint) is not CPUCheckpoint
            or type(checkpoint.schema_version) is not int
            or checkpoint.schema_version != 1
        ):
            raise ValueError("unsupported native CPU checkpoint")
        if (
            type(checkpoint.completed_steps) is not int
            or checkpoint.completed_steps < 0
        ):
            raise ValueError("invalid accepted step count")
        state = unpack_native(checkpoint.state)
        owner = cls(
            state,
            process,
            time_step=checkpoint.time_step,
            time=checkpoint.time,
            root_seed=checkpoint.root_seed,
            box_ids=checkpoint.box_ids,
        )
        if owner._plans != checkpoint.processes:
            raise ValueError(
                "CPU process configuration/order differs from checkpoint"
            )
        owner._context = CPUExecutionContext._restore(
            checkpoint.rng,
            root_seed=checkpoint.root_seed,
            box_ids=checkpoint.box_ids,
            namespaces={p._namespace for p in owner._processes},
        )
        owner._restore_auxiliary(checkpoint.auxiliary)
        owner.completed_steps = checkpoint.completed_steps
        return owner
