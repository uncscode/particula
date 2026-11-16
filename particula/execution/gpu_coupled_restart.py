"""Exact boundary restart for fixed GPU events and adaptive condensation."""

from dataclasses import dataclass, replace

import warp as wp

from particula.execution.archive import read_archive, write_archive
from particula.execution.checkpoint_io import _decode, _Encoder, _types
from particula.execution.checkpoint_io import _validate as _validate_prepared
from particula.execution.cpu_configuration import condensation_identity
from particula.execution.gpu_coupled import (
    GPUCondensationControls,
    GPUCoupledSimulation,
    _condensation_profile,
    _validate_events,
)
from particula.execution.gpu_restart import _device_identity, _runtime
from particula.execution.ode import StepStatistics
from particula.execution.prepared_processes import _ProcessPlan
from particula.execution.prepared_types import SimulationCheckpoint


@dataclass(frozen=True)
class GPUCoupledCheckpoint:
    """Save event/RNG bytes and the complete deterministic boundary policy.

    No in-progress adaptive controller is carried across a fixed event boundary:
    every cycle deliberately resets it from the new accepted physical state.
    Future forcing and source schedules are supplied again by the caller.
    """

    schema_version: int
    runtime: tuple
    device: tuple
    events: SimulationCheckpoint
    configuration: object
    controls: GPUCondensationControls
    order: str
    statistics: StepStatistics
    counters: tuple[int, int, int, int]


def checkpoint_coupled(owner):
    """Download exact continuation only after a successfully completed cycle."""
    if type(owner) is not GPUCoupledSimulation:
        raise TypeError("requires GPUCoupledSimulation")
    with owner._events._lock:
        owner._ready()
        return GPUCoupledCheckpoint(
            1,
            _runtime(),
            _device_identity(wp.get_device(owner.device.native)),
            owner._events.checkpoint(),
            condensation_identity(owner._strategy),
            owner.controls,
            owner.order,
            replace(owner.statistics),
            (
                owner.coupling_cycles,
                owner.coupling_downloads,
                owner.coupling_uploads,
                owner.control_readbacks,
            ),
        )


def _validate(saved):
    """Check the complete host envelope without allocating device state."""
    if (
        type(saved) is not GPUCoupledCheckpoint
        or type(saved.schema_version) is not int
        or saved.schema_version != 1
        or saved.runtime != _runtime()
    ):
        raise ValueError("coupled GPU restart requires matching schema/runtime")
    _validate_prepared(saved.events)
    if (
        saved.events.field_specs
        or saved.events.field_payloads
        or saved.events.extension_state
        or saved.events.clock is not None
        or saved.events.host_rng is not None
        or saved.events.execution_mode != "resident"
        or any(
            type(plan) is not _ProcessPlan
            or plan.family
            not in {"nucleation", "coagulation", "wall_loss", "dilution"}
            for plan in saved.events.processes
        )
    ):
        raise ValueError("unsupported coupled event checkpoint")
    if type(saved.controls) is not GPUCondensationControls:
        raise ValueError("invalid coupled condensation controls")
    saved.controls.__post_init__()
    if saved.order not in ("events-first", "condensation-first"):
        raise ValueError("invalid coupled process order")
    if (
        type(saved.device) is not tuple
        or len(saved.device) != 5
        or type(saved.device[0]) is not str
        or type(saved.device[1]) is not str
        or type(saved.device[2]) is not bool
        or type(saved.device[3]) is not int
        or (saved.device[2] and type(saved.device[4]) is not str)
        or (not saved.device[2] and saved.device[4] is not None)
    ):
        raise ValueError("invalid coupled device identity")
    if (
        type(saved.statistics) is not StepStatistics
        or any(
            type(x) is not int or x < 0 for x in vars(saved.statistics).values()
        )
        or saved.statistics.domain_rejected > saved.statistics.rejected
        or type(saved.counters) is not tuple
        or len(saved.counters) != 4
        or any(type(x) is not int or x < 0 for x in saved.counters)
    ):
        raise ValueError("invalid coupled work counters")
    if saved.counters[0] != saved.events.resident.completed_steps:
        raise ValueError(
            "coupled counters differ from the accepted event clock"
        )


def resume_coupled(saved, events, strategy, *, device):
    """Restore exact event streams and controls on the saved explicit device."""
    from particula.execution.checkpoint import _preflight_restart
    from particula.execution.prepared_checkpoint import (
        _host_state,
        resume_simulation,
    )

    _validate(saved)
    if condensation_identity(strategy) != saved.configuration:
        raise ValueError("condensation physics differs from coupled checkpoint")
    target = wp.get_device(device)
    if _device_identity(target) != saved.device:
        raise ValueError("coupled restart requires the exact saved device")
    if target.is_capturing:
        raise RuntimeError("coupled restart cannot run during CUDA capture")
    restart = _preflight_restart(
        saved.events.resident, saved.events.resident.device
    )
    native = _host_state(saved.events.resident, restart.primary)
    _validate_events(events, native)
    result = object.__new__(GPUCoupledSimulation)
    result._configure(strategy, saved.order, saved.controls)
    plan = _condensation_profile(native, strategy, saved.controls.method)
    result._vapor = plan.values["vapor"]
    result._molar_mass = native.gas.molar_mass.copy()
    result._events = resume_simulation(events, saved.events, device=device)
    result._initialize_counters()
    result.statistics = replace(saved.statistics)
    (
        result.coupling_cycles,
        result.coupling_downloads,
        result.coupling_uploads,
        result.control_readbacks,
    ) = saved.counters
    return result


def save_coupled_checkpoint(saved, path, *, compressed=True):
    """Atomically save canonical host bytes without pickle or uploads."""
    from particula.execution.checkpoint import _preflight_restart
    from particula.execution.prepared_checkpoint import _host_state

    _validate(saved)
    restart = _preflight_restart(
        saved.events.resident, saved.events.resident.device
    )
    host = _host_state(saved.events.resident, restart.primary)
    canonical = replace(
        saved,
        events=replace(
            saved.events,
            resident=replace(
                saved.events.resident,
                particles=host.particles,
                gas=host.gas,
                environment=host.environment,
            ),
        ),
    )
    encoder = _Encoder()
    tree = encoder.encode(canonical)
    write_archive(
        path,
        kind="restart",
        metadata={
            "schema": "particula-coupled-gpu-restart",
            "version": 1,
            "checkpoint": tree,
        },
        arrays=encoder.arrays,
        blobs=encoder.blobs,
        compressed=compressed,
    )


def load_coupled_checkpoint(path, *, max_bytes=512 * 1024 * 1024):
    """Load validated host records without physics or CUDA allocation."""
    metadata, arrays, blobs = read_archive(
        path, kind="restart", max_bytes=max_bytes
    )
    try:
        if (
            metadata["schema"] != "particula-coupled-gpu-restart"
            or type(metadata["version"]) is not int
            or metadata["version"] != 1
        ):
            raise ValueError("unsupported coupled restart archive schema")
        records, enums = _types()
        saved = _decode(metadata["checkpoint"], arrays, blobs, records, enums)
        _validate(saved)
        return saved
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("malformed coupled GPU restart record") from error
