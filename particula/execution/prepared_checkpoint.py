"""Prepared configuration around canonical resident continuation."""

from __future__ import annotations

from typing import Any

import numpy as np

from particula.aerosol import Aerosol
from particula.execution import Backend, Device
from particula.execution.prepared_types import SimulationCheckpoint


def checkpoint(owner: Any) -> SimulationCheckpoint:
    """Observe errors before saving canonical bytes and last-step results."""
    from particula.execution.checkpoint import (
        ResidentCheckpointController,
        _payload,
    )

    owner.synchronize()
    if not owner._plans:
        raise ValueError("checkpoint requires a bound process plan")
    with owner._scope():
        # This owner needs nonterminal checkpoints only. A short-lived
        # controller avoids attaching a session -> controller -> session cycle
        # that would otherwise retain large GPU arrays until cyclic GC runs.
        resident = ResidentCheckpointController(
            owner._session, owner._registry, owner._guard
        ).checkpoint()
        auxiliary = tuple(
            _payload("prepared", role, array)
            for role, array in sorted(owner._auxiliary.items())
        )
    return SimulationCheckpoint(
        1, resident, owner._plans, owner.time_step, auxiliary
    )


def _host_state(resident, primary) -> Aerosol:
    """Materialize validation state from canonical bytes, never inspection."""
    from particula.gas import EnvironmentData, GasData
    from particula.particles import ParticleData

    def unpack(family, role):
        payload = primary[family, role]
        return (
            np.frombuffer(payload.data, dtype=payload.dtype)
            .reshape(payload.shape)
            .copy()
        )

    partitioning = unpack("gas", "partitioning")
    if not np.all(partitioning == partitioning[0]):
        raise ValueError("prepared restart requires shared gas partitioning")
    return Aerosol(
        particles=ParticleData(
            unpack("particles", "masses"),
            unpack("particles", "concentration"),
            unpack("particles", "charge"),
            unpack("particles", "density"),
            unpack("particles", "volume"),
            distribution_type=resident.distribution_type,
        ),
        gas=GasData(
            list(resident.gas_names),
            unpack("gas", "molar_mass"),
            unpack("gas", "concentration"),
            partitioning[0].astype(bool),
        ),
        environment=EnvironmentData(
            *(
                unpack("environment", name)
                for name in ("temperature", "pressure", "saturation_ratio")
            )
        ),
    )


def _preflight(processes, saved, backend, device):
    """Reject schema/configuration/device drift before any allocation/upload."""
    from particula._cpu_batch import duration
    from particula.execution.checkpoint import (
        ResidentCheckpoint,
        _preflight_restart,
    )
    from particula.execution.gpu_session import _preflight_cpu_session
    from particula.execution.prepared_processes import (
        _plan_process,
        _ProcessPlan,
    )

    if type(saved) is not SimulationCheckpoint:
        raise TypeError("checkpoint must be an exact SimulationCheckpoint")
    if type(saved.schema_version) is not int or saved.schema_version != 1:
        raise ValueError("unsupported prepared checkpoint schema")
    if type(saved.resident) is not ResidentCheckpoint:
        raise TypeError("invalid resident checkpoint")
    if backend is not Backend.WARP:
        raise ValueError("prepared execution requires Backend.WARP")
    target = Device(
        backend, saved.resident.device.native if device is None else device
    )
    restart = _preflight_restart(saved.resident, target)
    dt = duration(saved.time_step, 1)
    if (
        type(saved.processes) is not tuple
        or not saved.processes
        or any(type(plan) is not _ProcessPlan for plan in saved.processes)
    ):
        raise ValueError("invalid prepared checkpoint process plans")
    host = _host_state(saved.resident, restart.primary)
    _preflight_cpu_session(host.particles, host.gas, host.environment, target)
    plans = tuple(_plan_process(p, host) for p in processes)
    if plans != saved.processes or len({p.family for p in plans}) != len(plans):
        raise ValueError("runnable configuration/order differs from checkpoint")
    families = {p.family for p in plans}
    actual = {p.family for p in restart.resource_payloads}
    if (
        actual
        != families - {"wall_loss", "dilution", "ion_source", "ion_cleanup"}
        or saved.resident.communication is not None
    ):
        raise ValueError(
            "checkpoint resource families differ from process plan"
        )
    if set(saved.resident.rng_resource_processes) != families & {
        "coagulation",
        "wall_loss",
    }:
        raise ValueError("checkpoint random streams differ from process plan")
    _validate_auxiliary(saved, plans)
    return target, dt, plans


def _validate_auxiliary(saved, plans):
    """Validate the optional last-step energy before setup or upload."""
    from particula.execution.checkpoint import _validate_payload

    latent = any(
        p.family == "condensation" and p.values["latent"] is not None
        for p in plans
    )
    expected = {}
    boxes = saved.resident.dimensions.n_boxes
    if latent:
        expected["latent_heat_energy"] = (
            "<f8",
            (boxes, saved.resident.dimensions.gas_species),
        )
    if any(p.family == "ion_source" for p in plans):
        expected["ion_source_rng"] = ("<u4", (boxes,))
        expected["ion_source_pairs"] = ("<i8", (boxes,))
    if any(p.family == "ion_cleanup" for p in plans):
        expected["ion_export_particles"] = ("<i8", (boxes,))
        expected["ion_export_mass"] = (
            "<f8",
            (boxes, saved.resident.dimensions.n_species),
        )
    if type(saved.auxiliary) is not tuple or len(saved.auxiliary) != len(
        expected
    ):
        raise ValueError(
            "checkpoint auxiliary inventory differs from process plan"
        )
    seen = set()
    for item in saved.auxiliary:
        _validate_payload(item)
        spec = expected.get(item.role)
        if (
            item.family != "prepared"
            or item.role in seen
            or spec is None
            or (item.dtype, item.shape) != spec
            or item.capacity is not None
        ):
            raise ValueError("invalid prepared checkpoint auxiliary payload")
        seen.add(item.role)
        values = np.frombuffer(item.data, dtype=item.dtype)
        if not np.all(np.isfinite(values)):
            raise ValueError("checkpoint auxiliary values must be finite")
        if item.role in (
            "ion_source_pairs",
            "ion_export_particles",
            "ion_export_mass",
        ) and np.any(values < 0):
            raise ValueError("checkpoint ion inventories must be nonnegative")


def _restore_after_binding(owner, saved):
    """Undo setup-only refresh/reset writes using canonical bytes once more.

    Recompilation reconstructs private scratch and constant bindings. Some
    direct adapters initialize shared registry buffers and thermodynamics at
    setup, so restore their exact checkpoint contents after compilation. Keep
    staging alive until the dedicated stream finishes every copy.
    """
    wp, session = owner._wp, owner._session
    targets = {
        (family, role): array
        for family, role, array, _ in owner._registry._enumerate_resources()
    }
    for item in saved.resident.payloads:
        if item.family in ("particles", "gas", "environment"):
            targets[item.family, item.role] = getattr(
                getattr(session, item.family), item.role
            )
    targets.update(
        {("prepared", name): array for name, array in owner._auxiliary.items()}
    )
    staging = []
    for item in (*saved.resident.payloads, *saved.auxiliary):
        target = targets[item.family, item.role]
        values = np.frombuffer(item.data, dtype=item.dtype).reshape(item.shape)
        source = wp.array(
            values, dtype=target.dtype, device=owner.device.native
        )
        staging.append(source)
        wp.copy(target, source)
    # Existing restart restores RNG words. Compilation acquires the exact
    # published arrays without initializing or resetting them.
    owner._wait()


def resume_simulation(
    processes: tuple[Any, ...],
    saved: SimulationCheckpoint,
    *,
    backend: Backend = Backend.WARP,
    device: str | None = None,
):
    """Restore owned arrays and counters on the exact checkpoint device."""
    from particula.execution.checkpoint import restart_resident_session
    from particula.execution.prepared import PreparedSimulation

    target, dt, plans = _preflight(processes, saved, backend, device)
    owner = PreparedSimulation._empty(target, dt)
    try:
        with owner._scope():
            owner._session, owner._registry, owner._guard = (
                restart_resident_session(saved.resident, owner.device)
            )
            owner._device_failure = owner._wp.zeros(
                1, dtype=owner._wp.int32, device=owner.device.native
            )
            owner._bind(plans)
            _restore_after_binding(owner, saved)
    except BaseException:
        if owner._guard is not None:
            owner.close()
        else:
            owner._wait()
        raise
    return owner
