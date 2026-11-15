"""Canonical registered state and versioned prepared-checkpoint validation."""

from dataclasses import replace
from math import isclose

import numpy as np

from particula.execution.checkpoint import _payload, _validate_payload
from particula.execution.process_extensions import (
    GPUProcessPlan,
    _RegisteredPlan,
)
from particula.extension_fields import (
    ExtensionFields,
    FieldSpec,
    field_dimensions,
)


def configurations(plans):
    """Strip live adapter implementations from checkpoint configuration."""
    return tuple(
        plan.configuration if type(plan) is _RegisteredPlan else plan
        for plan in plans
    )


def save_extensions(owner):
    """Download registered state into immutable byte payloads after sync."""
    fields = owner._extension_fields
    return dict(
        initial_time=owner._initial_time,
        execution_mode="cpu_transfer"
        if owner._host_context is not None
        else "resident",
        host_rng=(
            owner._host_context._checkpoint()
            if owner._host_context is not None
            else None
        ),
        field_specs=fields.specs if fields is not None else (),
        field_payloads=(
            tuple(
                _payload("fields", name, array)
                for name, array in fields.arrays.items()
            )
            + tuple(
                _payload("field_ledger", name, array)
                for name, array in fields.ledgers.items()
            )
            if fields is not None
            else ()
        ),
        extension_state=tuple(
            _payload("extension:" + name, role, array)
            for name, state in owner._extension_state.items()
            for role, array in state.items()
        ),
        clock=(
            _payload("execution", "clock", owner._clock)
            if owner._clock is not None
            else None
        ),
    )


def validate_envelope(saved):
    """Explicitly admit legacy built-in-only v1 or the extension-aware v2."""
    from particula.box_process import validate_time

    if type(saved.schema_version) is not int or saved.schema_version not in (
        1,
        2,
    ):
        raise ValueError("unsupported prepared checkpoint schema")
    validate_time(saved.initial_time)
    validate_time(saved.initial_time + float(saved.resident.simulated_time))
    if saved.execution_mode not in ("resident", "cpu_transfer"):
        raise ValueError("unsupported checkpoint execution mode")
    if saved.schema_version == 1 and (
        saved.initial_time != 0
        or saved.field_specs
        or saved.field_payloads
        or saved.extension_state
        or saved.clock is not None
        or saved.host_rng is not None
        or saved.execution_mode != "resident"
        or any(type(plan) is GPUProcessPlan for plan in saved.processes)
    ):
        raise ValueError("v1 checkpoints cannot contain extension state")


def _payload_map(payloads, expected):
    """Validate full inventory, primitive dtype, shape and finite bytes."""
    if type(payloads) is not tuple or len(payloads) != len(expected):
        raise ValueError("checkpoint extension payload inventory differs")
    result = {}
    for item in payloads:
        _validate_payload(item)
        key = (item.family, item.role)
        if (
            key in result
            or key not in expected
            or item.capacity is not None
            or (item.dtype, item.shape) != expected[key]
        ):
            raise ValueError("checkpoint extension payload schema differs")
        values = np.frombuffer(item.data, dtype=item.dtype).reshape(item.shape)
        if not np.all(np.isfinite(values)):
            raise ValueError("checkpoint extension values must be finite")
        result[key] = values.copy()
    return result


def restore_host_fields(saved, host):
    """Rebuild validated CPU fields from authoritative bytes before upload."""
    if type(saved.field_specs) is not tuple:
        raise TypeError("checkpoint field descriptors must be a tuple")
    dimensions = field_dimensions(host.particles, host.gas)
    fields = ExtensionFields(dimensions)
    expected = {}
    for spec in saved.field_specs:
        if type(spec) is not FieldSpec or replace(spec) != spec:
            raise ValueError("invalid checkpoint field descriptor")
        if spec.name in fields.specs:
            raise ValueError("duplicate checkpoint field descriptor")
        # Zero inactive slots are legal even when the active bound is positive.
        shape = tuple(dimensions[axis] for axis in spec.axes)
        fields.add(spec, np.full(shape, spec.initial_value, dtype=spec.dtype))
        expected["fields", spec.name] = (np.dtype(spec.dtype).str, shape)
    for name, array in fields.transaction_arrays():
        expected["field_ledger", name] = (array.dtype.str, array.shape)
    arrays = _payload_map(saved.field_payloads, expected)
    for name, target in fields.items():
        np.copyto(target, arrays["fields", name])
    for name, target in fields.transaction_arrays():
        np.copyto(target, arrays["field_ledger", name])
    fields.validate_layout(
        dimensions,
        host.particles.distribution_type,
        host.particles.concentration > 0,
    )
    from particula.aerosol import Aerosol

    return Aerosol(
        particles=host.particles,
        gas=host.gas,
        environment=host.environment,
        fields=fields,
    )


def validate_extension_state(saved, plans):
    """Check declared adapter state and device clock before any upload."""
    expected = {}
    custom = [p.configuration for p in plans if type(p) is _RegisteredPlan]
    for plan in custom:
        if not plan.capabilities.checkpoint:
            raise ValueError("registered process does not support checkpoints")
        if plan.stochastic:
            expected["extension:" + plan.name, "__rng"] = (
                "<u4",
                (saved.resident.dimensions.n_boxes,),
            )
        for spec in plan.state:
            expected["extension:" + plan.name, spec.name] = (
                np.dtype(spec.dtype).str,
                spec.shape,
            )
    _payload_map(saved.extension_state, expected)
    restore_host_rng(saved, plans)
    if not custom:
        if saved.clock is not None:
            raise ValueError("checkpoint contains an unexpected device clock")
        return
    clock = _payload_map(
        (saved.clock,), {("execution", "clock"): ("<f8", (1,))}
    )["execution", "clock"][0]
    expected_time = saved.initial_time + float(saved.resident.simulated_time)
    # The device adds dt once per step; the host guard retains exact elapsed
    # fractions. Admit the standard floating-point summation error bound while
    # preserving the saved clock bytes instead of rebuilding them.
    operations = max(1, saved.resident.completed_steps)
    epsilon = np.finfo(np.float64).eps
    rounding = operations * epsilon
    error_bound = (
        rounding
        / max(1.0 - rounding, epsilon)
        * (abs(saved.initial_time) + abs(float(saved.resident.simulated_time)))
    )
    if not isclose(
        clock, expected_time, rel_tol=0.0, abs_tol=max(error_bound, epsilon)
    ):
        raise ValueError("checkpoint device clock differs from simulation time")


def restore_targets(owner):
    """Resolve existing registered arrays; restart never replaces bindings."""
    targets = {}
    if owner._extension_fields is not None:
        targets.update(
            {
                ("fields", name): array
                for name, array in owner._extension_fields.arrays.items()
            }
        )
        targets.update(
            {
                ("field_ledger", name): array
                for name, array in owner._extension_fields.ledgers.items()
            }
        )
    targets.update(
        {
            ("extension:" + name, role): array
            for name, state in owner._extension_state.items()
            for role, array in state.items()
        }
    )
    if owner._clock is not None:
        targets["execution", "clock"] = owner._clock
    return targets


def restore_host_rng(saved, plans):
    """Validate transfer mode and host stream identities before upload."""
    from particula.execution.cpu_context import (
        CPUExecutionContext,
        _extension_rng_namespace,
    )

    transfers = [
        p.configuration
        for p in plans
        if type(p) is _RegisteredPlan
        and p.configuration.execution_mode == "cpu_transfer"
    ]
    expected_mode = "cpu_transfer" if transfers else "resident"
    if saved.execution_mode != expected_mode:
        raise ValueError("checkpoint execution mode differs from process plan")
    if not transfers:
        if saved.host_rng is not None:
            raise ValueError("unexpected host RNG checkpoint")
        return None
    metadata = saved.resident.rng_continuation
    namespaces = {
        _extension_rng_namespace(p.extension_id, p.name)
        for p in transfers
        if p.values["stochastic"]
    }
    context = CPUExecutionContext._restore(
        saved.host_rng,
        root_seed=metadata.root_seed,
        box_ids=metadata.logical_box_ids,
        namespaces=namespaces,
    )
    if saved.resident.completed_steps > 0 and saved.time_step > 0:
        expected = {
            (namespace, box)
            for namespace in namespaces
            for box in context.box_ids
        }
        if set(context._streams) != expected:
            raise ValueError("host RNG checkpoint stream inventory differs")
    return context
