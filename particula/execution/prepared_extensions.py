"""Connect explicit public GPU adapters to the resident preparation boundary."""

from particula import _cpu_batch
from particula.box_process import _WRITABLE
from particula.execution.extension_registry import PreparedOperation
from particula.execution.process_extensions import (
    GPUPreparationContext,
    GPUProcessPlan,
    _RegisteredPlan,
)


def plan_extension(adapter, process, aerosol):
    """Freeze validated extension configuration before device allocation."""
    staged = _cpu_batch.detached(aerosol)
    before = {
        name: value.copy() for name, value in _cpu_batch.arrays(staged).items()
    }
    metadata = (
        tuple(staged.gas.name),
        staged.particles.distribution_type,
        dict(staged.fields.specs),
    )
    identities = {
        name: id(value) for name, value in _cpu_batch.arrays(staged).items()
    }
    plan = adapter.plan(process, staged)
    if type(plan) is not GPUProcessPlan:
        raise TypeError("adapter.plan must return GPUProcessPlan")
    import numpy as np

    after = _cpu_batch.arrays(staged)
    if (
        metadata
        != (
            tuple(staged.gas.name),
            staged.particles.distribution_type,
            dict(staged.fields.specs),
        )
        or before.keys() != after.keys()
        or any(
            value.dtype != after[name].dtype
            or identities[name] != id(after[name])
            or not np.array_equal(value, after[name])
            for name, value in before.items()
        )
    ):
        raise ValueError(
            "GPU adapter planning must not mutate validation state"
        )
    available = _cpu_batch.arrays(aerosol)
    if set(plan.reads + plan.writes) - available.keys():
        raise ValueError("GPU process declares unavailable arrays")
    if set(plan.writes) - (_WRITABLE | aerosol.fields.specs.keys()):
        raise ValueError("GPU process declares immutable metadata writes")
    return _RegisteredPlan(plan, adapter)


def compile_extension(plan, owner):
    """Pass a narrow resource context, never the owner, to user preparation."""
    import warp as wp

    if owner._clock is None:
        owner._clock = wp.array(
            [owner._initial_time], dtype=wp.float64, device=owner.device.native
        )
    import numpy as np

    configuration = plan.configuration
    state = {
        spec.name: wp.array(
            np.full(spec.shape, spec.initial_value, dtype=spec.dtype),
            dtype=getattr(wp, spec.dtype),
            device=owner.device.native,
        )
        for spec in configuration.state
    }
    owner._extension_state[configuration.name] = state
    arrays = {}
    for name in configuration.reads + configuration.writes:
        family, role = name.split(".", 1)
        if family in ("particles", "gas", "environment"):
            arrays[name] = getattr(getattr(owner._session, family), role)
        else:
            arrays[name] = owner._extension_fields.arrays[name]
    context = GPUPreparationContext(
        arrays=arrays,
        state=state,
        device=owner.device.native,
        stream=owner._stream,
        time_step=owner.time_step,
        clock=owner._clock,
        failure=owner._device_failure,
        logical_box_ids=owner._session.metadata.stream.logical_box_ids,
    )
    operation = plan.adapter.prepare(configuration, context)
    if type(operation) is not PreparedOperation:
        raise TypeError("adapter.prepare must return PreparedOperation")
    if operation.name != configuration.name or (
        operation.capabilities != configuration.capabilities
    ):
        raise ValueError(
            "prepared operation differs from admitted configuration"
        )
    # Keep array identity reachable to capture validation even if a callback
    # retains only an opaque kernel closure.
    operation = PreparedOperation(
        operation.name,
        operation.handler,
        (
            *operation.resources,
            *arrays.values(),
            *state.values(),
            owner._clock,
            owner._device_failure,
        ),
        operation.capabilities,
    )
    owner._bindings[plan.family] = operation
    return operation


def clock_operation(owner):
    """Retain one guarded device-clock advance after the complete sequence."""
    from particula.execution.extension_registry import ProcessCapabilities
    from particula.execution.prepared_guards import advance_clock

    wp = owner._wp
    inputs = [owner._clock, wp.float64(owner.time_step), owner._device_failure]

    def enqueue():
        wp.launch(
            advance_clock, dim=1, inputs=inputs, device=owner.device.native
        )

    return PreparedOperation(
        "execution.clock",
        enqueue,
        tuple(inputs),
        ProcessCapabilities(resident=True, capture=True, checkpoint=True),
    )
