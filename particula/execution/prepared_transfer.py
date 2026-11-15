"""An explicit, synchronous host boundary within a prepared sequence."""

from particula import _cpu_batch
from particula.aerosol import Aerosol
from particula.execution.cpu_context import CPUExecutionContext
from particula.execution.extension_registry import PreparedOperation


def download_state(owner):
    """Download full state after the caller has drained the owner stream."""
    from particula.gpu.conversion import (
        from_warp_environment_data,
        from_warp_gas_data,
        from_warp_particle_data,
    )

    return Aerosol(
        particles=from_warp_particle_data(
            owner._session.particles, sync=False
        ).copy(),
        gas=from_warp_gas_data(
            owner._session.gas,
            name=list(owner._session.metadata.gas_names),
            sync=False,
        ).copy(),
        environment=from_warp_environment_data(
            owner._session.environment, sync=False
        ).copy(),
        fields=owner._extension_fields.download(),
    )


def compile_transfer(plan, owner):
    """Retain upload staging and execute the explicit CPU callback."""
    from particula.cpu_transfer import _CPUTransferAdapter

    if type(plan.adapter) is not _CPUTransferAdapter:
        raise TypeError(
            "CPU transfer plans require the explicit CPUTransfer wrapper"
        )
    process = plan.adapter.process
    wp = owner._wp
    metadata = owner._session.metadata.stream
    if owner._host_context is None:
        owner._host_context = CPUExecutionContext(
            metadata.root_seed, metadata.logical_box_ids
        )
    targets = {}
    for name in plan.configuration.writes:
        family, role = name.split(".", 1)
        targets[name] = (
            getattr(getattr(owner._session, family), role)
            if family in ("particles", "gas", "environment")
            else owner._extension_fields.arrays[name]
        )
    # Reuse upload staging allocations. Publication changes values only.
    staging = {
        name: wp.empty_like(target, device=owner.device.native)
        for name, target in targets.items()
    }

    def enqueue():
        owner._wait()
        failure = int(owner._device_failure.numpy()[0])
        for observe in owner._observers:
            observe()
        if failure:
            raise ValueError("prior GPU process failed before CPU transfer")
        host = download_state(owner)
        process.execute(
            host,
            owner.time_step,
            context=owner._host_context,
            time=float(owner._clock.numpy()[0]),
        )
        arrays = _cpu_batch.arrays(host)
        # All CPU boxes have succeeded before any device candidate is copied.
        for name, target in staging.items():
            target.assign(arrays[name])
        for name, target in targets.items():
            wp.copy(target, staging[name])
        owner._wait()

    operation = PreparedOperation(
        process.name,
        enqueue,
        (*targets.values(), *staging.values()),
        plan.configuration.capabilities,
    )
    owner._bindings[plan.family] = operation
    return operation
