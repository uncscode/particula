"""Compose native event decisions with registered particle-field ownership."""

from dataclasses import replace
from math import ceil, floor
from typing import Any

import numpy as np

from particula.execution.extension_registry import (
    PreparedOperation,
    ProcessCapabilities,
)
from particula.execution.preparation_context import BuiltinPreparationContext
from particula.execution.process_extensions import _RegisteredPlan


def validate_particle_fields(aerosol, plans):
    """Reject undefined particle events before uploading any state."""
    if not aerosol.fields.particle_names():
        return
    for plan in plans:
        if type(plan) is _RegisteredPlan or plan.family == "cohort":
            continue
        if plan.family not in {
            "coagulation",
            "condensation",
            "dilution",
            "wall_loss",
            "nucleation",
        }:
            raise ValueError(
                "process has no registered particle-field event contract"
            )
        if plan.family == "coagulation":
            aerosol.fields.require_additive()
        if plan.family == "nucleation":
            if plan.values["representative_volume_scaling"]:
                raise ValueError(
                    "particle fields forbid representative volume scaling"
                )
            if plan.values["resampling"]:
                aerosol.fields.require_additive()


def attach_transport(prepared, context: BuiltinPreparationContext):
    """Retain sparse overlap records from the native resampling sweep."""
    if prepared.resampling is None or not any(
        "particle" in spec.axes for spec in context.fields.specs
    ):
        return prepared
    wp, device = context.wp, context.device.native
    boxes, slots, _ = prepared.dimensions
    transport = (
        wp.zeros(boxes, dtype=wp.int32, device=device),
        wp.empty((boxes, 2 * slots), dtype=wp.int32, device=device),
        wp.empty((boxes, 2 * slots), dtype=wp.int32, device=device),
        wp.empty((boxes, 2 * slots), dtype=wp.float64, device=device),
    )
    return replace(
        prepared, resampling=replace(prepared.resampling, transport=transport)
    )


def wrap_particle_fields(context: BuiltinPreparationContext, plans, operations):
    """Retain lifecycle operations without per-step allocations or readbacks."""
    specs = tuple(s for s in context.fields.specs if "particle" in s.axes)
    if not specs:
        return operations
    wp, counts = context.wp, context.particles.concentration
    before = wp.empty_like(counts)
    fields = []
    for spec in specs:
        array = context.fields.arrays[spec.name]
        channels = 1 if len(array.shape) == 2 else array.shape[2]
        values = array.reshape((*counts.shape, channels))
        dtype = getattr(wp, spec.dtype)
        lower: float | int
        upper: float | int
        if spec.dtype == "float64":
            maximum = float(np.finfo(np.float64).max)
            lower = spec.minimum if spec.minimum is not None else -maximum
            upper = spec.maximum if spec.maximum is not None else maximum
        else:
            limits = np.iinfo(spec.dtype)
            lower = (
                max(limits.min, ceil(spec.minimum))
                if spec.minimum is not None
                else limits.min
            )
            upper = (
                min(limits.max, floor(spec.maximum))
                if spec.maximum is not None
                else limits.max
            )
        source = removed = None
        additive = int(spec.particle_policy == "additive")
        if additive:
            source = context.fields.ledgers[
                "__field_ledger__.source." + spec.name
            ].reshape((counts.shape[0], channels))
            removed = context.fields.ledgers[
                "__field_ledger__.removal." + spec.name
            ].reshape((counts.shape[0], channels))
        fields.append(
            (
                spec,
                values,
                dtype(lower),
                dtype(upper),
                dtype(0),
                dtype(spec.initial_value),
                source,
                removed,
                additive,
            )
        )
    return tuple(
        _wrap(context, plan, operation, before, fields)
        for plan, operation in zip(plans, operations, strict=True)
    )


def _wrap(context: BuiltinPreparationContext, plan, operation, before, fields):  # noqa: C901
    """Prepare a single family's event launches against fixed buffers."""
    from particula.execution import particle_field_kernels as kernels

    wp, counts = context.wp, context.particles.concentration
    failure = context.failure
    after_launches: list[tuple[Any, tuple[int, ...], tuple[Any, ...]]] = []
    copies = []
    for (
        _spec,
        values,
        lower,
        upper,
        zero,
        initial,
        source,
        removed,
        additive,
    ) in fields:
        dim = (values.shape[0], values.shape[2])
        if plan.family == "coagulation":
            resources = context.registry.acquire_coagulation(counts.shape[1])
            after_launches.append(
                (
                    kernels.merge_fields,
                    dim,
                    (
                        values,
                        resources.collision_pairs,
                        resources.n_collisions,
                        failure,
                    ),
                )
            )
        elif plan.family in ("dilution", "wall_loss", "condensation"):
            after_launches.append(
                (
                    kernels.retire_fields,
                    dim,
                    (values, zero, before, counts, removed, additive, failure),
                )
            )
        elif plan.family == "nucleation":
            binding = context.bindings["nucleation"]
            if binding.resampling is not None and additive:
                transport = binding.resampling.transport
                original = wp.empty_like(values)
                copies.append((original, values))
                after_launches.append(
                    (
                        kernels.remap_fields,
                        dim,
                        (
                            values,
                            original,
                            *transport,
                            binding.resampling.buffer_arrays[2],
                            binding.resampling.buffer_arrays[6],
                            failure,
                        ),
                    )
                )
            after_launches.append(
                (
                    kernels.birth_fields,
                    dim,
                    (
                        values,
                        initial,
                        counts,
                        binding.p4_arrays[10],
                        binding.p4_arrays[9],
                        source,
                        additive,
                        failure,
                    ),
                )
            )
        after_launches.append(
            (
                kernels.validate_fields,
                dim,
                (
                    values,
                    lower,
                    upper,
                    zero,
                    counts,
                    source,
                    removed,
                    additive,
                    failure,
                ),
            )
        )
    custom = type(plan) is _RegisteredPlan

    def enqueue():
        wp.copy(before, counts)
        for target, source in copies:
            wp.copy(target, source)
        operation()
        if custom:
            wp.launch(
                kernels.preserve_activity,
                dim=counts.shape,
                inputs=(before, counts, failure),
                device=context.device.native,
            )
        for kernel, dim, inputs in after_launches:
            if all(dim):
                wp.launch(
                    kernel, dim=dim, inputs=inputs, device=context.device.native
                )

    retained = (
        before,
        operation,
        tuple(copies),
        tuple(value for _, _, inputs in after_launches for value in inputs),
    )
    capabilities = (
        plan.configuration.capabilities
        if custom
        else ProcessCapabilities(resident=True, capture=True, checkpoint=True)
    )
    return PreparedOperation(
        plan.family + ".fields", enqueue, retained, capabilities
    )
