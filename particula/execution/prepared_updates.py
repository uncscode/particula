"""Explicit host forcing and thermodynamic freshness for prepared sessions."""

from __future__ import annotations

from typing import Any

import numpy as np

from particula.execution.preparation_context import BuiltinPreparationContext


def forcing_array(value, shape, name, *, positive):
    """Own finite physical host values with scalar/species/box broadcasting."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    try:
        result = np.broadcast_to(raw, shape).astype(np.float64).copy()
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to {shape}") from exc
    if not np.all(np.isfinite(result)) or np.any(
        result <= 0 if positive else result < 0
    ):
        raise ValueError(f"{name} has nonphysical values")
    return result


def apply_forcing(owner: Any, updates: tuple[tuple[Any, np.ndarray], ...]):
    """Observe prior work and finish an explicit, validated upload boundary."""
    owner.synchronize()
    try:
        with owner._scope():
            # Hold staging allocations until all copies/refreshes have finished.
            staging = tuple(
                owner._wp.array(
                    values, dtype=owner._wp.float64, device=owner.device.native
                )
                for _, values in updates
            )
            for (target, _), source in zip(updates, staging, strict=True):
                owner._wp.copy(target, source)
            if owner._refresh is not None:
                owner._refresh()
            owner._wait()
    except BaseException:
        owner._fault()
        raise


def compose_operations(context: BuiltinPreparationContext, plans, operations):
    """Refresh saturation before declared consumers and after complete steps."""
    if "condensation" not in context.bindings:
        return operations
    from particula.execution.prepared_guards import refresh_saturation
    from particula.gpu.kernels.thermodynamics import (
        _refresh_vapor_pressure_kernel_guarded,
    )

    prepared = context.bindings["condensation"].prepared
    wp = context.wp
    saturation = context.environment.saturation_ratio
    failure = context.failure
    dim = context.gas.concentration.shape
    vapor_inputs = [
        prepared.thermodynamics_modes,
        prepared.thermodynamics_parameters,
        prepared.refresh_temperature,
        context.gas.vapor_pressure,
        failure,
    ]
    ratio_inputs = [
        context.gas.concentration,
        prepared.gas_molar_mass,
        context.gas.vapor_pressure,
        prepared.refresh_temperature,
        saturation,
        failure,
    ]

    def refresh():
        wp.launch(
            _refresh_vapor_pressure_kernel_guarded,
            dim=dim,
            inputs=vapor_inputs,
            device=prepared.device,
        )
        wp.launch(
            refresh_saturation,
            dim=dim,
            inputs=ratio_inputs,
            device=prepared.device,
        )

    from particula.execution.extension_registry import (
        PreparedOperation,
        ProcessCapabilities,
    )

    refresh_operation = PreparedOperation(
        "execution.thermodynamics",
        refresh,
        (tuple(vapor_inputs), tuple(ratio_inputs)),
        ProcessCapabilities(resident=True, capture=True, checkpoint=True),
    )
    context.refresh = refresh_operation
    from particula.execution.process_extensions import _RegisteredPlan

    result = []
    for plan, operation in zip(plans, operations, strict=True):
        if plan.family == "nucleation" or (
            type(plan) is _RegisteredPlan
            and "environment.saturation_ratio" in plan.configuration.reads
        ):
            result.append(refresh_operation)
        result.append(operation)
        if type(plan) is _RegisteredPlan and set(plan.configuration.writes) & {
            "gas.concentration",
            "environment.temperature",
        }:
            result.append(refresh_operation)
    result.append(refresh_operation)
    refresh()
    return tuple(result)


def validate_extension_freshness(plans):
    """Reject stale saturation consumers if no refresh provider is available."""
    from particula.execution.process_extensions import _RegisteredPlan

    if any(plan.family == "condensation" for plan in plans):
        return
    invalidates = {"gas.concentration", "environment.temperature"}
    # The sequence repeats: a writer at the end affects a reader next timestep.
    dirty = any(
        type(plan) is _RegisteredPlan
        and bool(set(plan.configuration.writes) & invalidates)
        for plan in plans
    )
    custom_reader = any(
        type(plan) is _RegisteredPlan
        and "environment.saturation_ratio" in plan.configuration.reads
        for plan in plans
    )
    native_reader = any(
        plan.family == "nucleation"
        and dict(plan.values["config"])["saturation_lower"] is not None
        for plan in plans
    )
    native_writer = any(
        plan.family in {"dilution", "nucleation"} for plan in plans
    )
    if (dirty and (custom_reader or native_reader)) or (
        native_writer and custom_reader
    ):
        raise ValueError(
            "saturation consumers require a thermodynamic refresh provider"
        )
