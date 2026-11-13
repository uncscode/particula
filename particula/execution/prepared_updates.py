"""Explicit host forcing and thermodynamic freshness for prepared sessions."""

from __future__ import annotations

from typing import Any

import numpy as np


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


def compose_operations(owner, plans, operations):
    """Refresh saturation before source consumption and after complete steps."""
    if "condensation" not in owner._bindings:
        return operations
    from particula.execution.prepared_guards import refresh_saturation
    from particula.gpu.kernels.thermodynamics import (
        _refresh_vapor_pressure_kernel_guarded,
    )

    prepared = owner._bindings["condensation"].prepared
    session, wp = owner._session, owner._wp
    saturation = session.environment.saturation_ratio
    failure = owner._device_failure
    dim = session.gas.concentration.shape
    vapor_inputs = [
        prepared.thermodynamics_modes,
        prepared.thermodynamics_parameters,
        prepared.refresh_temperature,
        prepared.gas_vapor_pressure,
        failure,
    ]
    ratio_inputs = [
        prepared.gas_concentration,
        prepared.gas_molar_mass,
        prepared.gas_vapor_pressure,
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

    owner._refresh = refresh
    result = []
    for plan, operation in zip(plans, operations, strict=True):
        if plan.family == "nucleation":
            result.append(refresh)
        result.append(operation)
    result.append(refresh)
    refresh()
    return tuple(result)
