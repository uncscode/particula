"""Prepared device-only validation of declared custom process destinations."""

from math import ceil, floor
from typing import no_type_check

import numpy as np
import warp as wp


@no_type_check
@wp.kernel
def _float_values(
    values: wp.array(dtype=wp.float64),
    lower: wp.float64,
    upper: wp.float64,
    strict_lower: int,
    failure: wp.array(dtype=wp.int32),
):
    """Latch finite/bounds failures before dependent kernels can run."""
    if failure[0] != 0:
        return
    value = values[wp.tid()]
    if (
        not wp.isfinite(value)
        or value < lower
        or value > upper
        or (strict_lower != 0 and value == lower)
    ):
        wp.atomic_max(failure, 0, 90)


@no_type_check
@wp.kernel
def _int32_values(
    values: wp.array(dtype=wp.int32),
    lower: wp.int64,
    upper: wp.int64,
    failure: wp.array(dtype=wp.int32),
):
    """Compare integer bounds without losing precision through float casts."""
    if failure[0] != 0:
        return
    value = wp.int64(values[wp.tid()])
    if value < lower or value > upper:
        wp.atomic_max(failure, 0, 90)


@no_type_check
@wp.kernel
def _int64_values(
    values: wp.array(dtype=wp.int64),
    lower: wp.int64,
    upper: wp.int64,
    failure: wp.array(dtype=wp.int32),
):
    """Retain all 64 integer bits in field-bound validation."""
    if failure[0] != 0:
        return
    value = values[wp.tid()]
    if value < lower or value > upper:
        wp.atomic_max(failure, 0, 90)


def _float_launch(array, lower, upper, strict, failure):
    """Retain flattened views and typed launch inputs during setup."""
    flat = array.flatten()
    return (
        _float_values,
        flat.size,
        (flat, wp.float64(lower), wp.float64(upper), strict, failure),
    )


def prepare_validation(configuration, arrays, state, owner):  # noqa: C901
    """Prepare bounds once; validation enqueues no allocation or readback."""
    launches = []
    descriptors = {spec.name: spec for spec in owner._extension_fields.specs}
    maximum = np.finfo(np.float64).max
    failure = owner._device_failure
    for name in configuration.writes:
        array = arrays[name]
        spec = descriptors.get(name)
        if spec is not None and spec.dtype != "float64":
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
            flat = array.flatten()
            kernel = _int32_values if spec.dtype == "int32" else _int64_values
            launches.append(
                (
                    kernel,
                    flat.size,
                    (flat, wp.int64(lower), wp.int64(upper), failure),
                )
            )
            continue
        lower, upper, strict = -maximum, maximum, 0
        if spec is not None:
            lower = spec.minimum if spec.minimum is not None else lower
            upper = spec.maximum if spec.maximum is not None else upper
        elif name in (
            "particles.masses",
            "particles.concentration",
            "gas.concentration",
        ):
            lower = 0.0
        elif name in (
            "particles.volume",
            "environment.temperature",
            "environment.pressure",
        ):
            lower, strict = 0.0, 1
            if (
                name == "particles.volume"
                and owner._session.metadata.distribution_type
                != "particle_resolved"
            ):
                lower, upper, strict = 1.0, 1.0, 0
        launches.append(_float_launch(array, lower, upper, strict, failure))
    for spec in configuration.state:
        if spec.dtype == "float64":
            launches.append(
                _float_launch(state[spec.name], -maximum, maximum, 0, failure)
            )
    launches = tuple(launches)

    def validate():
        for kernel, size, inputs in launches:
            if size:
                wp.launch(
                    kernel, dim=size, inputs=inputs, device=owner.device.native
                )

    resources = tuple(value for _, _, inputs in launches for value in inputs)
    return validate, resources
