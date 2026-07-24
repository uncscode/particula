"""Classify and activate fixed-capacity particle slots on a Warp device.

This module provides concrete-module-only P3 diagnostics and the
package-exported P4 ``activate_slots_gpu`` boundary. Both APIs read only
particle masses, concentration, and charge; density and volume are
intentionally unobserved. P4 validates all metadata, ownership, current slot
state, counts, capacity, and selected request prefixes before launching its
writer, then writes caller-owned activation and diagnostic sidecars. Caller-
owned simulation arrays are never implicitly transferred; validation may
synchronize and read back private scalar status buffers.
"""

# mypy: disable-error-code="valid-type, misc"

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import warp as wp
except ImportError as exc:  # pragma: no cover - handled by test guards
    raise ImportError(
        "Warp is required for GPU slot-management helpers. "
        "Install with: pip install warp-lang"
    ) from exc

from particula.gpu.kernels.environment import _is_warp_array_like

_CATEGORY_ACTIVE = wp.int32(1)
_CATEGORY_FREE = wp.int32(2)
_CATEGORY_INVALID = wp.int32(3)
_INVALID_SLOT_MESSAGE = "Invalid particle slot state."
_INVALID_COUNT_MESSAGE = "requested_counts exceeds request capacity."
_CAPACITY_MESSAGE = "Requested activation exceeds free slot capacity."
_INVALID_REQUEST_MESSAGE = "Requested activation record is invalid."
_ALIAS_MESSAGE = "Activation arrays must not share storage."


@wp.kernel  # pragma: no cover
def _classify_slots(
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    categories: wp.array2d(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Classify one slot and atomically flag an invalid slot state.

    The kernel writes a category for every slot but does not write any
    caller-provided diagnostic sidecar.
    """
    box, particle = wp.tid()
    species_count = masses.shape[2]
    mass_valid = bool(True)
    mass_zero = bool(True)
    total_mass = wp.float64(0.0)
    for species in range(species_count):
        mass = masses[box, particle, species]
        if not wp.isfinite(mass) or mass < 0.0:
            mass_valid = False
        if mass != 0.0:
            mass_zero = False
        total_mass = total_mass + mass

    slot_concentration = concentration[box, particle]
    slot_charge = charge[box, particle]
    active = (
        wp.isfinite(slot_concentration)
        and slot_concentration > 0.0
        and mass_valid
        and wp.isfinite(total_mass)
        and total_mass > 0.0
        and wp.isfinite(slot_charge)
    )
    free = (
        slot_concentration == 0.0
        and mass_zero
        and slot_charge == 0.0
        and wp.isfinite(slot_charge)
    )
    if active:
        categories[box, particle] = _CATEGORY_ACTIVE
    elif free:
        categories[box, particle] = _CATEGORY_FREE
    else:
        categories[box, particle] = _CATEGORY_INVALID
        wp.atomic_max(invalid, 0, 1)


@wp.kernel  # pragma: no cover
def _write_diagnostics(
    categories: wp.array2d(dtype=wp.int32),
    free_indices: wp.array2d(dtype=wp.int32),
    active_counts: wp.array(dtype=wp.int32),
    free_counts: wp.array(dtype=wp.int32),
) -> None:
    """Write per-box counts and ascending free indices from slot categories.

    Unused free-index entries are overwritten with ``-1``.
    """
    box = wp.tid()
    particle_count = categories.shape[1]
    active_count = wp.int32(0)
    free_count = wp.int32(0)
    for particle in range(particle_count):
        category = categories[box, particle]
        if category == _CATEGORY_ACTIVE:
            active_count = active_count + 1
        elif category == _CATEGORY_FREE:
            free_indices[box, free_count] = particle
            free_count = free_count + 1
    for particle in range(free_count, particle_count):
        free_indices[box, particle] = -1
    active_counts[box] = active_count
    free_counts[box] = free_count


@wp.kernel  # pragma: no cover
def _write_empty_diagnostics(
    active_counts: wp.array(dtype=wp.int32),
    free_counts: wp.array(dtype=wp.int32),
) -> None:
    """Set both diagnostic counts to zero for a zero-capacity box."""
    box = wp.tid()
    active_counts[box] = 0
    free_counts[box] = 0


@wp.kernel  # pragma: no cover
def _build_free_workspace(
    categories: wp.array2d(dtype=wp.int32),
    workspace: wp.array2d(dtype=wp.int32),
    free_counts: wp.array(dtype=wp.int32),
) -> None:
    """Compact ascending free slots into private fixed-shape workspace."""
    box = wp.tid()
    free_count = wp.int32(0)
    for particle in range(categories.shape[1]):
        if categories[box, particle] == _CATEGORY_FREE:
            workspace[box, free_count] = particle
            free_count = free_count + 1
    free_counts[box] = free_count


@wp.kernel  # pragma: no cover
def _validate_activation_preflight(
    categories: wp.array2d(dtype=wp.int32),
    free_counts: wp.array(dtype=wp.int32),
    request_masses: wp.array3d(dtype=wp.float64),
    request_concentration: wp.array2d(dtype=wp.float64),
    request_charge: wp.array2d(dtype=wp.float64),
    requested_counts: wp.array(dtype=wp.int32),
    invalid_count: wp.array(dtype=wp.int32),
    insufficient_capacity: wp.array(dtype=wp.int32),
    invalid_request: wp.array(dtype=wp.int32),
) -> None:
    """Validate state, counts, capacity, and selected request records."""
    box, request = wp.tid()
    count = requested_counts[box]
    if request == 0:
        if count < 0 or count > request_masses.shape[1]:
            wp.atomic_max(invalid_count, 0, 1)
        elif count > free_counts[box]:
            wp.atomic_max(insufficient_capacity, 0, 1)
    if request >= count or count < 0 or count > request_masses.shape[1]:
        return
    total_mass = wp.float64(0.0)
    valid = (
        wp.isfinite(request_concentration[box, request])
        and (request_concentration[box, request] > 0.0)
        and wp.isfinite(request_charge[box, request])
    )
    for species in range(request_masses.shape[2]):
        mass = request_masses[box, request, species]
        if not wp.isfinite(mass) or mass < 0.0:
            valid = False
        total_mass = total_mass + mass
    if not valid or not wp.isfinite(total_mass) or total_mass <= 0.0:
        wp.atomic_max(invalid_request, 0, 1)


@wp.kernel  # pragma: no cover
def _activate_requests(
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    request_masses: wp.array3d(dtype=wp.float64),
    request_concentration: wp.array2d(dtype=wp.float64),
    request_charge: wp.array2d(dtype=wp.float64),
    requested_counts: wp.array(dtype=wp.int32),
    free_workspace: wp.array2d(dtype=wp.int32),
    activated_counts: wp.array(dtype=wp.int32),
) -> None:
    """Copy validated request prefixes to their precomputed free slots."""
    box, request = wp.tid()
    if request == 0:
        activated_counts[box] = requested_counts[box]
    if request >= requested_counts[box]:
        return
    particle = free_workspace[box, request]
    for species in range(masses.shape[2]):
        masses[box, particle, species] = request_masses[box, request, species]
    concentration[box, particle] = request_concentration[box, request]
    charge[box, particle] = request_charge[box, request]


@wp.kernel  # pragma: no cover
def _write_activated_counts(
    requested_counts: wp.array(dtype=wp.int32),
    activated_counts: wp.array(dtype=wp.int32),
) -> None:
    """Copy validated requested counts into caller-owned output storage."""
    box = wp.tid()
    activated_counts[box] = requested_counts[box]


@wp.kernel  # pragma: no cover
def _validate_empty_activation_extent(
    requested_counts: wp.array(dtype=wp.int32),
    invalid_count: wp.array(dtype=wp.int32),
    insufficient_capacity: wp.array(dtype=wp.int32),
    request_capacity: wp.int32,
) -> None:
    """Reject nonzero counts when no request or destination extent exists."""
    box = wp.tid()
    count = requested_counts[box]
    if count < 0 or count > request_capacity:
        wp.atomic_max(invalid_count, 0, 1)
    elif count != 0:
        wp.atomic_max(insufficient_capacity, 0, 1)


def _get_required_field(particles: Any, name: str) -> Any:
    """Return a required classification field or raise a stable error.

    Field access is isolated so malformed particle containers consistently
    produce ``ValueError`` rather than leaking attribute-access exceptions.
    """
    try:
        return getattr(particles, name)
    except Exception as exc:
        raise ValueError(f"particles.{name} must be a Warp array.") from exc


def _validate_array_schema(
    values: Any,
    name: str,
    dtype: Any,
    ndim: int,
    shape: tuple[int, ...] | None = None,
    device: Any | None = None,
) -> Any:
    """Validate one Warp particle-field or diagnostic-sidecar schema.

    This performs metadata validation only. It neither scans field values nor
    mutates the supplied array.
    """
    try:
        is_warp_array = _is_warp_array_like(values)
    except Exception:
        is_warp_array = False
    if not is_warp_array:
        raise ValueError(f"{name} must be a Warp array.")
    try:
        if values.dtype != dtype:
            raise ValueError(f"{name} must use the required Warp dtype.")
        if values.ndim != ndim:
            raise ValueError(f"{name} must have rank {ndim}.")
        if shape is not None and values.shape != shape:
            raise ValueError(f"{name} shape must match particle masses.")
        if device is not None and str(values.device) != str(device):
            raise ValueError(f"{name} device must match particle device.")
    except Exception as exc:
        raise ValueError(f"{name} has invalid Warp array metadata.") from exc
    return values


def _validate_particles(particles: Any) -> tuple[Any, Any, Any, int, int, Any]:
    """Validate classification fields and derive the mass-authoritative schema.

    Only masses, concentration, and charge participate in classification;
    density and volume remain unobserved.
    """
    masses = _validate_array_schema(
        _get_required_field(particles, "masses"),
        "particles.masses",
        wp.float64,
        3,
    )
    try:
        n_boxes, n_particles, _ = masses.shape
        device = masses.device
    except Exception as exc:
        raise ValueError(
            "particles.masses has invalid Warp array metadata."
        ) from exc
    concentration = _validate_array_schema(
        _get_required_field(particles, "concentration"),
        "particles.concentration",
        wp.float64,
        2,
        (n_boxes, n_particles),
        device,
    )
    charge = _validate_array_schema(
        _get_required_field(particles, "charge"),
        "particles.charge",
        wp.float64,
        2,
        (n_boxes, n_particles),
        device,
    )
    return masses, concentration, charge, n_boxes, n_particles, device


def _validate_activation_inputs(
    particles: Any,
    request_masses: Any,
    request_concentration: Any,
    request_charge: Any,
    requested_counts: Any,
    activated_counts: Any,
    free_indices: Any,
    active_counts: Any,
    free_counts: Any,
) -> tuple[
    Any,
    Any,
    Any,
    Any,
    Any,
    Any,
    Any,
    Any,
    Any,
    Any,
    Any,
    int,
    int,
    int,
    Any,
]:
    """Validate schemas and storage ownership for activation inputs."""
    masses, concentration, charge, n_boxes, n_particles, device = (
        _validate_particles(particles)
    )
    n_species = masses.shape[2]
    request_masses = _validate_array_schema(
        request_masses, "request_masses", wp.float64, 3, device=device
    )
    try:
        request_capacity = request_masses.shape[1]
        if (
            request_masses.shape[0] != n_boxes
            or request_masses.shape[2] != n_species
        ):
            raise ValueError
    except Exception as exc:
        raise ValueError(
            "request_masses shape must match particle masses."
        ) from exc
    request_shape = (n_boxes, request_capacity)
    request_concentration = _validate_array_schema(
        request_concentration,
        "request_concentration",
        wp.float64,
        2,
        request_shape,
        device,
    )
    request_charge = _validate_array_schema(
        request_charge, "request_charge", wp.float64, 2, request_shape, device
    )
    requested_counts = _validate_array_schema(
        requested_counts, "requested_counts", wp.int32, 1, (n_boxes,), device
    )
    activated_counts = _validate_array_schema(
        activated_counts, "activated_counts", wp.int32, 1, (n_boxes,), device
    )
    free_indices = _validate_array_schema(
        free_indices,
        "free_indices",
        wp.int32,
        2,
        (n_boxes, n_particles),
        device,
    )
    active_counts = _validate_array_schema(
        active_counts, "active_counts", wp.int32, 1, (n_boxes,), device
    )
    free_counts = _validate_array_schema(
        free_counts, "free_counts", wp.int32, 1, (n_boxes,), device
    )
    _validate_activation_ownership(
        (
            request_masses,
            request_concentration,
            request_charge,
            requested_counts,
        ),
        (
            masses,
            concentration,
            charge,
            activated_counts,
            free_indices,
            active_counts,
            free_counts,
        ),
    )
    return (
        masses,
        concentration,
        charge,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
        activated_counts,
        free_indices,
        active_counts,
        free_counts,
        n_boxes,
        n_particles,
        request_capacity,
        device,
    )


def _validate_zero_capacity_activation(
    requested_counts: Any,
    request_capacity: int,
    active_counts: Any,
    free_counts: Any,
    device: Any,
) -> tuple[Any, Any]:
    """Validate count and capacity constraints when no particle slots exist."""
    invalid_count = wp.zeros(1, dtype=wp.int32, device=device)
    insufficient_capacity = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _validate_empty_activation_extent,
        dim=int(requested_counts.shape[0]),
        inputs=[
            requested_counts,
            invalid_count,
            insufficient_capacity,
            request_capacity,
        ],
        device=device,
    )
    if int(invalid_count.numpy()[0]):
        raise ValueError(_INVALID_COUNT_MESSAGE)
    if int(insufficient_capacity.numpy()[0]):
        raise ValueError(_CAPACITY_MESSAGE)
    wp.launch(
        _write_empty_diagnostics,
        dim=int(requested_counts.shape[0]),
        inputs=[active_counts, free_counts],
        device=device,
    )
    return active_counts, free_counts


def _activate_nonempty_requests(
    masses: Any,
    concentration: Any,
    charge: Any,
    request_masses: Any,
    request_concentration: Any,
    request_charge: Any,
    requested_counts: Any,
    activated_counts: Any,
    free_indices: Any,
    active_counts: Any,
    free_counts: Any,
    n_boxes: int,
    n_particles: int,
    request_capacity: int,
    device: Any,
) -> tuple[Any, Any, Any, Any]:
    """Validate and apply activation requests for nonempty slot capacity."""
    categories = wp.empty((n_boxes, n_particles), dtype=wp.int32, device=device)
    free_workspace = wp.empty(
        (n_boxes, n_particles), dtype=wp.int32, device=device
    )
    workspace_counts = wp.empty(n_boxes, dtype=wp.int32, device=device)
    invalid_state = wp.zeros(1, dtype=wp.int32, device=device)
    invalid_count = wp.zeros(1, dtype=wp.int32, device=device)
    insufficient_capacity = wp.zeros(1, dtype=wp.int32, device=device)
    invalid_request = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _classify_slots,
        dim=(n_boxes, n_particles),
        inputs=[masses, concentration, charge, categories, invalid_state],
        device=device,
    )
    wp.launch(
        _build_free_workspace,
        dim=n_boxes,
        inputs=[categories, free_workspace, workspace_counts],
        device=device,
    )
    if request_capacity:
        wp.launch(
            _validate_activation_preflight,
            dim=(n_boxes, request_capacity),
            inputs=[
                categories,
                workspace_counts,
                request_masses,
                request_concentration,
                request_charge,
                requested_counts,
                invalid_count,
                insufficient_capacity,
                invalid_request,
            ],
            device=device,
        )
    else:
        # A zero request extent still needs count/capacity validation.
        wp.launch(
            _validate_empty_activation_extent,
            dim=n_boxes,
            inputs=[
                requested_counts,
                invalid_count,
                insufficient_capacity,
                request_capacity,
            ],
            device=device,
        )
    if int(invalid_state.numpy()[0]):
        raise ValueError(_INVALID_SLOT_MESSAGE)
    if int(invalid_count.numpy()[0]):
        raise ValueError(_INVALID_COUNT_MESSAGE)
    if int(insufficient_capacity.numpy()[0]):
        raise ValueError(_CAPACITY_MESSAGE)
    if int(invalid_request.numpy()[0]):
        raise ValueError(_INVALID_REQUEST_MESSAGE)

    if request_capacity:
        wp.launch(
            _activate_requests,
            dim=(n_boxes, request_capacity),
            inputs=[
                masses,
                concentration,
                charge,
                request_masses,
                request_concentration,
                request_charge,
                requested_counts,
                free_workspace,
                activated_counts,
            ],
            device=device,
        )
    else:
        wp.launch(
            _write_activated_counts,
            dim=n_boxes,
            inputs=[requested_counts, activated_counts],
            device=device,
        )
    wp.launch(
        _classify_slots,
        dim=(n_boxes, n_particles),
        inputs=[masses, concentration, charge, categories, invalid_state],
        device=device,
    )
    wp.launch(
        _write_diagnostics,
        dim=n_boxes,
        inputs=[categories, free_indices, active_counts, free_counts],
        device=device,
    )
    return activated_counts, free_indices, active_counts, free_counts


def _warp_array_memory_range(array: Any) -> tuple[int, int] | None:
    """Return a supported contiguous Warp array's byte range.

    Views are rejected because a byte range cannot distinguish their gaps from
    owned storage. Metadata validation has already established the required
    dtype before this helper is called.
    """
    itemsize = {
        wp.float64: np.dtype(np.float64).itemsize,
        wp.int32: np.dtype(np.int32).itemsize,
    }.get(array.dtype)
    if itemsize is None:
        raise ValueError("overlap validation does not support this Warp dtype")
    expected: list[int] = []
    stride = itemsize
    for dimension in reversed(array.shape):
        expected.insert(0, stride)
        stride *= dimension
    strides = getattr(array, "strides", None)
    if strides is not None and tuple(strides) != tuple(expected):
        raise ValueError(
            "overlap-checked Warp arrays must be contiguous, non-view arrays"
        )
    size = int(np.prod(array.shape, dtype=np.int64))
    if size == 0:
        return None
    start = int(array.ptr)
    return start, start + size * itemsize


def _validate_activation_ownership(
    read_arrays: tuple[Any, ...], writable_arrays: tuple[Any, ...]
) -> None:
    """Reject direct and partial overlap with mutable activation targets."""
    all_writable = tuple(writable_arrays)
    for index, first in enumerate(all_writable):
        first_range = _warp_array_memory_range(first)
        for second in all_writable[index + 1 :]:
            second_range = _warp_array_memory_range(second)
            if first is second or (
                first_range is not None
                and second_range is not None
                and first_range[0] < second_range[1]
                and second_range[0] < first_range[1]
            ):
                raise ValueError(_ALIAS_MESSAGE)
    for read_array in read_arrays:
        read_range = _warp_array_memory_range(read_array)
        for writable in all_writable:
            write_range = _warp_array_memory_range(writable)
            if read_array is writable or (
                read_range is not None
                and write_range is not None
                and read_range[0] < write_range[1]
                and write_range[0] < read_range[1]
            ):
                raise ValueError(_ALIAS_MESSAGE)


def get_slot_diagnostics_gpu(
    particles: Any,
    free_indices: Any,
    active_counts: Any,
    free_counts: Any,
) -> tuple[Any, Any, Any]:
    """Write read-only slot diagnostics into supplied Warp ``int32`` sidecars.

    Import this concrete direct-Warp primitive from
    ``particula.gpu.kernels.slot_management``; it is not package-exported. A
    slot is active iff concentration is finite and positive, all mass lanes
    are finite and nonnegative, float64 total mass is finite and positive, and
    charge is finite. A slot is free iff concentration, all masses, and charge
    are exactly zero. Any other state raises ``ValueError`` before diagnostic
    writes. The primitive reads only those classification fields, leaving all
    particle storage, including density and volume, unchanged.

    Args:
        particles: Container with same-device float64 ``masses`` shaped
            ``(n_boxes, n_particles, n_species)``, plus concentration and
            charge fields shaped ``(n_boxes, n_particles)``.
        free_indices: Same-device int32 output shaped
            ``(n_boxes, n_particles)``.
        active_counts: Same-device int32 output shaped ``(n_boxes,)``.
        free_counts: Same-device int32 output shaped ``(n_boxes,)``.

    Returns:
        The exact supplied ``(free_indices, active_counts, free_counts)``.

    Raises:
        ValueError: If a schema is malformed or any slot is neither active nor
            free.

    Notes:
        Valid calls overwrite every diagnostic entry. Free indices are
        ascending and unused positions are ``-1``. Schema and state rejection
        occurs before caller-owned diagnostics are written. Callers own device
        placement and synchronization before reading device-resident diagnostics
        on the host.
    """
    masses, concentration, charge, n_boxes, n_particles, device = (
        _validate_particles(particles)
    )
    _validate_array_schema(
        free_indices,
        "free_indices",
        wp.int32,
        2,
        (n_boxes, n_particles),
        device,
    )
    _validate_array_schema(
        active_counts,
        "active_counts",
        wp.int32,
        1,
        (n_boxes,),
        device,
    )
    _validate_array_schema(
        free_counts,
        "free_counts",
        wp.int32,
        1,
        (n_boxes,),
        device,
    )
    _validate_activation_ownership(
        (masses, concentration, charge),
        (free_indices, active_counts, free_counts),
    )

    if n_boxes == 0:
        return free_indices, active_counts, free_counts

    if n_particles == 0:
        wp.launch(
            _write_empty_diagnostics,
            dim=n_boxes,
            inputs=[active_counts, free_counts],
            device=device,
        )
        return free_indices, active_counts, free_counts

    categories = wp.empty((n_boxes, n_particles), dtype=wp.int32, device=device)
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _classify_slots,
        dim=(n_boxes, n_particles),
        inputs=[masses, concentration, charge, categories, invalid],
        device=device,
    )
    if invalid.numpy()[0] != 0:
        raise ValueError(_INVALID_SLOT_MESSAGE)
    wp.launch(
        _write_diagnostics,
        dim=n_boxes,
        inputs=[categories, free_indices, active_counts, free_counts],
        device=device,
    )
    return free_indices, active_counts, free_counts


def activate_slots_gpu(
    particles: Any,
    request_masses: Any,
    request_concentration: Any,
    request_charge: Any,
    requested_counts: Any,
    activated_counts: Any,
    free_indices: Any,
    active_counts: Any,
    free_counts: Any,
) -> tuple[Any, Any, Any, Any]:
    """Copy selected Warp request prefixes into ascending free slots.

    For each box, request rank ``r`` is copied into the ``r``-th ascending
    free fixed-capacity slot. Complete preflight validates metadata, storage
    ownership, existing slot state, counts, capacity, and only selected
    request-prefix records before the mutation writer launches. Valid calls
    write the supplied sidecars; ``activated_counts`` equals
    ``requested_counts``, and diagnostics describe the post-activation state.

    This direct boundary never implicitly transfers caller-owned simulation
    arrays and accesses only particle masses, concentration, and charge.
    Validation may synchronize and read back private scalar status buffers.
    Rejected calls leave accessible caller-owned inputs and outputs unchanged.
    Rollback is not guaranteed after the mutation writer launches.

    Args:
        particles: Same-device fixed-capacity storage with float64 ``masses``
            shaped ``(n_boxes, n_particles, n_species)`` and ``concentration``
            and ``charge`` shaped ``(n_boxes, n_particles)``.
        request_masses: Same-device float64 request masses shaped
            ``(n_boxes, request_capacity, n_species)``.
        request_concentration: Same-device float64 request concentrations
            shaped ``(n_boxes, request_capacity)``.
        request_charge: Same-device float64 request charges shaped
            ``(n_boxes, request_capacity)``.
        requested_counts: Same-device int32 selected-prefix lengths shaped
            ``(n_boxes,)``.
        activated_counts: Caller-owned same-device int32 output shaped
            ``(n_boxes,)``.
        free_indices: Caller-owned same-device int32 output shaped
            ``(n_boxes, n_particles)``; free slots are ascending and its unused
            entries are ``-1``.
        active_counts: Caller-owned same-device int32 output shaped
            ``(n_boxes,)``.
        free_counts: Caller-owned same-device int32 output shaped
            ``(n_boxes,)``.

    Returns:
        The exact supplied ``(activated_counts, free_indices, active_counts,
        free_counts)`` sidecars.

    Raises:
        ValueError: If schemas, device or storage ownership, slot state,
            selected request records, counts, or capacity are invalid.
    """
    (
        masses,
        concentration,
        charge,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
        activated_counts,
        free_indices,
        active_counts,
        free_counts,
        n_boxes,
        n_particles,
        request_capacity,
        device,
    ) = _validate_activation_inputs(
        particles,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
        activated_counts,
        free_indices,
        active_counts,
        free_counts,
    )
    if n_boxes == 0:
        return activated_counts, free_indices, active_counts, free_counts

    if n_particles == 0:
        _validate_zero_capacity_activation(
            requested_counts,
            request_capacity,
            active_counts,
            free_counts,
            device,
        )
        wp.launch(
            _write_activated_counts,
            dim=n_boxes,
            inputs=[requested_counts, activated_counts],
            device=device,
        )
        return activated_counts, free_indices, active_counts, free_counts

    return _activate_nonempty_requests(
        masses,
        concentration,
        charge,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
        activated_counts,
        free_indices,
        active_counts,
        free_counts,
        n_boxes,
        n_particles,
        request_capacity,
        device,
    )
