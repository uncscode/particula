"""Discover active and free fixed particle slots on a Warp device.

This concrete-module-only helper mirrors the CPU slot classification without
transferring particle fields to the host. It writes caller-owned diagnostic
sidecars only after the complete particle schema and state preflight succeeds.
"""

# mypy: disable-error-code="valid-type, misc"

from __future__ import annotations

from typing import Any

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


@wp.kernel
def _classify_slots(
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    categories: wp.array2d(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Classify one particle slot and record an invalid-state flag."""
    box, particle = wp.tid()
    species_count = masses.shape[2]
    mass_valid = wp.bool(True)
    mass_zero = wp.bool(True)
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


@wp.kernel
def _write_diagnostics(
    categories: wp.array2d(dtype=wp.int32),
    free_indices: wp.array2d(dtype=wp.int32),
    active_counts: wp.array(dtype=wp.int32),
    free_counts: wp.array(dtype=wp.int32),
) -> None:
    """Write ascending free indices and counts from completed categories."""
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


@wp.kernel
def _write_empty_diagnostics(
    active_counts: wp.array(dtype=wp.int32),
    free_counts: wp.array(dtype=wp.int32),
) -> None:
    """Overwrite counts for a box whose particle capacity is zero."""
    box = wp.tid()
    active_counts[box] = 0
    free_counts[box] = 0


def _get_required_field(particles: Any, name: str) -> Any:
    """Return a particle classification field or raise a stable error."""
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
    """Validate one Warp diagnostic or particle array schema."""
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
    """Validate classification fields and derive mass-authoritative schema."""
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


def get_slot_diagnostics_gpu(
    particles: Any,
    free_indices: Any,
    active_counts: Any,
    free_counts: Any,
) -> tuple[Any, Any, Any]:
    """Write read-only slot diagnostics into supplied Warp ``int32`` sidecars.

    Import this concrete direct-Warp primitive from
    ``particula.gpu.kernels.slot_management``. A slot is active iff its
    concentration is finite and positive, all mass lanes are finite and
    nonnegative, its float64 total mass is finite and positive, and its charge
    is finite. A slot is free iff concentration, all masses, and charge are
    exactly zero. Any other state raises ``ValueError`` before output writes.

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
        Particle fields are read only. Valid calls overwrite all diagnostics;
        free indices are ascending and unused positions are ``-1``.
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
