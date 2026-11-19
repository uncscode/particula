"""Storage reservations and device-resident undo for adaptive prepared steps."""

from dataclasses import fields, is_dataclass
from types import FunctionType, MethodType

from particula.gpu.kernels._coagulation_subcycling import (
    CoagulationSubcyclingConfig,
    adaptive_capacity,
    check_memory,
    interval_bytes,
    workspace_bytes,
)


def reserve_prepared(aerosol, plans):
    """Check host shape bounds before upload, including dependent processes.

    The direct sampler and primary layouts have explicit dimensions. Composed
    adapters reserve space for their optional work arrays even when individual
    branches alias them. The retained-array inventory is checked again at bind.
    """
    coag = next((p for p in plans if p.family == "coagulation"), None)
    if coag is None or "subcycling" not in coag.values:
        return
    values = coag.values
    policy = CoagulationSubcyclingConfig(*values["subcycling"][1:])
    b, n, s = aerosol.particles.masses.shape
    g = len(aerosol.gas.name)
    capacity = adaptive_capacity(b, n, values["max_collision_buffer_bytes"])
    # Gas includes derived vapor pressure, a per-box int32 partitioning mask,
    # and the environment saturation ratio.
    primary = 8 * (b * n * (s + 2) + s + 3 * b + g) + 28 * b * g
    registered = sum(a.nbytes for _, a in aerosol.fields.items()) + sum(
        a.nbytes for _, a in aerosol.fields.transaction_arrays()
    )
    work = workspace_bytes(
        b,
        n,
        capacity,
        values.get("relative_permittivity"),
        "hamaker_constant" in values,
    )
    if values.get("atmospheric"):
        from particula.dynamics.ions.attachment_table import (
            centered_attachment_tables,
        )

        work += centered_attachment_tables().nbytes + b * (20 * n + 8)
    # Optional field validation needs a before-count image. Resampling may
    # also keep a pre-event image of each field.
    dependent = registered + (8 * b * n if registered else 0)
    for plan in plans:
        if plan.family == "condensation":
            # Full material/gas trial, final, energy and mapping arrays.
            dependent += (
                16 * (primary + 8 * b * n * g)
                + 8 * len(plan.values.get("bat_table", ()))
                + 512
            )
        elif plan.family == "nucleation":
            # Includes resampling, transport, status and commit arrays.
            dependent += 64 * (primary + 8 * b * n * g) + 2048
        elif plan.family != "coagulation":
            dependent += 4 * primary + 128 * b
    controller = interval_bytes(b, n, policy.max_subcycles)
    diagnostics = 8 * b * ((n + 255) // 256 + 1) * (s + 3)
    categories = dict(
        primary=primary + registered,
        sampler=work,
        other_processes=dependent,
        controller=controller,
        diagnostics=diagnostics,
    )
    categories["rollback"] = sum(categories.values())
    check_memory(policy, categories)


def _bound_children(item, wp):
    """Visit the immutable carriers and closures used by native adapters."""
    if wp.types.is_struct(item):
        return (getattr(item, name) for name in item._cls.vars)
    if isinstance(item, (list, tuple)):
        return item
    if isinstance(item, dict):
        return item.values()
    if is_dataclass(item) and not isinstance(item, type):
        return (getattr(item, field.name) for field in fields(item))
    if isinstance(item, FunctionType):
        return (cell.cell_contents for cell in item.__closure__ or ())
    if isinstance(item, MethodType):
        return (item.__self__,)
    return ()


def rollback_buffers(owner, call):
    """Retain one device backup per unique bound array."""
    wp = owner._wp
    arrays = {}
    visited = {id(owner)}
    pending = [
        owner._session.particles,
        owner._session.gas,
        owner._session.environment,
        owner._extension_fields,
        owner._operations,
        owner._bindings,
        owner._auxiliary,
        owner._clock,
        owner._device_failure,
        owner._diagnostics,
        owner._observers,
    ]
    pending.extend(a for _, _, a, _ in owner._registry._enumerate_resources())
    pending.extend(
        a for _, _, a in owner._registry._enumerate_published_rng_streams()
    )
    while pending:
        item = pending.pop()
        if id(item) in visited:
            continue
        visited.add(id(item))
        if isinstance(item, wp.array):
            if item.ptr:
                arrays[item.ptr] = item
        else:
            pending.extend(_bound_children(item, wp))
    live_bytes = sum(a.capacity for a in arrays.values())
    owner._adaptive_allocation = {"live": live_bytes, "rollback": live_bytes}
    check_memory(call.adaptive.config, owner._adaptive_allocation)
    return tuple((a, wp.empty_like(a)) for a in arrays.values())
