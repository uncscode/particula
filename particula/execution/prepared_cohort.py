"""Prepare and checkpoint original-seed lineage tracking on resident arrays."""

import numpy as np


def plan_cohort(process, aerosol):
    """Freeze the initial labeling rule and first-passage threshold."""
    from particula.execution.prepared_processes import _ProcessPlan

    process._validate_batch(aerosol)
    return _ProcessPlan(
        "cohort",
        tuple(
            (name, getattr(process, name))
            for name in (
                "seed_species",
                "background_species",
                "target_diameter",
            )
        ),
    )


def compile_cohort(_plan, _owner):
    """The observer is attached around physical operations after compilation."""
    return lambda: None


def validate_cohort_plans(plans):
    """Reject untracked removal, resampling and marker-material sources."""
    cohort = next((p for p in plans if p.family == "cohort"), None)
    if cohort is None:
        return
    supported = {
        "cohort",
        "coagulation",
        "condensation",
        "ion_source",
        "ion_cleanup",
    }
    if any(p.family not in supported for p in plans):
        raise ValueError(
            "seed cohorts do not support removal or resampling processes"
        )
    markers = {
        cohort.values["seed_species"],
        cohort.values["background_species"],
    }
    for plan in plans:
        if plan.family in ("ion_source", "ion_cleanup"):
            if markers & {
                plan.values["positive_species"],
                plan.values["negative_species"],
            }:
                raise ValueError(
                    "ion materials must be distinct from cohort markers"
                )


def wrap_cohort_operations(owner, plans, operations):
    """Initialize labels and observe after each ordered physical process."""
    cohort = next((p for p in plans if p.family == "cohort"), None)
    if cohort is None:
        return operations
    from particula.gpu.kernels.cohort import (
        initialize_cohort,
        merge_cohort,
        observe_cohort,
    )

    wp, session = owner._wp, owner._session
    p, device = session.particles, owner.device.native
    shape = p.concentration.shape
    pending = wp.zeros(shape, dtype=wp.int64, device=device)
    background = wp.zeros(shape, dtype=wp.int32, device=device)
    statistics = wp.zeros((shape[0], 4), dtype=wp.int64, device=device)
    owner._auxiliary.update(
        cohort_pending=pending,
        cohort_background=background,
        cohort_statistics=statistics,
    )
    wp.launch(
        initialize_cohort,
        dim=shape[0],
        device=device,
        inputs=[
            p.masses,
            p.concentration,
            cohort.values["seed_species"],
            cohort.values["background_species"],
            pending,
            background,
            statistics,
        ],
    )
    observe_args = [
        p.masses,
        p.concentration,
        p.density,
        wp.float64(np.pi / 6 * cohort.values["target_diameter"] ** 3),
        pending,
        background,
        statistics,
        owner._device_failure,
    ]

    def observe():
        wp.launch(
            observe_cohort, dim=shape[0], inputs=observe_args, device=device
        )

    observe()
    resources = (
        owner._registry.acquire_coagulation(shape[1])
        if any(p.family == "coagulation" for p in plans)
        else None
    )

    def wrap(operation, family):
        def enqueue():
            operation()
            if family == "coagulation":
                wp.launch(
                    merge_cohort,
                    dim=shape[0],
                    device=device,
                    inputs=[
                        resources.collision_pairs,
                        resources.n_collisions,
                        pending,
                        background,
                        owner._device_failure,
                    ],
                )
            observe()

        return enqueue

    return tuple(
        operation if plan.family == "cohort" else wrap(operation, plan.family)
        for plan, operation in zip(plans, operations, strict=True)
    )


def validate_cohort_checkpoint(saved):
    """Check outcome partition and unresolved labels before any upload."""
    arrays = {
        item.role: np.frombuffer(item.data, dtype=item.dtype).reshape(
            item.shape
        )
        for item in saved.auxiliary
        if item.role.startswith("cohort_")
    }
    if not arrays:
        return
    stats = arrays["cohort_statistics"]
    pending = arrays["cohort_pending"]
    background = arrays["cohort_background"]
    if (
        np.any(stats < 0)
        or np.any(pending < 0)
        or np.any(stats[:, 0] > saved.resident.dimensions.n_particles)
        or np.any(pending > stats[:, :1])
        or np.any((background != 0) & (background != 1))
        or np.any(stats[:, 0] != stats[:, 1:].sum(axis=1))
        or np.any(stats[:, 3] != pending.sum(axis=1))
    ):
        raise ValueError("invalid checkpoint cohort lineage partition")
