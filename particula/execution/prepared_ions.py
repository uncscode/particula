"""Prepared paired-ion source with independent, checkpointed random words."""

import hashlib
import json

import numpy as np


def plan_ion_source(process, aerosol):
    """Freeze physical configuration after native source validation."""
    from particula.execution.prepared_processes import _ProcessPlan

    process._validate_batch(aerosol)
    return _ProcessPlan(
        "ion_source",
        tuple(
            (name, getattr(process, name))
            for name in (
                "production_rate",
                "positive_species",
                "negative_species",
                "positive_mass",
                "negative_mass",
            )
        ),
    )


def compile_ion_source(plan, owner):
    """Allocate source-owned state once; ordinary steps only enqueue kernels."""
    from particula.gpu.kernels.ion_source import inject_ion_pairs

    wp, session = owner._wp, owner._session
    p = session.particles
    stream = session.metadata.stream
    words = [
        int.from_bytes(
            hashlib.sha256(
                json.dumps(
                    [
                        "particula.ion-pair-source.v1",
                        int(stream.root_seed),
                        box,
                    ],
                    separators=(",", ":"),
                ).encode()
            ).digest()[:4],
            "little",
        )
        for box in stream.logical_box_ids
    ]
    rng = wp.array(
        np.asarray(words, dtype=np.uint32),
        dtype=wp.uint32,
        device=owner.device.native,
    )
    injected = wp.zeros(
        session.dimensions.n_boxes, dtype=wp.int64, device=owner.device.native
    )
    owner._auxiliary["ion_source_rng"] = rng
    owner._auxiliary["ion_source_pairs"] = injected
    v = plan.values
    args = [
        p.masses,
        p.concentration,
        p.charge,
        p.volume,
        wp.float64(v["production_rate"]),
        wp.float64(owner.time_step),
        v["positive_species"],
        v["negative_species"],
        wp.float64(v["positive_mass"]),
        wp.float64(v["negative_mass"]),
        rng,
        injected,
        owner._device_failure,
    ]

    def enqueue():
        wp.launch(
            inject_ion_pairs,
            dim=session.dimensions.n_boxes,
            inputs=args,
            device=owner.device.native,
        )

    return enqueue


def plan_ion_cleanup(process, aerosol):
    """Freeze the explicit reserved material map for neutral export."""
    from particula.execution.prepared_processes import _ProcessPlan

    process._validate_batch(aerosol)
    return _ProcessPlan(
        "ion_cleanup",
        (
            ("positive_species", process.positive_species),
            ("negative_species", process.negative_species),
            ("interval_steps", process.interval_steps),
        ),
    )


def compile_ion_cleanup(plan, owner):
    """Allocate checkpointed export ledgers and retain one device launch."""
    from particula.gpu.kernels.ion_cleanup import remove_neutral_ion_products

    wp, session = owner._wp, owner._session
    dimensions = session.dimensions
    mass = wp.zeros(
        (dimensions.n_boxes, dimensions.n_species),
        dtype=wp.float64,
        device=owner.device.native,
    )
    particles = wp.zeros(
        dimensions.n_boxes, dtype=wp.int64, device=owner.device.native
    )
    phase = wp.zeros(
        dimensions.n_boxes, dtype=wp.int64, device=owner.device.native
    )
    owner._auxiliary["ion_cleanup_phase"] = phase
    owner._auxiliary["ion_export_mass"] = mass
    owner._auxiliary["ion_export_particles"] = particles
    p = session.particles
    args = [
        p.masses,
        p.concentration,
        p.charge,
        plan.values["positive_species"],
        plan.values["negative_species"],
        mass,
        particles,
        owner._device_failure,
        wp.int64(plan.values["interval_steps"]),
        phase,
    ]

    def enqueue():
        wp.launch(
            remove_neutral_ion_products,
            dim=dimensions.n_boxes,
            inputs=args,
            device=owner.device.native,
        )

    return enqueue


def plan_ion_exclusion(process, aerosol):
    """Freeze the reserved material lanes used to gate vapor proposals."""
    from particula.execution.prepared_processes import _ProcessPlan

    process._validate_batch(aerosol)
    return _ProcessPlan(
        "ion_condensation",
        (
            ("positive_species", process.positive_species),
            ("negative_species", process.negative_species),
        ),
    )


def compile_ion_exclusion(_plan, _owner):
    """Condensation compilation consumes this companion configuration."""
    return lambda: None


def validate_ion_plans(plans):
    """Require condensation and compatible, nonvolatile reserved ion lanes."""
    from particula.execution.prepared_atmospheric import (
        validate_atmospheric_companions,
    )

    validate_atmospheric_companions(plans)
    exclusion = next((p for p in plans if p.family == "ion_condensation"), None)
    if exclusion is None:
        return
    condensation = next((p for p in plans if p.family == "condensation"), None)
    if condensation is None:
        raise ValueError("ion condensation exclusion requires condensation")
    indices = (
        exclusion.values["positive_species"],
        exclusion.values["negative_species"],
    )
    if any(material in indices for _, material in condensation.values["pairs"]):
        raise ValueError("reserved ion materials must be nonvolatile")
    for plan in plans:
        if plan.family in ("ion_source", "ion_cleanup"):
            if (
                plan.values["positive_species"],
                plan.values["negative_species"],
            ) != indices:
                raise ValueError("ion processes require matching material maps")


def validate_cleanup_phase(saved, plans):
    """Reject a restart clock outside the configured cleanup period."""
    plan = next((p for p in plans if p.family == "ion_cleanup"), None)
    if plan is None:
        return
    item = next(p for p in saved.auxiliary if p.role == "ion_cleanup_phase")
    values = np.frombuffer(item.data, dtype=item.dtype)
    if np.any(values < 0) or np.any(values >= plan.values["interval_steps"]):
        raise ValueError("checkpoint ion cleanup phase is outside its interval")
