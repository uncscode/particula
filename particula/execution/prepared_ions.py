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
