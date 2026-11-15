"""Stable logical-box random words for explicitly stochastic GPU extensions."""

import hashlib
import json

import numpy as np


def initial_words(configuration, root_seed, box_ids):
    """Derive versioned uint32 words without depending on lane or process order.

    These are Warp random-state words, separate from CPU PCG64 continuation.
    Reject collisions within a process instead of silently sharing a stream.
    """
    words = []
    for box_id in box_ids:
        payload = json.dumps(
            [
                "particula.gpu_extension_rng",
                1,
                int(root_seed),
                configuration.extension_id,
                configuration.name,
                box_id,
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        words.append(
            int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
        )
    if len(set(words)) != len(words):
        raise ValueError("derived custom GPU RNG word collision")
    return np.asarray(words, dtype=np.uint32)


def validate_streams(plans, root_seed, box_ids):
    """Check initial derivation before uploading any simulation buffers."""
    from particula.execution.process_extensions import _RegisteredPlan

    for plan in plans:
        if type(plan) is _RegisteredPlan and plan.configuration.stochastic:
            initial_words(plan.configuration, root_seed, box_ids)
