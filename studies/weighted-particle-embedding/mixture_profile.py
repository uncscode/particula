"""Route conditional composition profiles to training-fitted local bases."""

import numpy as np
from conditional_geometry import (
    baseline,
    fit_geometry,
    read_geometry,
    store_geometry,
)


def distances(x, centers):
    """Compute squared Euclidean distances in the shared routing coordinates."""
    return np.square(x[:, None] - centers[None]).sum(-1)


def fit_mixture(core, profiles, totals, degree, experts):
    """Fit training clusters and their local profile bases."""
    residual = (profiles - baseline(core, totals)).reshape(len(core), -1)
    router_scale = residual.std(0).clip(1e-7)
    # Bound only the routing features so a rare extreme cannot claim a whole
    # expert at initialization. The local bases fit the original profiles.
    x = np.tanh(residual / router_scale)
    _, _, axes = np.linalg.svd(x - x.mean(0), full_matrices=False)
    order = np.argsort(x @ axes[0])
    centers = np.stack(
        [x[group].mean(0) for group in np.array_split(order, experts)]
    )
    labels = np.full(len(x), -1)
    for _ in range(120):
        new = distances(x, centers).argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        if np.unique(labels).size != experts:
            raise ValueError("An expert received no training families")
        centers = np.stack([x[labels == i].mean(0) for i in range(experts)])
    fits = [
        fit_geometry(
            core[labels == i],
            profiles[labels == i],
            totals[labels == i],
            degree,
        )
        for i in range(experts)
    ]
    if min(len(m["axes"]) for m in fits) < degree * 5:
        raise ValueError("Too few training records for a local basis")
    return dict(
        degree=np.asarray(degree),
        router_scale=router_scale,
        router_centers=centers,
        train_count=np.bincount(labels),
        center=np.stack([m["center"] for m in fits]),
        scale=np.stack([m["scale"] for m in fits]),
        axes=np.stack([m["axes"] for m in fits]),
    )


def select(model, i):
    """Expose the chosen local linear model in the shared profile format."""
    return dict(
        degree=model["degree"],
        **{k: model[k][i] for k in ("scale", "center", "axes")},
    )


def encode_mixture(core, profiles, totals, model, width):
    """Store an expert identifier and the learned shape coordinates."""
    residual = (profiles - baseline(core, totals)).reshape(len(core), -1)
    labels = distances(
        np.tanh(residual / model["router_scale"]), model["router_centers"]
    ).argmin(1)
    labels[totals[:, 0] == 0] = 0
    record = np.zeros((len(core), width))
    for i in np.unique(labels):
        use = labels == i
        record[use] = store_geometry(
            core[use], profiles[use], totals[use], select(model, i), width
        )
    return record, labels


def decode_mixture(record, labels, totals, model):
    """Restore positive node shares using the recorded expert for each box."""
    degree = int(model["degree"])
    shape = np.zeros((len(record), 2 * degree, 4))
    corrections = np.zeros(len(record), int)
    for i in np.unique(labels):
        use = labels == i
        shape[use], corrections[use] = read_geometry(
            record[use], totals[use], select(model, i)
        )
    return shape, corrections


def pack_routes(labels, experts):
    """Charge the byte-aligned payload for every routed aerosol."""
    bits = (experts - 1).bit_length()
    stream = (labels[:, None] >> np.arange(bits)) & 1
    return np.packbits(stream.ravel(), bitorder="little")


def unpack_routes(packed, count, experts):
    """Extract the transmitted route and reject invalid class identities."""
    bits = (experts - 1).bit_length()
    stream = np.unpackbits(packed, bitorder="little", count=count * bits)
    labels = stream.reshape(count, bits) @ (1 << np.arange(bits))
    if (labels >= experts).any():
        raise ValueError("Invalid mixture route")
    return labels
