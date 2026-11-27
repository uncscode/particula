"""Pool and compress mixing-state profiles conditional on dry radius."""

import numpy as np
from moment_quadrature import conditional_shares, size_nodes
from projected_moments import context, expand_size


def geometry(code, totals, degree):
    """Resolve two composition classes on each occupied Gaussian radius."""
    size = code[:, : 2 * degree - 1]
    core = np.delete(size, 2, axis=1)
    profiles = np.zeros((len(code), degree, 5))
    for b in np.flatnonzero(totals[:, 0] > 0):
        u, n = size_nodes(np.r_[1.0, size[b]])
        k = len(u)
        moments = code[b, 2 * degree - 1 :].reshape(degree, 5)
        cond = (
            np.linalg.solve(np.vander(u, k, increasing=True).T, moments[:k])
            / n[:, None]
        )
        for j, raw in enumerate(cond):
            shares, _ = conditional_shares(raw)
            weight, water, _, rich = shares.T
            occupied = weight > 1e-12
            f = rich[occupied] / weight[occupied]
            w = water[occupied] / weight[occupied]
            upper = weight[1] if occupied.all() else 0.5
            profiles[b, j] = (
                weight[occupied] @ f,
                f[-1] - f[0],
                upper,
                water.sum(),
                w[-1] - w[0],
            )
        profiles[b, k:] = profiles[b, k - 1]
    return core, profiles


def baseline(core, totals):
    """A single internally mixed class with the exact bulk water loading."""
    c = context(core, totals)
    f, w = c[:, -5], c[:, -2]
    return np.column_stack(
        [f, np.zeros_like(f), np.full_like(f, 0.5), w, np.zeros_like(w)]
    )[:, None]


def fit_geometry(core, profiles, totals, degree):
    """Fit principal axes after removing known bulk composition."""
    residual = (profiles - baseline(core, totals)).reshape(len(core), -1)
    mu = residual.mean(0)
    scale = residual.std(0).clip(1e-7)
    _, _, axes = np.linalg.svd((residual - mu) / scale, full_matrices=False)
    return dict(degree=np.asarray(degree), center=mu, scale=scale, axes=axes)


def store_geometry(core, profiles, totals, model, width):
    """Return the independent size moments and learned profile coordinates."""
    keep = width - core.shape[1]
    if not 0 <= keep <= len(model["axes"]):
        raise ValueError("Width outside profile basis")
    r = (profiles - baseline(core, totals)).reshape(len(core), -1)
    record = np.column_stack(
        (core, (r - model["center"]) / model["scale"] @ model["axes"][:keep].T)
    )
    record[totals[:, 0] == 0] = 0
    return record


def read_geometry(record, totals, model):
    """Decode positive node shares, restoring every nonempty inventory."""
    degree = int(model["degree"])
    c = 2 * degree - 2
    core = record[:, :c].astype(np.float64)
    projected = record[:, c:].astype(np.float64)
    profile = (
        projected @ model["axes"][: projected.shape[1]] * model["scale"]
        + model["center"]
    ).reshape(-1, degree, 5)
    profile += baseline(core, totals)
    return shape_from_profile(core, profile, totals, degree)


def shape_from_profile(core, profile, totals, degree):
    """Constrain predicted geometry and match the supplied inventories."""
    size = expand_size(core)
    shape = np.zeros((len(core), degree * 2, 4))
    changed = np.zeros(len(core), dtype=int)
    for b in np.flatnonzero(totals[:, 0] > 0):
        u, n = size_nodes(np.r_[1.0, size[b]])
        k = len(n)
        mean, df, upper, water, dw = profile[b, :k].T
        hi = upper.clip(0, 1)
        mix = np.column_stack((1 - hi, hi))
        raw_f = mean[:, None] + np.column_stack((-hi, 1 - hi)) * df[:, None]
        raw_w = water[:, None] + np.column_stack((-hi, 1 - hi)) * dw[:, None]
        f = raw_f.clip(0, 1)
        wet = raw_w.clip(0)
        changed[b] = np.count_nonzero(
            (raw_f != f) | (raw_w != wet)
        ) + np.count_nonzero(upper != hi)
        count = (n[:, None] * mix).ravel()
        dry_weight = (
            n[:, None] * mix * np.maximum(1 + u[:, None], 0) ** 3
        ).ravel()
        shape[b, : 2 * k] = np.column_stack(
            (
                count,
                dry_weight * wet.ravel(),
                dry_weight * (1 - f.ravel()),
                dry_weight * f.ravel(),
            )
        )
        for i in np.flatnonzero((shape[b].sum(0) == 0) & (totals[b] > 0)):
            # Allocate a known nonzero inventory over the available dry mass.
            shape[b, : 2 * k, i] = dry_weight
            changed[b] += 1
    norm = shape.sum(1, keepdims=True)
    return np.divide(
        shape, norm, out=np.zeros_like(shape), where=norm > 0
    ), changed
