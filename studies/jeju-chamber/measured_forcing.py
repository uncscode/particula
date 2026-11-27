"""Prescribe chamber temperature contrasts from the measured coldest and mean air."""

import numpy as np


def impose_cold_probe(heat, field, grid, dt, fraction, matrices, mirror=False):
    """Preserve mean air and spatial ordering, or reflect that order horizontally."""
    n = grid.size
    history = field["forcing"]
    free = history[..., :n].astype(float)
    shape = free - free.mean(axis=-1, keepdims=True)
    nx, ny, nz = grid.shape
    if mirror:
        shaped = shape.reshape(*shape.shape[:-1], nz, ny, nx)
        shape = (shaped[..., ::-1, :] if ny > 1 else shaped[..., ::-1]).reshape(
            shape.shape
        )
    span = -shape.min(axis=-1, keepdims=True)
    # When the unforced field is spatially uniform there is no measured location.
    # Seed that single direction by the lower half of the grid and retain the mean.
    fallback = np.repeat(np.arange(nz), ny * nx).astype(float)
    fallback -= fallback.mean()
    fallback /= abs(fallback.min())
    unit = np.divide(shape, span, out=np.zeros_like(shape), where=span > 1.0e-8)
    unit = np.where(span > 1.0e-8, unit, fallback)
    deficit = np.maximum(0.0, heat.air[:, ::dt] - heat.minimum[:, ::dt])
    target = deficit[None, :, :, None] * unit
    new = (1 - fraction) * free + fraction * target
    history[..., :n] = new
    alive = heat.valid[:, ::dt]
    error = new.min(axis=-1) + deficit[None]
    field["imposed_min_rmse_k"] = np.sqrt(
        (error**2 * alive).sum(axis=2) / alive.sum(axis=1)
    )
    field["imposed_mean_max_k"] = np.max(np.abs(new.mean(axis=-1)))
    shift = new - free
    support = alive[None]
    field["imposed_temperature_adjustment_rms_k"] = np.sqrt(
        ((shift**2).mean(axis=-1) * support).sum(axis=2) / support.sum(axis=2)
    )
    # Difference of the two discrete heat updates. The upper and wall source
    # terms cancel. Retain the extra local tendency needed by the imposed spread.
    if matrices.ndim == 5:
        mixed = np.empty_like(shift[:, :, :-1])
        minute = np.arange(mixed.shape[2]) * dt // 60
        for k in np.unique(minute):
            use = minute == k
            mixed[:, :, use, :] = np.einsum(
                "crij,crtj->crti",
                matrices[k],
                shift[:, :, :-1, :][:, :, use, :],
            )
    else:
        mixed = np.einsum("cij,crtj->crti", matrices, shift[:, :, :-1])
    propagated = np.exp(-dt * history[:, :, :-1, 2 * n :]) * mixed
    propagated -= propagated.mean(axis=-1, keepdims=True)
    extra = (shift[:, :, 1:] - propagated) * 60 / dt
    use = support[:, :, 1:] & support[:, :, :-1]
    field["extra_local_tendency_rms_kmin"] = np.sqrt(
        ((extra**2).mean(axis=-1) * use).sum(axis=2) / use.sum(axis=2)
    )

    if fraction == 1:
        assert np.max(np.abs(error[:, alive])) < 5.0e-7
    assert field["imposed_mean_max_k"] < 2.0e-7
