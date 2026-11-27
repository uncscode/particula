"""Build a dry-state-specific approximation to the measured vapor history."""

import argparse
from pathlib import Path

import numpy as np
from history_population import prehistory
from projection_study import digest


def powers(a):
    """Cubic tensor products in the two normalized volatile coefficients."""
    x = np.polynomial.polynomial.polyvander(a[:, 0] / 0.9, 3)
    y = np.polynomial.polynomial.polyvander(a[:, 1] / 0.9, 3)
    return (x[:, :, None] * y[:, None, :]).reshape(len(a), 16)


def interpolant(d):
    """Fit response from 16 histories at fixed dry state."""
    v = 0.7 * np.array([-1.0, -0.5, 0.5, 1.0])
    points = np.stack(np.meshgrid(v, v, indexing="ij"), -1).reshape(-1, 2)
    basis = np.linalg.inv(powers(points))
    local = {
        k: d[k].repeat(len(points), 0)
        for k in ("fixed", "number", "mean", "previous", "basis")
    }
    local["coef"] = np.tile(points, (len(d["fixed"]), 1))
    hist = prehistory(local, subdivisions=4)[0]
    change = ((hist - hist[:, -1:]) * 1e6).reshape(len(d["fixed"]), 16, -1)
    return (basis @ change).reshape(len(d["fixed"]), 16, hist.shape[1], 2)


def prepare(record, out):
    """Store the observation map without using labels to construct it."""
    with np.load(record) as a:
        data = {
            k: a[k][::2]
            for k in ("fixed", "number", "mean", "previous", "basis")
        }
        truth = a["history"]
        coef = a["coef"]
        ids = a["ids"]
    blocks = []
    for i in range(0, len(ids), 128):
        blocks.append(interpolant({k: v[i : i + 128] for k, v in data.items()}))
    response = np.concatenate(blocks).astype(np.float32)
    pred = np.einsum("bc,bctj->btj", powers(coef), response.repeat(2, 0))
    error = np.abs(pred - (truth - truth[:, -1:]) * 1e6)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        ids=ids,
        coeff=response,
        input_sha256=digest(record),
        interpolation_mae=error.mean((0, 1)),
        interpolation_max=error.max((0, 1)),
    )
    tmp.replace(out)
    print(
        len(ids),
        out,
        "history approximation MAE (mg/m3)",
        error.mean((0, 1)),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("record", type=Path)
    p.add_argument("out", type=Path)
    prepare(**vars(p.parse_args()))
