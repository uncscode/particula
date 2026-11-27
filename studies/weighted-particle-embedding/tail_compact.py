"""Compress four composition moments into a centered three-class rule."""

import argparse
from pathlib import Path

import numpy as np
import torch
from multicycle_storage import forcing
from node_transfer import write_table
from precision_horizon import pack, unpack
from quadrature import restore_arrays
from tail_quadrature import encode
from tail_study import physical, read, score


def shape(mass, counts, bins, *, learn_tail=False):
    """Encode size inventories, mean, variance, skew offset and tail weight."""
    raw = encode(mass, counts, bins, 2 if learn_tail else 3)
    n, dry, water = raw[:, :, :3].transpose(2, 0, 1)
    moments = np.divide(
        raw[:, :, 3:7],
        dry[:, :, None],
        out=np.zeros_like(raw[:, :, 3:7]),
        where=dry[:, :, None] > 0,
    )
    # Source moments are of u=2*f-1. Compute central moments in f units.
    u, m2, m3 = moments[:, :, :3].transpose(2, 0, 1)
    mean = (1 + u) / 2
    var = np.maximum(m2 - u * u, 0) / 4
    third = (m3 - 3 * u * m2 + 2 * u**3) / 8
    nonzero = var > 1e-12
    shift = np.divide(third, var, out=np.zeros_like(var), where=nonzero)
    tail = np.zeros_like(var)
    if not learn_tail:
        m4 = moments[:, :, 3]
        fourth = (m4 - 4 * u * m3 + 6 * u * u * m2 - 3 * u**4) / 16
        prod = np.maximum(
            np.divide(fourth, var, out=np.zeros_like(var), where=nonzero)
            - shift**2,
            var,
        )
        tail = np.divide(var, prod, out=np.zeros_like(var), where=nonzero)

    def shares(a):
        total = a.sum(1, keepdims=True)
        return np.divide(a, total, out=np.zeros_like(a), where=total > 0)

    code = np.stack(
        (
            shares(n),
            shares(dry),
            shares(water),
            mean,
            2 * np.sqrt(var),
            (1 + shift) / 2,
            tail,
        ),
        axis=-1,
    )
    if code.min() < -1e-8 or code.max() > 1 + 1e-8:
        raise ArithmeticError("Composition coordinates outside physical bounds")
    totals = np.column_stack(
        (counts.sum(1), (mass * counts[:, :, None]).sum(1))
    )
    return code.clip(0, 1), totals


def nodes(code, totals):
    """Match four centered moments using nodes at mean-a, mean and mean+b."""
    n, dry, water, mean, sigma, offset, tail = code.transpose(2, 0, 1)
    shift = 2 * offset - 1
    variance = (sigma / 2) ** 2
    prod = np.divide(variance, tail, out=np.zeros_like(tail), where=tail > 0)
    length = np.sqrt(shift**2 + 4 * prod)
    a = (length - shift) / 2
    b = (length + shift) / 2
    left = np.divide(tail * b, length, out=np.zeros_like(b), where=length > 0)
    right = tail - left
    weights = np.stack((left, 1 - tail, right), axis=-1)
    f = np.stack((mean - a, mean, mean + b), axis=-1)
    clipped = int(
        (((f < 0) | (f > 1)) & (weights > 0) & (n[:, :, None] > 0)).sum()
    )
    f = f.clip(0, 1)
    actual = (weights * f).sum(-1)
    gain = np.divide(
        np.maximum(mean - actual, 0),
        1 - actual,
        out=np.zeros_like(mean),
        where=actual < 1,
    )
    loss = np.divide(
        np.maximum(actual - mean, 0),
        actual,
        out=np.zeros_like(mean),
        where=actual > 0,
    )
    f = f + (1 - f) * gain[:, :, None] - f * loss[:, :, None]
    units = (
        np.stack(
            (
                np.broadcast_to(n[:, :, None], f.shape),
                np.broadcast_to(water[:, :, None], f.shape),
                dry[:, :, None] * (1 - f),
                dry[:, :, None] * f,
            ),
            axis=-1,
        )
        * weights[:, :, :, None]
    )
    units = np.where(n[:, :, None, None] > 0, units, 0.0)
    counts, mass = restore_arrays(units.reshape(len(n), -1, 4), totals)
    return mass, counts, clipped


def store(path, code, totals, bits, **common):
    """Write compact coordinates and decode using the physical unit interval."""
    if bits == 64:
        payload = dict(shape=code)
    elif bits == 16:
        payload = dict(shape=code.astype(np.float16))
    else:
        packed, _ = pack(code, 0.5, 0.5, bits)
        payload = dict(packed=packed, dimensions=code.shape)
    np.savez_compressed(path, **common, **payload, totals=totals, bits=bits)


def restore(a):
    """Recover the measured quantization levels before reconstructing nodes."""
    if "packed" in a:
        code = unpack(
            a["packed"], 0.5, 0.5, int(a["bits"]), a["dimensions"]
        ).astype(np.float64)
    else:
        code = a["shape"].astype(np.float64)
    return (*nodes(code, a["totals"]), code)


def assess(root, prefix, folder, out, bins, bits, substeps, split="validation"):
    """Compare storage and resolved response across size and bit budgets."""
    parent = root / "composition-tails"
    d = read(parent / f"{prefix}128" / split)
    coarse = read(parent / f"{prefix}64" / split)
    np.testing.assert_array_equal(d["ids"], coarse["ids"])
    if len(d["ids"]) != 64:
        raise ValueError("Incomplete independent families")
    t = np.broadcast_to(d["time"], (len(d["ids"]), len(d["time"]))).copy()
    delta = abs(d["w"] - coarse["w"])
    truth = dict(
        time=torch.as_tensor(t.repeat(2, 0)),
        water=torch.as_tensor(d["w"]),
        ref=(d["w"][1::2] - d["w"][::2], delta[1::2] + delta[::2]),
    )
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in bins:
        codes, totals = shape(d["m"], d["n"], size)
        for k in bits:
            name = f"centered-three-b{size}-p{k}"
            path = folder / (name + ".npz")
            store(
                path,
                codes,
                totals,
                k,
                ids=d["ids"],
                levels=d["levels"][::2],
                time=d["time"],
                volume=d["volume"],
                density=d["density"],
            )
            with np.load(path) as a:
                mass, n, clipped, record = restore(a)
                levels = a["levels"]
                steps = (len(a["time"]) - 2) // 2
                force = forcing(
                    np.full((len(levels), 2), steps), levels, a["time"][1]
                )
                w, g = physical(
                    mass,
                    n,
                    levels[:, 0].repeat(2),
                    force,
                    float(a["volume"]),
                    substeps,
                )
                row = score(name, path, 0, w, g, truth, t, record, "")
                rows.append(
                    dict(
                        radius_bins=size,
                        precision=k,
                        clipped_nodes=clipped,
                        **row,
                    )
                )
    write_table(rows, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("root", "folder", "out"):
        p.add_argument(k, type=Path)
    p.add_argument(
        "--split",
        choices=("validation", "test", "challenge"),
        default="validation",
    )
    p.add_argument("--prefix", default="refinement-s")
    p.add_argument("--bins", type=int, nargs="+", default=[4, 6, 8, 12, 16])
    p.add_argument("--bits", type=int, nargs="+", default=[8, 10, 12, 16, 64])
    p.add_argument("--substeps", type=int, default=12)
    torch.set_num_threads(1)
    assess(**vars(p.parse_args()))
