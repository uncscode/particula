"""Compress conditional moments while preserving the dry-size quadrature."""

import numpy as np
from moment_quadrature import encode


def context(size, totals):
    """Use intensive dry/water composition and normalized number moments."""
    dry = totals[:, 2:].sum(1)
    denom = np.where(dry > 0, dry, 1.0)
    fraction = totals[:, 3] / denom
    hydration = totals[:, 1] / denom
    features = np.column_stack(
        [fraction, fraction**2, fraction**3, hydration, fraction * hydration]
    )
    # Expand only known quantities and keep dimensions independent of box size.
    return np.column_stack([np.ones(len(size)), size, features])


def parts(code, totals, degree):
    """Separate independent dry moments and homogeneous composition terms."""
    size = code[:, : 2 * degree - 1]
    core = np.delete(size, 2, axis=1)
    c = context(core, totals)
    marginal = np.column_stack([np.ones(len(size)), size[:, : degree - 1]])
    homogeneous = (marginal[:, :, None] * c[:, None, -5:]).reshape(
        len(size), -1
    )
    return core, c, homogeneous


def fit(codes, totals, degree):
    """Fit a conditional mean and residual axes on the training boxes."""
    if degree < 2:
        raise ValueError("At least two dry radius nodes are needed")
    core, c, base = parts(codes, totals, degree)
    joint = codes[:, 2 * degree - 1 :] - base
    # Center/scale the non-constant columns for a stable conditional regression.
    loc = c.mean(0)
    scale = c.std(0).clip(1e-8)
    loc[0], scale[0] = 0.0, 1.0
    c = (c - loc) / scale
    linear = np.linalg.lstsq(c, joint, rcond=1e-8)[0]
    remainder = joint - c @ linear
    joint_scale = remainder.std(0).clip(1e-10)
    _, singular, axes = np.linalg.svd(
        remainder / joint_scale, full_matrices=False
    )
    return dict(
        degree=np.asarray(degree),
        context_mean=loc,
        context_scale=scale,
        linear=linear,
        joint_scale=joint_scale,
        axes=axes,
        singular_values=singular,
    )


def expand_size(core):
    """Restore third centered radius moment from the exact dry mass scale."""
    # E[(1+u)^3] = 1 for a nonempty dry population.
    return np.column_stack(
        [core[:, :2], -3 * (core[:, 0] + core[:, 1]), core[:, 2:]]
    )


def project(code, totals, model, width):
    """Store independent size moments plus learned conditional coordinates."""
    degree = int(model["degree"])
    core, c, base = parts(code, totals, degree)
    keep = width - core.shape[1]
    if not 0 <= keep <= len(model["axes"]):
        raise ValueError("Width outside the projected moment range")
    center = (
        (c - model["context_mean"]) / model["context_scale"] @ model["linear"]
    )
    residual = (code[:, 2 * degree - 1 :] - base - center) / model[
        "joint_scale"
    ]
    record = np.column_stack([core, residual @ model["axes"][:keep].T])
    record[totals[:, 0] == 0] = 0
    return record


def recover(record, totals, model):
    """Use only the transmitted coordinates and exact physical inventories."""
    degree = int(model["degree"])
    offset = 2 * degree - 2
    core = record[:, :offset].astype(np.float64)
    size = expand_size(core)
    c = context(core, totals)
    marginal = np.column_stack([np.ones(len(size)), size[:, : degree - 1]])
    base = (marginal[:, :, None] * c[:, None, -5:]).reshape(len(size), -1)
    center = (
        (c - model["context_mean"]) / model["context_scale"] @ model["linear"]
    )
    rest = record[:, offset:].astype(np.float64)
    residual = rest @ model["axes"][: rest.shape[1]] * model["joint_scale"]
    joint = base + center + residual
    joint[totals[:, 0] == 0] = 0
    return np.column_stack([size, joint])


def training_records(folders, degree):
    """Read initial particles from unique training families."""
    codes, totals, ids = [], [], set()
    for folder in folders:
        for file in sorted(folder.glob("families-*.npz")):
            with np.load(file) as d:
                group = d["ids"]
                if set(group) & ids:
                    raise ValueError("Repeated families in projection training")
                ids.update(group)
                mass = d["mass0"] if "mass0" in d else d["mass"][0]
                c, t = encode(mass, d["counts"], degree)
                codes.append(c)
                totals.append(t)
    return (
        np.concatenate(codes),
        np.concatenate(totals),
        np.asarray(sorted(ids)),
    )
