"""Store within-size composition moments and resolve more than two classes."""

import numpy as np
from moment_quadrature import size_nodes


def encode(mass, counts, bins, classes):
    """Pool dry-weighted composition powers and inventories by size."""
    if bins < 1 or classes < 1:
        raise ValueError("Expected positive size and composition resolutions")
    record = np.zeros((len(mass), bins, 3 + 2 * classes - 1))
    for box, (m, n) in enumerate(zip(mass, counts, strict=True)):
        active = n > 0
        if not active.any():
            continue
        m, n = m[active], n[active]
        dry = m[:, 1:].sum(-1)
        if (dry <= 0).any():
            raise ValueError("Expected a positive dry seed")
        # Equal material density makes log dry mass affine in log dry radius.
        axis = np.log(dry)
        span = np.ptp(axis)
        if span < 1e-12 or bins == 1:
            group = np.zeros(len(n), dtype=int)
        else:
            group = (
                np.floor((axis - axis.min()) / span * bins + 1e-10)
                .astype(int)
                .clip(0, bins - 1)
            )
        comp = 2 * m[:, 2] / dry - 1
        powers = comp[:, None] ** np.arange(1, 2 * classes)
        features = np.column_stack(
            (n, n * dry, n * m[:, 0], (n * dry)[:, None] * powers)
        )
        np.add.at(record[box], group, features)
    return record


def decode(record):
    """Decode positive Gaussian rules and preserve bulk inventories."""
    boxes, bins, dim = record.shape
    classes = (dim - 2) // 2
    if dim != 2 + 2 * classes or classes < 1:
        raise ValueError("Invalid composition moment record")
    counts = np.zeros((boxes, bins, classes))
    mass = np.zeros((*counts.shape, 3))
    for box, i in zip(*np.where(record[:, :, 0] > 0), strict=True):
        values = record[box, i]
        num, dry, water = values[:3]
        moment = np.r_[1.0, values[3:] / dry]
        x, w = size_nodes(moment)
        frac = ((x + 1) / 2).clip(0, 1)
        mean = (1 + moment[1]) / 2
        current = w @ frac
        if current < mean and current < 1:
            frac += (1 - frac) * (mean - current) / (1 - current)
        elif current > mean and current > 0:
            frac *= mean / current
        target = mass[box, i, : len(w)]
        target[:, 0] = water / num
        target[:, 1] = dry / num * (1 - frac)
        target[:, 2] = dry / num * frac
        counts[box, i, : len(w)] = num * w
    return mass.reshape(boxes, bins * classes, 3), counts.reshape(
        boxes, bins * classes
    )
