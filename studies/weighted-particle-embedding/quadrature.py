"""Positive size quadrature within dry composition classes."""

import numpy as np


def size_quadrature(radius, counts, nodes):
    """Approximate the number measure by positive Gaussian quadrature."""
    unique, inverse = np.unique(radius, return_inverse=True)
    weights = np.bincount(inverse, weights=counts)
    if len(unique) <= nodes:
        return unique, weights
    # Map size onto [-1, 1] to condition the symmetric Lanczos recurrence.
    middle = 0.5 * (unique[0] + unique[-1])
    scale = 0.5 * (unique[-1] - unique[0])
    size = (unique - middle) / scale
    basis = np.sqrt(weights / weights.sum())
    previous = np.zeros_like(basis)
    beta = 0.0
    diagonal, subdiagonal = [], []
    for k in range(nodes):
        v = size * basis - beta * previous
        alpha = np.dot(basis, v)
        diagonal.append(alpha)
        v -= alpha * basis
        beta = np.linalg.norm(v)
        if k + 1 == nodes or beta <= 1e-14:
            break
        subdiagonal.append(beta)
        previous, basis = basis, v / beta
    off = np.diag(subdiagonal, 1)
    locations, vectors = np.linalg.eigh(np.diag(diagonal) + off + off.T)
    return middle + scale * locations, weights.sum() * vectors[0] ** 2


def classify(frac, count, classes):
    """Partition a continuous dry composition coordinate by weighted k-means."""
    if classes == 1 or np.ptp(frac) < 1e-12:
        return np.zeros(len(frac), dtype=int)
    center = np.linspace(frac.min(), frac.max(), classes)
    for _ in range(32):
        group = np.abs(frac[:, None] - center).argmin(1)
        next_center = center.copy()
        for i in np.unique(group):
            selected = group == i
            next_center[i] = np.average(frac[selected], weights=count[selected])
        if np.allclose(next_center, center, rtol=0, atol=1e-12):
            break
        center = next_center
    return np.abs(frac[:, None] - center).argmin(1)


def quadrature_record(masses, counts, classes, nodes):
    """Store four normalized inventories per node in each occupied class."""
    if classes < 1 or nodes < 1:
        raise ValueError("Quadrature requires positive class and node counts")
    shape = np.zeros((len(counts), classes * nodes, 4))
    inventories = np.column_stack(
        [counts.sum(1), (masses * counts[..., None]).sum(1)]
    )
    for b, (mass, count) in enumerate(zip(masses, counts, strict=True)):
        active = count > 0
        if not active.any():
            continue
        mass, count = mass[active], count[active]
        dry = mass[:, 1:].sum(1)
        if (dry <= 0).any():
            raise ValueError(
                "Each active particle needs dry material for size quadrature"
            )
        # Equal dry densities make the cube root of dry mass a size coordinate.
        groups = classify(mass[:, 2] / dry, count, classes)
        for g in np.unique(groups):
            keep = groups == g
            size, w = size_quadrature(np.cbrt(dry[keep]), count[keep], nodes)
            amount = (count[keep, None] * mass[keep]).sum(0)
            # Model water and dry species as proportional to volume within
            # each class. This approximation can lose within-class correlations.
            weighted_volume = w * size**3
            part = shape[b, g * nodes : g * nodes + len(w)]
            part[:, 0] = w
            part[:, 1:] = (weighted_volume / weighted_volume.sum())[
                :, None
            ] * amount
    return np.divide(
        shape,
        inventories[:, None],
        out=np.zeros_like(shape),
        where=inventories[:, None] > 0,
    ).astype(np.float32), inventories


def composition_profile(fraction, water_ratio, weight, classes):
    """Resolve conditional number, water and both dry materials at one size."""
    order = np.argsort(fraction)
    _, inv = np.unique(np.round(fraction[order], 12), return_inverse=True)
    w = np.bincount(inv, weights=weight[order])
    x = np.bincount(inv, weights=weight[order] * fraction[order]) / w
    wr = np.bincount(inv, weights=water_ratio[order] * weight[order]) / w
    f, n = size_quadrature(x, w, classes)
    n /= n.sum()
    profile = np.zeros((classes, 4))
    ratio = np.interp(f, x, wr)
    profile[: len(n)] = n[:, None] * np.column_stack(
        [np.ones_like(f), ratio, 1 - f, f]
    )
    return profile


def conditional_record(masses, counts, classes, nodes):
    """Resolve a separate composition distribution at each dry-size node."""
    if classes < 1 or nodes < 1:
        raise ValueError("Positive quadrature dimensions are required")
    shape = np.zeros((len(counts), nodes * classes, 4))
    occupied = counts > 0
    totals = np.column_stack(
        [counts.sum(1), (masses * counts[..., None]).sum(1)]
    )
    for b in range(len(counts)):
        mass = masses[b, occupied[b]]
        w = counts[b, occupied[b]]
        if not len(w):
            continue
        dry = mass[:, 1:].sum(1)
        if (dry <= 0).any():
            raise ValueError("Each occupied particle requires a dry size")
        axis = np.cbrt(dry)
        # Roundoff in the sum of the two dry species can separate equal sizes.
        _, group = np.unique(np.round(np.log(axis), 12), return_inverse=True)
        count = np.bincount(group, weights=w)
        size = np.bincount(group, weights=axis * w) / count
        location, n = size_quadrature(size, count, nodes)
        profiles = []
        for i in range(len(size)):
            ids = group == i
            profiles.append(
                composition_profile(
                    mass[ids, 2] / dry[ids],
                    mass[ids, 0] / dry[ids],
                    w[ids],
                    classes,
                )
            )
        prof = np.stack(profiles)
        # Linear interpolation of nonnegative weighted shares preserves number
        # and prevents division by a vanishing composition class.
        left = np.maximum(0, np.searchsorted(size, location, side="right") - 1)
        right = np.minimum(len(size) - 1, left + 1)
        f = np.divide(
            location - size[left],
            size[right] - size[left],
            out=np.zeros_like(location),
            where=right != left,
        ).clip(0, 1)
        cond = (1 - f[:, None, None]) * prof[left] + f[:, None, None] * prof[
            right
        ]
        cond *= n[:, None, None]
        cond[..., 1:] *= location[:, None, None] ** 3
        m = len(n) * classes
        shape[b, :m] = cond.reshape(-1, 4)
    sums = shape.sum(1, keepdims=True)
    if np.any((totals > 0) & (sums[:, 0] <= 0)):
        raise FloatingPointError("Quadrature lost a nonempty material")
    return np.divide(
        shape, sums, out=np.zeros_like(shape), where=sums > 0
    ).astype(np.float32), totals


def restore_arrays(shape, totals):
    """Recover represented counts and per-particle species masses."""
    weights = shape.astype(np.float64)
    denom = weights.sum(axis=1, keepdims=True)
    contained = weights / np.where(denom > 0, denom, 1.0) * totals[:, None, :]
    counts = contained[..., 0]
    means = np.divide(
        contained[..., 1:],
        counts[..., None],
        out=np.zeros_like(contained[..., 1:]),
        where=counts[..., None] > 0,
    )
    np.testing.assert_allclose(counts.sum(-1), totals[:, 0], rtol=1e-14)
    np.testing.assert_allclose(
        (counts[..., None] * means).sum(1),
        totals[:, 1:],
        rtol=1e-14,
        atol=1e-35,
    )
    return counts, means
