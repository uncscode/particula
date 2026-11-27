"""Direct weighted joint moments with a size-conditioned quadrature decoder."""

import numpy as np


def encode(mass, counts, nodes):
    """Pool 7n−1 features and retain exact number and inventories."""
    if nodes < 2:
        raise ValueError("Use at least two dry size nodes")
    dry = mass[..., 1:].sum(-1)
    if np.any((counts > 0) & (dry <= 0)):
        raise ValueError("Each active particle requires dry material")
    n = counts.sum(1, keepdims=True)
    totals = np.concatenate([n, (mass * counts[..., None]).sum(1)], 1)
    dry_mean = totals[:, 2:].sum(1, keepdims=True) / np.where(n > 0, n, 1.0)
    u = np.cbrt(dry / np.where(dry_mean > 0, dry_mean, 1.0)) - 1
    weights = counts / np.where(n > 0, n, 1.0)
    f = mass[..., 2] / np.where(dry > 0, dry, 1.0)
    water = mass[..., 0] / np.where(dry > 0, dry, 1.0)
    powers = u[..., None] ** np.arange(2 * nodes)
    size = np.einsum("bi,bij->bj", weights, powers[..., 1:])
    variables = np.stack([f, f**2, f**3, water, water * f], -1)
    joint = np.einsum(
        "bi,bij,bik->bjk", weights, powers[..., :nodes], variables
    )
    return np.concatenate([size, joint.reshape(len(n), -1)], 1), totals


def size_nodes(moments):
    """Recover a positive Gaussian rule using a rank-revealing Hankel solve."""
    n = len(moments) // 2
    i = np.arange(n)
    h = moments[i[:, None] + i]
    shifted = moments[i[:, None] + i + 1]
    eig, vec = np.linalg.eigh(h)
    keep = eig > eig[-1] * 1e-12
    basis = vec[:, keep] / np.sqrt(eig[keep])
    jacobi = basis.T @ shifted @ basis
    u, vector = np.linalg.eigh((jacobi + jacobi.T) / 2)
    w = (h[0] @ basis @ vector) ** 2
    return u, w / w.sum()


def conditional_shares(raw):
    """Reconstruct a positive two-point composition rule and its water."""
    mean = np.clip(raw[0], 0, 1)
    variance = np.clip(raw[1] - mean**2, 0, mean * (1 - mean))
    water = max(raw[3], 0)
    if variance < 1e-13:
        result = np.array([[1.0, water, 1 - mean, mean], [0.0, 0.0, 0.0, 0.0]])
    else:
        skew = raw[2] - 3 * mean * raw[1] + 2 * mean**3
        center = np.clip(
            mean + skew / variance, variance / mean, 1 - variance / (1 - mean)
        )
        spread = np.sqrt(variance)
        f, vec = np.linalg.eigh([[mean, spread], [spread, center]])
        f = f.clip(0, 1)
        n = vec[0] ** 2
        # Water-fraction cross-moment apportions water to the two dry classes.
        high = np.clip((raw[4] - water * f[0]) / (f[1] - f[0]), 0, water)
        result = np.column_stack([n, [water - high, high], n * (1 - f), n * f])
    n, q, _, rich = result.T
    f = np.divide(rich, n, out=np.zeros_like(n), where=n > 0)
    recovered = np.array([n @ f, n @ (f**2), n @ (f**3), q.sum(), q @ f])
    adjusted = not np.allclose(recovered, raw, rtol=2e-6, atol=2e-8)
    return result, adjusted


def decode(code, totals, nodes):
    """Decode using only the pooled moments and the four exact inventories."""
    if code.shape != (len(totals), 7 * nodes - 1):
        raise ValueError("Unexpected number of stored moments")
    shape = np.zeros((len(totals), nodes * 2, 4))
    corrected = np.zeros(len(totals), int)
    for b, (x, inventory) in enumerate(
        zip(code.astype(np.float64), totals, strict=True)
    ):
        if inventory[0] == 0:
            continue
        split = 2 * nodes - 1
        u, n = size_nodes(np.r_[1.0, x[:split]])
        k = len(n)
        vander = np.vander(u, k, increasing=True).T
        joint = x[split:].reshape(nodes, 5)
        profile = np.linalg.solve(vander, joint[:k]) / n[:, None]
        for j in range(k):
            composition, adjusted = conditional_shares(profile[j])
            corrected[b] += adjusted
            composition[:, 1:] *= max(1 + u[j], 0) ** 3
            shape[b, 2 * j : 2 * j + 2] = composition * n[j]
    sums = shape.sum(1, keepdims=True)
    if np.any((totals > 0) & (sums[:, 0] <= 0)):
        raise ValueError("Decoded quadrature lost a nonzero inventory")
    return np.divide(
        shape, sums, out=np.zeros_like(shape), where=sums > 0
    ), corrected
