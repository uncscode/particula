"""Reconstruct a 3-by-3 volatile mass table from pooled mixed moments."""

import numpy as np
from dual_joint_record import conditional_rho
from moment_quadrature import decode, encode, size_nodes
from quadrature import restore_arrays

SUPPORT = np.array([-1.0, 0.0, 1.0])
POLY = np.stack((SUPPORT, SUPPORT**2 - 2 / 3), -1)
DUAL = POLY @ np.linalg.inv(POLY.T @ POLY)
TERMS = {
    "marginal": (),
    "covariance": ((0, 0),),
    "third": ((0, 0), (1, 0), (0, 1)),
    "fourth": ((0, 0), (1, 0), (0, 1), (1, 1)),
}


def joint_probabilities(moments):
    """Invert centered products and preserve positive marginal probabilities."""
    delta = DUAL @ moments @ DUAL.T
    factor = np.minimum(
        1, (1 / 9) / np.maximum(-delta.min(axis=(-2, -1)), 1e-30)
    )
    prob = np.maximum(0, 1 / 9 + factor[..., None, None] * delta)
    np.testing.assert_allclose(prob.sum(-1), 1 / 3, atol=1e-13)
    np.testing.assert_allclose(prob.sum(-2), 1 / 3, atol=1e-13)
    return prob, factor


def encode_joint(m, n, k, detail, conditioning):
    """Pool mixed volatile products by dry size and optional dry composition."""
    core, total = encode(m[:, :, [0, 2, 3]], n, k)
    dry = m[..., 2:].sum(-1)
    w = n / n.sum(1, keepdims=True)
    org = (m[..., 1] * n).sum(-1)
    means = np.column_stack((total[:, 1], org)) / total[:, 2:].sum(
        1, keepdims=True
    )
    ratios = m[..., :2] / dry[..., None]
    dev = ratios - means[:, None]
    span = np.sqrt(1.5 * (w[..., None] * dev**2).sum(1))
    z = np.divide(
        dev, span[:, None], out=np.zeros_like(dev), where=span[:, None] > 0
    )
    y = np.stack((z, z**2 - 2 / 3), -1)
    f = m[..., 3] / dry
    u = np.cbrt(dry / (w * dry).sum(1, keepdims=True)) - 1
    size = u[..., None] ** np.arange(1 if conditioning == "global" else k)
    factor = (
        np.stack((np.ones_like(f), f), -1)
        if conditioning == "joint"
        else np.ones_like(f[..., None])
    )
    values = []
    for a, b in TERMS[detail]:
        v = y[..., 0, a] * y[..., 1, b]
        values.append(
            np.einsum("bi,bij,biv,bi->bjv", w, size, factor, v, optimize=True)
        )
    mixed = (
        np.stack(values, 1)
        if values
        else np.empty((len(m), 0, size.shape[-1], factor.shape[-1]))
    )
    mixed[abs(mixed) < 5e-13] = 0
    return dict(
        shape=core.astype(np.float32),
        totals=total,
        organic=org,
        spread=span.astype(np.float32),
        mixed=mixed.astype(np.float32),
        nodes=k,
        detail=detail,
        conditioning=conditioning,
    )


def unpack_moments(raw, shape, num, f, k, terms, conditioning):
    """Recover local conditional mixed moments on distinct diameter nodes."""
    moments = np.zeros((*num.shape, 2, 2))
    for i, code in enumerate(shape):
        u, w = size_nodes(np.r_[1.0, code[: 2 * k - 1]].astype(float))
        stop = 2 * len(w)
        for j, (a, b) in enumerate(terms):
            if conditioning == "global":
                moments[i, :, a, b] = raw[i, j, 0, 0]
            else:
                moments[i, :stop, a, b] = conditional_rho(
                    raw[i, j],
                    u,
                    w,
                    f[i, :stop].reshape(-1, 2),
                    num[i, :stop].reshape(-1, 2),
                ).ravel()
    return moments


def decode_joint(v):
    """Restore particles and audits for both volatile and dry inventories."""
    k = int(v["nodes"])
    total = v["totals"]
    shape = v["shape"]
    split, _ = decode(shape, total, k)
    n, core = restore_arrays(split, total)
    dry = core[..., 1:].sum(-1)
    f = np.divide(core[..., 2], dry, out=np.zeros_like(dry), where=dry > 0)
    moments = unpack_moments(
        v["mixed"],
        shape,
        n,
        f,
        k,
        TERMS[str(v["detail"])],
        str(v["conditioning"]),
    )
    prob, scale = joint_probabilities(moments)
    means = np.column_stack((total[:, 1], v["organic"])) / total[:, 2:].sum(
        1, keepdims=True
    )
    signs = np.stack(np.meshgrid(SUPPORT, SUPPORT, indexing="ij"), -1).reshape(
        -1, 2
    )
    loads = means[:, None] + v["spread"][:, None] * signs
    if np.min(loads) < -1e-8:
        raise ValueError("Ternary levels violate nonnegativity")
    wet = dry[:, :, None, None] * loads[:, None].clip(0)
    m = np.concatenate(
        (wet, np.broadcast_to(core[:, :, None, 1:], wet.shape)), -1
    ).reshape(len(n), -1, 4)
    counts = (n[..., None, None] * prob).reshape(len(n), -1)
    exact = np.column_stack((total[:, 1], v["organic"], total[:, 2:]))
    np.testing.assert_allclose(
        (m * counts[..., None]).sum(1), exact, rtol=2e-13
    )
    np.testing.assert_allclose(counts.sum(1), total[:, 0], rtol=2e-13)
    return m, counts, float(scale.min())
