"""Store marginal volatile spreads and dry-conditioned cross moments."""

import numpy as np
from moment_quadrature import decode, encode, size_nodes
from quadrature import restore_arrays


def record(mass, counts, nodes, mode):
    """Encode binary marginal loads and their optional joint correlation."""
    dry = mass[..., 2:].sum(-1)
    w = counts / counts.sum(1, keepdims=True)
    mean_dry = (w * dry).sum(1, keepdims=True)
    core, totals = encode(mass[:, :, [0, 2, 3]], counts, nodes)
    org = (counts * mass[..., 1]).sum(1)
    mu = np.column_stack((totals[:, 1], org)) / totals[:, 2:].sum(1)[:, None]
    ratio = mass[..., :2] / dry[..., None]
    dev = ratio - mu[:, None]
    sd = np.sqrt((w[..., None] * dev**2).sum(1))
    z = np.divide(
        dev, sd[:, None], out=np.zeros_like(dev), where=sd[:, None] > 0
    )
    product = z.prod(-1)
    u = np.cbrt(dry / mean_dry) - 1
    f = mass[..., 3] / dry
    order = {"marginal": 0, "global": 1, "size": nodes, "joint": nodes}[mode]
    powers = u[..., None] ** np.arange(order)
    vals = (
        np.stack((product, product * f), -1)
        if mode == "joint"
        else product[..., None]
    )
    cross = np.einsum("bi,bij,bik->bjk", w, powers, vals, optimize=True)
    # Enforce the algebraic zero at roundoff so the numerical sign cannot
    # encode the identity of a member with equal mathematical moments.
    cross[np.abs(cross) < 5e-13] = 0
    return dict(
        shape=core.astype(np.float32),
        totals=totals,
        organic=org,
        spread=sd.astype(np.float32),
        cross=cross.astype(np.float32),
        mode=mode,
        nodes=nodes,
    )


def conditional_rho(raw, u, p, frac, weights):
    """Invert size and composition moments for the standardized cross term."""
    cells = len(u)
    design = np.vander(u, cells, increasing=True).T
    joint = np.linalg.solve(design, raw[:cells]) / p[:, None]
    rho = np.repeat(joint[:, :1], 2, axis=1)
    if raw.shape[-1] == 2:
        prob = weights / weights.sum(1, keepdims=True)
        diff = frac[:, 1] - frac[:, 0]
        valid = (np.abs(diff) > 1e-7) & (prob.min(1) > 1e-7)
        hi = (joint[valid, 1] - frac[valid, 0] * joint[valid, 0]) / diff[valid]
        rho[valid, 1] = hi / prob[valid, 1]
        rho[valid, 0] = (joint[valid, 0] - hi) / prob[valid, 0]
    return rho


def reconstruct(data):
    """Decode to four joint cells per recovered dry-size/composition class."""
    shape, totals = data["shape"], data["totals"]
    k = int(data["nodes"])
    decoded, _ = decode(shape, totals, k)
    count, core = restore_arrays(decoded, totals)
    dry = core[..., 1:].sum(-1)
    f = np.divide(core[..., 2], dry, out=np.zeros_like(dry), where=dry > 0)
    mode = str(data["mode"])
    corr = np.zeros_like(count)
    if mode == "global":
        corr[:] = data["cross"][:, 0, :1]
    elif mode in ("size", "joint"):
        for i, code in enumerate(shape):
            u, p = size_nodes(np.r_[1.0, code[: 2 * k - 1]].astype(float))
            stop = 2 * len(p)
            corr[i, :stop] = conditional_rho(
                data["cross"][i],
                u,
                p,
                f[i, :stop].reshape(-1, 2),
                count[i, :stop].reshape(-1, 2),
            ).ravel()
    excess = np.maximum(0, abs(corr) - 1)
    corr = corr.clip(-1, 1)
    means = (
        np.column_stack((totals[:, 1], data["organic"]))
        / totals[:, 2:].sum(1)[:, None]
    )
    signs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    loads = means[:, None] + data["spread"][:, None] * signs
    if loads.min() < -1e-8:
        raise ValueError(
            "Binary marginal reconstruction requires positive loads"
        )
    wet = dry[:, :, None, None] * loads[:, None].clip(0)
    fixed = np.broadcast_to(core[:, :, None, 1:], wet.shape)
    mass = np.concatenate((wet, fixed), -1).reshape(len(count), -1, 4)
    weights = (
        count[..., None] * (1 + corr[..., None] * signs.prod(1)) / 4
    ).reshape(len(count), -1)
    expected = np.column_stack((totals[:, 1], data["organic"], totals[:, 2:]))
    np.testing.assert_allclose(
        (mass * weights[..., None]).sum(1), expected, rtol=2e-13
    )
    np.testing.assert_allclose(weights.sum(1), totals[:, 0], rtol=2e-13)
    return mass, weights, excess.max()
