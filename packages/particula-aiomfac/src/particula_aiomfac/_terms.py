"""Activity terms translated from AIOMFAC-web v3.15, dataset 1.

ModSRunifac and ModMRpart, Andi Zuend and contributors. The first axis in
each composition array indexes independent mixtures. Molar masses use kg/mol.
"""

import numpy as np


def short_range(p, x, temperature):
    """Modified UNIFAC residual and combinatorial terms at one temperature."""
    psi = np.exp(-p.interaction / temperature)

    def group_residual(groups):
        surface = groups * p.q
        theta = surface / surface.sum(axis=1, keepdims=True)
        denom = theta @ psi
        return p.q * (1 - np.log(denom) - (theta / denom) @ psi.T)

    residual = group_residual(x @ p.counts) @ p.counts.T
    neutral_counts = p.counts[: len(p.neutrals)]
    residual[:, : len(neutral_counts)] -= (
        neutral_counts * group_residual(neutral_counts)
    ).sum(axis=1)
    volume = p.volume / (x @ p.volume)[:, None]
    ratio = volume * (x @ p.area)[:, None] / p.area
    comb = (
        np.log(volume) + 1 - volume - 5 * p.area * (np.log(ratio) + 1 - ratio)
    )
    # Subtract the combinatorial ionic reference at infinite dilution in pure water.
    v, q = p.volume[len(neutral_counts) :], p.area[len(neutral_counts) :]
    comb[:, len(neutral_counts) :] -= (
        np.log(v / 0.92)
        + 5 * q * np.log(q * 0.92 / (1.40 * v))
        + 5 * (v - q)
        - v
        + 1
        + v / 0.92 * 2.32
    )
    return comb + residual


def _interaction(first, second, decay, root):
    """B(I) or C(I) and its first derivative with respect to ionic strength."""
    argument = decay * root[:, None, None]
    change = second * np.exp(-np.minimum(argument, 301)) * (argument <= 300)
    return first + change, -0.5 / root[:, None, None] * decay * change


def electrolyte_terms(p, xneutral, molal, temperature):
    """Return MR, LR, ionic strength and the ionic molality-scale conversion."""
    strength = 0.5 * (molal @ (p.charge**2))
    root = np.sqrt(strength)
    denom = np.sqrt(78.54 * temperature)
    a = 1.327757e5 * np.sqrt(997.0) / denom**3
    b = 6.359696 * np.sqrt(997.0) / denom
    factor = 1 + b * root
    lr = np.column_stack(
        (
            (2 * a / b**3 * (factor - 1 / factor - 2 * np.log(factor)))[:, None]
            * p.neutral_mw,
            -(a * root / factor)[:, None] * p.charge**2,
        )
    )
    mr = np.zeros_like(lr)
    scale = np.zeros_like(strength)
    active = strength > np.finfo(float).eps
    if active.any():
        mr[active] = _middle_range(
            p, xneutral[active], molal[active], strength[active]
        )
        scale[active] = np.log(
            0.01801528
            * (
                1 / (xneutral[active] @ p.neutral_mw)
                + molal[active].sum(axis=1)
            )
        )
    return mr, lr, strength, scale


def _middle_range(p, solvent_x, molal, strength):
    # n indexes independent states, g solvent main groups, i and j ions.
    groups = solvent_x @ p.neutral_counts
    groups /= groups.sum(axis=1, keepdims=True)
    main = groups @ p.group_map
    weighted_mass = (groups * p.group_mw) @ p.group_map
    mw = np.divide(weighted_mass, main, out=np.zeros_like(main), where=main > 0)
    average = weighted_mass.sum(axis=1)
    mc, ma = np.split(molal, [p.ncation], axis=1)
    pairs = mc[:, :, None] * ma[:, None, :]
    z = molal @ np.abs(p.charge)
    root = np.sqrt(strength)
    bn, dbn = _interaction(p.bgroup, p.cgroup, 1.2, root)
    bc, dbc = _interaction(*p.bpair, root)
    cc, dcc = _interaction(*p.cpair, root)
    solvent_deriv = np.einsum("ng,ngi,ni->n", main, dbn, molal)
    pair_deriv = np.sum((dbc + z[:, None, None] * dcc) * pairs, axis=(1, 2))
    pair_c = np.sum(cc * pairs, axis=(1, 2))
    # Each unordered pair of like cations occurs once in the neutral term.
    ccpair = np.triu(mc[:, :, None] * mc[:, None, :])
    like = np.einsum("nij,ij->n", ccpair, p.like)
    three = np.einsum("nij,ijk,nk->n", ccpair, p.three, ma)
    group_ion = np.einsum("ngi,ni->ng", bn, molal)
    group_mr = (
        group_ion
        - mw
        * (
            ((main * group_ion).sum(axis=1) + strength * solvent_deriv)
            / average
        )[:, None]
        - mw
        * (
            np.sum(bc * pairs, axis=(1, 2))
            + strength * pair_deriv
            + 2 * z * pair_c
            + like
            + 2 * three
        )[:, None]
    )
    matrix = bc + z[:, None, None] * cc
    cation = (
        np.einsum("nij,nj->ni", matrix, ma)
        + mc @ p.like.T
        + np.einsum("ijk,nj,nk->ni", p.three, mc, ma)
    )
    anion = np.einsum("ni,nij->nj", mc, matrix) + np.einsum(
        "nij,ijk->nk", ccpair, p.three
    )
    ion_mr = (
        np.einsum("ng,ngi->ni", main, bn) / average[:, None]
        + 0.5 * p.charge**2 * (solvent_deriv / average + pair_deriv)[:, None]
        + abs(p.charge) * pair_c[:, None]
        + np.column_stack((cation, anion))
    )
    return np.column_stack((group_mr @ p.main_counts.T, ion_mr))
