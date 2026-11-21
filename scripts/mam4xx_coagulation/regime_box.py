"""Sectional atmospheric aging with bipolar ion balance."""

import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
from scipy.special import ndtr
from scripts.mam4xx_coagulation.regime_rates import tables

# Lance et al. (2004) Table 1 lists geometric mean DIAMETERS in micrometres.
MODES = {
    "marine": [(0.018, 1.4, 310.0), (0.075, 1.6, 70.0), (0.58, 2.49, 3.1)],
    "continental": [
        (0.016, 1.6, 1000.0),
        (0.067, 2.1, 800.0),
        (0.93, 2.2, 0.72),
    ],
    "polluted": [
        (0.014, 1.8, 106000.0),
        (0.054, 2.16, 32000.0),
        (0.86, 2.21, 5.4),
    ],
}


def initial(edges, regime, coarse, factor):
    """Integrate the observed modal number distribution on common edges."""
    n = np.zeros(len(edges) - 1)
    for diameter, sigma, count in MODES[regime][: 3 if coarse else 2]:
        n += (
            count
            * factor
            * np.diff(ndtr(np.log(edges / (1000 * diameter)) / np.log(sigma)))
        )
    return n


def select_kernel(mechanism, prob, bare, full, brown, settle, shear, epsilon):
    """Select a complete thermal law and its specified mechanical addition."""
    m, width = prob.shape
    neutral = np.arange(width // 2, m * width, width)
    model = full if mechanism in ("dispersion", "combined") else bare
    if mechanism in ("fine", "brown", "settling", "shear"):
        kernel = brown.copy()
        if mechanism == "settling":
            kernel += settle
        if mechanism == "shear":
            kernel += shear * np.sqrt(epsilon / 0.01)
    elif mechanism in ("neutral", "dispersion"):
        kernel = model[np.ix_(neutral, neutral)]
    elif mechanism == "dynamic":
        kernel = bare
    else:
        kernel = np.einsum(
            "ia,iajb,jb->ij",
            prob,
            model.reshape(m, width, m, width),
            prob,
            optimize=True,
        )
    return kernel


def evolve(
    regime,
    mechanism,
    per_decade=8,
    cap=3,
    temp=298.15,
    hamaker=8.9e-20,
    factor=1.0,
    source=3.0,
    dielectric=5.9,
    epsilon=0.01,
    rtol=2e-6,
):
    """Compute endpoint spectra, volume redistribution and integral budgets."""
    start = time.monotonic()
    edge, d, q, plus, minus, prob, bare, full, support, brown, settle, shear = (
        tables(per_decade, cap, temp, hamaker, dielectric)
    )
    dynamic = mechanism == "dynamic"
    n = initial(edge, regime, mechanism != "fine", factor)
    m = len(d)
    width = len(q)
    neutral = np.arange(cap, m * width, width)
    kernel = select_kernel(
        mechanism, prob, bare, full, brown, settle, shear, epsilon
    )
    if mechanism == "diagnostic":
        zero = np.zeros_like(prob)
        zero[:, cap] = 1.0
        thermal = select_kernel(
            mechanism, zero, bare, full, brown, settle, shear, epsilon
        )
        np.testing.assert_allclose(thermal, bare[np.ix_(neutral, neutral)])
    nq = width if dynamic else 1
    size = m * nq
    di = np.repeat(d, nq)
    zi = np.tile(q, m) if dynamic else np.zeros(m, dtype=int)
    volume = di**3
    i, j = np.triu_indices(size)
    merged = volume[i] + volume[j]
    k = np.searchsorted(d**3, merged) - 1
    charge = zi[i] + zi[j]
    keep = (k < m - 1) & (abs(charge) <= cap)
    i, j, v, k, charge = (a[keep] for a in (i, j, merged, k, charge))
    high = (v - d[k] ** 3) / (d[k + 1] ** 3 - d[k] ** 3)
    destination = k * nq + (charge + cap if dynamic else 0)
    left, right = i, j
    rate = kernel[i, j] * np.where(i == j, 0.5, 1.0)
    events = np.arange(len(rate))
    rows = [i, j, destination, destination + nq]
    cols = [events] * 4
    vals = [-np.ones_like(high), -np.ones_like(high), 1 - high, high]
    if dynamic:
        # Charge addition, one-for-one ion consumption, and ion recombination.
        for sign, coeff, ion in [(1, plus, size), (-1, minus, size + 1)]:
            f = np.flatnonzero(abs(zi + sign) <= cap)
            idx = np.arange(len(rate), len(rate) + len(f))
            rows.extend([f, f + sign, np.full(len(f), ion)])
            cols.extend([idx] * 3)
            vals.extend([-np.ones(len(f)), np.ones(len(f)), -np.ones(len(f))])
            left = np.r_[left, f]
            right = np.r_[right, np.full(len(f), ion)]
            rate = np.r_[rate, coeff.ravel()[f]]
        rows.append(np.array([size, size + 1]))
        cols.append(np.full(2, len(rate)))
        vals.append(-np.ones(2))
        left = np.r_[left, size]
        right = np.r_[right, size + 1]
        rate = np.r_[rate, 1.6e-6]
    dim = size + 2 * dynamic
    s = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, len(rate)),
    ).tocsr()
    # Scale each reaction column once. Using sparse stoichiometry preserves the
    # volume and charge of every admitted collision, including equal size bins.
    changes = s.multiply(rate).tocsr()
    np.testing.assert_allclose(
        np.r_[zi, [1, -1] if dynamic else []] @ s, 0, atol=3e-14
    )
    y0 = n.copy()
    forcing = np.zeros(dim)
    if dynamic:
        y0 = np.zeros(dim)
        y0[neutral] = n
        y0[-2:] = np.sqrt(source / 1.6e-6)
        forcing[-2:] = source

    def rhs(_, y):
        return changes @ (y[left] * y[right]) + forcing

    def jac(_, y):
        ids = np.arange(len(left))
        deriv = coo_matrix(
            (np.r_[y[right], y[left]], (np.r_[ids, ids], np.r_[left, right])),
            shape=(len(ids), dim),
        ).tocsr()
        return changes @ deriv

    result = solve_ivp(
        rhs,
        (0.0, 86400.0),
        y0,
        method="BDF",
        jac=jac,
        rtol=rtol,
        atol=1e-10,
        t_eval=[0.0, 3600.0, 21600.0, 86400.0],
    )
    if not result.success or result.y.min() < -5e-8:
        raise ValueError(
            f"Integration failed: {result.message}, {result.y.min()}"
        )
    states = result.y[:size].T.reshape(-1, m, nq)
    series = states.sum(axis=2)
    vol = series @ (d**3)
    np.testing.assert_allclose(vol / vol[0], 1.0, rtol=1e-8)
    charge_residual = 0.0
    if dynamic:
        charge_residual = float(
            np.max(abs((states @ q).sum(axis=1) + result.y[-2] - result.y[-1]))
        )
        if charge_residual > 2e-6:
            raise ValueError("Charge balance failed")
    uf = series[:, d < 100].sum(axis=1)
    above = series[:, d >= 100].sum(axis=1)
    vlarge = (
        series[:, d >= 100] @ (d[d >= 100] ** 3) * np.pi / 6 * 1e-9
    )  # µm³/cm³
    start_states = (n[:, None] * prob).ravel()
    flux = full * start_states[:, None] * start_states[None, :]
    return dict(
        regime=regime,
        mechanism=mechanism,
        diameter_nm=d.tolist(),
        spectrum_cm3=series.tolist(),
        uf_cm3=uf.tolist(),
        above_cm3=above.tolist(),
        volume_above_100_um3_cm3=vlarge.tolist(),
        budget=float(np.max(abs(vol / vol[0] - 1))),
        charge_residual=charge_residual,
        free_each_cm3=result.y[-2:].T.tolist() if dynamic else [],
        initial_tail_collision_fraction=float(
            flux[~support].sum() / flux.sum()
        ),
        elapsed=time.monotonic() - start,
    )
