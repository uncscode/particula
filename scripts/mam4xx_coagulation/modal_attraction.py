"""Neutral capture integrated over evolving MAM4xx lognormal modes."""

from functools import lru_cache

import numpy as np
import warp as wp
from particula import particles
from particula.dynamics import get_hard_sphere_kernel_via_system_state
from particula.dynamics.coagulation.van_der_waals_kernel import (
    MAX_RADIUS_RATIO,
    get_van_der_waals_kernel_via_system_state,
)
from particula.gpu.dynamics.van_der_waals import prepare_capture
from scipy.integrate import solve_ivp
from scipy.special import ndtr
from scripts.mam4xx_coagulation.regime_box import MODES
from scripts.mam4xx_coagulation.regime_rates import RHO, P, capture

TEMP = 298.15
HAMAKER = 8.9e-20
HOURS = np.array([0, 1, 6, 24])
HIST_EDGES = 100 * 10 ** (np.arange(-16, 25) / 8)


def distribution(regime, widths, coarse):
    """Conserve published number and third moment under a width change."""
    dg, sigma, n = np.asarray(MODES[regime][: 3 if coarse else 2]).T
    if widths == "host":
        fixed = np.array([1.6, 1.8, 1.8])[: len(n)]
        dg = dg * np.exp(1.5 * (np.log(sigma) ** 2 - np.log(fixed) ** 2))
        sigma = fixed
    return np.column_stack([dg, sigma, n])


def capture_values(ri, rj, mi, mj, diffusion, hamaker):
    """Return full neutral GPU pair kernels in cm³ s⁻¹."""
    zero = np.zeros_like(ri)
    args = np.column_stack(
        [ri, rj, mi, mj, zero, zero, diffusion, np.full_like(ri, TEMP)]
    )
    device = "cuda:0"
    out = wp.empty(len(args), dtype=wp.float64, device=device)
    wp.launch(
        capture,
        len(args),
        [
            prepare_capture(hamaker, 5.9, device),
            wp.array(args, dtype=wp.float64, device=device),
            out,
        ],
        device=device,
    )
    values = out.numpy() * 1e6
    if not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("Unqualified neutral capture pair")
    return values


@lru_cache(maxsize=12)
def kernel_grid(per_decade, hamaker):
    """Build complete pair laws on cell midpoints, with exact support mask."""
    edges = 10 ** (np.arange(-2 * per_decade, 7 * per_decade + 1) / per_decade)
    radius = np.sqrt(edges[:-1] * edges[1:]) * 0.5e-9
    mass = 4 * np.pi / 3 * RHO * radius**3
    zero = np.zeros_like(radius)
    bare = (
        get_hard_sphere_kernel_via_system_state(radius, mass, zero, TEMP, P)
        * 1e6
    )
    diffusion = particles.get_diffusion_coefficient_via_system_state(
        radius, TEMP, P
    )
    i, j = np.triu_indices(len(radius))
    ok = radius[j] / radius[i] <= MAX_RADIUS_RATIO
    i, j = i[ok], j[ok]
    vals = capture_values(
        radius[i],
        radius[j],
        mass[i],
        mass[j],
        diffusion[i] + diffusion[j],
        hamaker,
    )
    full = bare.copy()
    full[i, j] = full[j, i] = vals
    outside = (
        np.maximum.outer(radius, radius) / np.minimum.outer(radius, radius)
        > MAX_RADIUS_RATIO
    )
    # Independently verify supported equal and unequal pairs on the CPU.
    take = np.array([3 * per_decade, 4 * per_decade])
    cpu = get_van_der_waals_kernel_via_system_state(
        radius[take], mass[take], zero[take], TEMP, P, hamaker, 5.9
    )
    np.testing.assert_allclose(full[np.ix_(take, take)], cpu * 1e6, rtol=0.012)
    return edges, bare, full, outside


def fractions(edges, median, log_sigma):
    """Integrate a number or volume lognormal over each finite cell."""
    z = np.log(edges[None, :] / median[:, None]) / log_sigma[:, None]
    return np.where(z[:, :-1] >= 0, -np.diff(ndtr(-z)), np.diff(ndtr(z)))


def solve_modes(modes, kernel, edges, outside, *, rtol=2e-8):
    """Retain products in the receiving mode, transferring donor volume."""
    dg, sigma, start_n = modes.T
    logs = np.log(sigma)
    start_v = np.pi / 6 * dg**3 * np.exp(4.5 * logs**2) * start_n
    count = len(start_n)
    donor, receiver = np.triu_indices(count, 1)

    def state(y):
        n, v = y[:count] * start_n, y[count:] * start_v
        median = 1e3 * np.cbrt(6 * v / (np.pi * n * np.exp(4.5 * logs**2)))
        return n, v, median

    def moments(y, rates):
        n, v, median = state(y)
        number = fractions(edges, median, logs)
        volume = fractions(edges, median * np.exp(3 * logs**2), logs)
        for weights in (number, volume):
            np.testing.assert_allclose(
                weights.sum(axis=1), 1.0, atol=1e-10, rtol=0
            )
        events = ((n[:, None] * number) @ rates) @ (n[:, None] * number).T
        transfers = (((v[:, None] * volume) @ rates) @ (n[:, None] * number).T)[
            donor, receiver
        ]
        return events, transfers

    def rhs(_, y):
        events, transfers = moments(y, kernel)
        dn = -0.5 * np.diag(events) - np.triu(events, 1).sum(axis=1)
        dv = np.bincount(receiver, transfers, minlength=count) - np.bincount(
            donor, transfers, minlength=count
        )
        return np.r_[dn / start_n, dv / start_v]

    sol = solve_ivp(
        rhs,
        (0, 86400),
        np.ones(2 * count),
        t_eval=3600 * HOURS,
        rtol=rtol,
        atol=rtol * 0.001,
    )
    if not sol.success or np.any(sol.y <= 0):
        raise ValueError(sol.message)
    result = {
        k: []
        for k in (
            "mode_n",
            "mode_v",
            "uf_cm3",
            "above_cm3",
            "volume_above_100_um3_cm3",
            "spectrum_cm3",
            "unsupported_loss_fraction",
            "unsupported_transfer_fraction",
            "transfer_um3_cm3_s",
        )
    }
    for y in sol.y.T:
        n, v, median = state(y)
        np.testing.assert_allclose(v.sum(), start_v.sum(), rtol=5e-10)
        if np.sum(n) > start_n.sum() * (1 + 1e-12):
            raise ValueError("Coagulation increased number")
        cut = np.log(100 / median) / logs
        h = fractions(HIST_EDGES, median, logs)
        under = ndtr(np.log(HIST_EDGES[0] / median) / logs)
        over = ndtr(-np.log(HIST_EDGES[-1] / median) / logs)
        hist = n @ np.column_stack([under, h, over])
        np.testing.assert_allclose(hist.sum(), n.sum())
        events, transfer = moments(y, kernel)
        tail, tail_transfer = moments(y, kernel * outside)
        for key, value in {
            "mode_n": n,
            "mode_v": v,
            "uf_cm3": n @ ndtr(cut),
            "above_cm3": n @ ndtr(-cut),
            "volume_above_100_um3_cm3": v @ ndtr(3 * logs - cut),
            "spectrum_cm3": hist,
            "unsupported_loss_fraction": tail.sum() / events.sum(),
            "unsupported_transfer_fraction": tail_transfer / transfer,
            "transfer_um3_cm3_s": transfer,
        }.items():
            result[key].append(np.asarray(value).tolist())
    return result
