"""Modal quadrature with explicit number and donor-volume moments."""

from pathlib import Path

import numpy as np
from particula.dynamics.coagulation.brownian_kernel import (
    get_brownian_kernel_via_system_state,
)
from scipy.integrate import solve_ivp
from scipy.special import ndtr, roots_hermitenorm
from scripts.mam4xx_coagulation.particles import (
    DENSITY,
    NUMBER,
    PRESSURE,
    TEMPERATURE,
    WIDTH,
    cpu_initial_rate,
)


def matched(diameter, order, modes=None, temperature=TEMPERATURE, factor=1.0):
    """Advance two lognormals, assigning mixed collisions to accumulation."""
    x, w = roots_hermitenorm(order)
    w = w / np.sqrt(2 * np.pi)
    if modes is None:
        width = WIDTH
        n0 = NUMBER * np.array([0.8, 0.2])
        diameters = np.array([26.0, diameter]) * 1e-9
    else:
        diameters, width, n_cm3 = np.asarray(modes[:2]).T
        diameters = diameters * 1e-6
        n0 = n_cm3 * 1e6 * factor
    unit = np.exp(np.log(width)[:, None] * x)
    v0 = n0 * np.pi / 6 * diameters**3 * np.exp(4.5 * np.log(width) ** 2)

    def rates(_, y):
        n = n0 * y[:2]
        volumes = v0 * y[2:]
        d = (
            np.cbrt(
                6 * volumes / (np.pi * n * np.exp(4.5 * np.log(width) ** 2))
            )[:, None]
            * unit
        )
        vol = np.pi * d.ravel() ** 3 / 6
        k = get_brownian_kernel_via_system_state(
            d.ravel() / 2, vol * DENSITY, temperature, PRESSURE
        )
        cross = k[:order, order:]
        inter = n.prod() * (w @ cross @ w)
        transfer = n.prod() * ((w * vol[:order]) @ cross @ w)
        dn = [
            -0.5 * n[0] ** 2 * (w @ k[:order, :order] @ w) - inter,
            -0.5 * n[1] ** 2 * (w @ k[order:, order:] @ w),
        ]
        return np.r_[np.asarray(dn) / n0, [-transfer / v0[0], transfer / v0[1]]]

    initial = rates(0, np.ones(4))
    times = np.array([0, 3600, 21600, 86400])
    sol = solve_ivp(
        rates, (0, times[-1]), np.ones(4), t_eval=times, rtol=2e-9, atol=1e-12
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    volume = v0[:, None] * sol.y[2:]
    np.testing.assert_allclose(volume.sum(axis=0) / v0.sum(), 1.0, rtol=2e-12)
    loss = -initial[:2] @ n0
    if modes is None:
        np.testing.assert_allclose(
            loss, cpu_initial_rate(diameter, 4.0, 32), rtol=2e-7
        )
    # Independent cross-mode expectation using a donor volume-weighted
    # lognormal (its log-median shifts by three log-variance units).
    d0 = diameters[:, None] * unit
    d0[0] *= np.exp(3 * np.log(width[0]) ** 2)
    d = d0.ravel()
    k = get_brownian_kernel_via_system_state(
        d / 2, DENSITY * np.pi * d**3 / 6, temperature, PRESSURE
    )
    np.testing.assert_allclose(
        -initial[2] * v0[0],
        v0[0] * n0[1] * (w @ k[:order, order:] @ w),
        rtol=1e-6,
    )
    n = n0[:, None] * sol.y[:2]
    median = np.cbrt(
        6 * volume / (np.pi * n * np.exp(4.5 * np.log(width[:, None]) ** 2))
    )
    frac = ndtr(np.log(100e-9 / median) / np.log(width[:, None]))
    volume_below = ndtr(
        np.log(100e-9 / median) / np.log(width[:, None])
        - 3 * np.log(width[:, None])
    )
    return dict(
        uf_cm3=((n * frac).sum(axis=0) / 1e6).tolist(),
        above_cm3=((n * (1 - frac)).sum(axis=0) / 1e6).tolist(),
        volume_above_100_um3_cm3=(
            (volume * (1 - volume_below)).sum(axis=0) * 1e12
        ).tolist(),
        diameter=diameter,
        order=order,
        time_s=times.tolist(),
        number_cm3=((n0[:, None] * sol.y[:2]).T / 1e6).tolist(),
        volume_fraction=(volume.T / v0.sum()).tolist(),
        initial_number_loss_cm3_s=loss / 1e6,
        initial_donor_volume_m3_m3_s=-initial[2] * v0[0],
    )


def anchors():
    """Regenerate neutral modal comparisons before varying mechanisms."""
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/data/mam4xx-coagulation/modal.csv"
    )
    archive = np.genfromtxt(path, delimiter=",", names=True)
    output = []
    for d in (60.0, 220.0):
        fine = matched(d, 32)
        coarse = matched(d, 24)
        np.testing.assert_allclose(
            coarse["number_cm3"], fine["number_cm3"], rtol=2e-7
        )
        row = archive[
            (archive["acc_nm"] == d)
            & (archive["ratio"] == 4)
            & (archive["dt_s"] == 10)
            & (archive["time_s"] == 3600)
        ][0]
        fine["archived_1h_cm3"] = (row["aitken_m3"] + row["accum_m3"]) / 1e6
        output.append(fine)
    return output
