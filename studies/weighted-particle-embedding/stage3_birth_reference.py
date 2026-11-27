"""Continuous newborn growth by quadrature over cumulative vapor exposure."""

import argparse
from pathlib import Path

import numpy as np
from particula.gas import get_molecule_mean_free_path
from scipy.constants import Avogadro
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import brentq
from stage3_birth_growth import D0, MOLAR, RHO, write


def curve(initial, rate, law, end, du):
    """Solve finite-reservoir nucleation and growth on an exposure grid.

    u = integral C(t)/C0 dt. Radius depends only on exposure age. For
    activation the births per du are constant. For kinetic nucleation they
    scale with the fraction of the initial gas still present. Conservation
    gives a scalar Volterra equation for that fraction.
    """
    path = float(
        get_molecule_mean_free_path(
            molar_mass=MOLAR, temperature=293.0, pressure=101325.0
        )
    )
    age = np.linspace(0, end, int(np.ceil(end / du)) + 1)
    du = age[1]

    def rhs(_, y):
        r = y[0] * 1e-9
        kn = path / r
        beta = 0.75 * (1 + kn) / (kn * kn + 1.283 * kn + 0.75)
        return [1e9 * initial * 1e-5 * beta / (RHO * r)]

    sol = solve_ivp(
        rhs,
        (0, end),
        [D0 * 1e9 / 2],
        t_eval=age,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    if not sol.success:
        raise ArithmeticError(sol.message)
    r = sol.y[0]
    m = 4 * np.pi / 3 * RHO * (r * 1e-9) ** 3
    c0 = initial * Avogadro / MOLAR
    j0 = (
        rate * c0
        if law == "activation"
        else rate / (1e-9 * Avogadro / MOLAR) * c0 * c0
    )
    if initial == 0:
        return [
            dict(
                time_s=t,
                number_m3=0.0,
                above_5nm_m3=0.0,
                above_10nm_m3=0.0,
                gas_kg_m3=0.0,
                particle_kg_m3=0.0,
                largest_diameter_nm=0.0,
                first5_continuous_s=None,
                first10_continuous_s=None,
            )
            for t in np.arange(0, end + 1)
        ]
    scale = j0 / initial
    if law == "activation":
        fraction = 1 - scale * cumulative_trapezoid(m, age, initial=0)
    else:
        fraction = np.ones(len(age))
        for i in range(1, len(age)):
            carried = 0.5 * m[i] + (m[1:i] @ fraction[i - 1 : 0 : -1])
            fraction[i] = (1 - scale * du * carried) / (
                1 + 0.5 * scale * du * m[0]
            )
            if fraction[i] <= 0:
                fraction = fraction[:i]
                break
    positive = np.flatnonzero(fraction <= 0)
    if positive.size:
        fraction = fraction[: positive[0]]
    u = age[: len(fraction)]
    # The terminal zero-gas exposure is approached asymptotically. Adaptive
    # integration in physical time stays inside the tabulated positive range.
    last = len(u) - 1
    if last < len(age) - 1:
        tail_slope = (fraction[-1] - fraction[-2]) / du
        extend = u[-1] - fraction[-1] / tail_slope
        u = np.r_[u, extend]
        fraction = np.r_[fraction, 0.0]
    times = np.arange(0, end + 1.0)
    tsol = solve_ivp(
        lambda _, y: np.interp(y, u, fraction),
        (0, end),
        [0.0],
        t_eval=times,
        method="DOP853",
        atol=1e-11,
        rtol=2e-10,
        max_step=0.5,
        dense_output=True,
    )
    exposure = tsol.y[0]
    number = j0 * (
        u
        if law == "activation"
        else cumulative_trapezoid(fraction, u, initial=0)
    )
    critical = [
        np.interp(d, 2 * r, age) if 2 * r[-1] >= d else np.inf for d in [5, 10]
    ]
    arrival = [
        brentq(lambda t, threshold=x: tsol.sol(t)[0] - threshold, 0, end)
        if x <= exposure[-1]
        else None
        for x in critical
    ]
    out = []
    for t, ex in zip(times, exposure, strict=True):
        gas = initial * np.interp(ex, u, fraction)
        above = [np.interp(max(0.0, ex - x), u, number) for x in critical]
        out.append(
            dict(
                time_s=t,
                first5_continuous_s=arrival[0],
                first10_continuous_s=arrival[1],
                number_m3=np.interp(ex, u, number),
                above_5nm_m3=above[0],
                above_10nm_m3=above[1],
                gas_kg_m3=gas,
                particle_kg_m3=initial - gas,
                largest_diameter_nm=np.interp(ex, age, 2 * r)
                if ex > 0
                else 0.0,
            )
        )
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--initial", nargs="+", type=float, default=[0.0, 1e-10, 1e-9, 1e-8]
    )
    p.add_argument("--rate", nargs="+", type=float, default=[1e-7])
    p.add_argument("--end", type=int, default=128)
    p.add_argument(
        "--du", nargs="+", type=float, default=[0.0625, 0.03125, 0.015625]
    )
    a = p.parse_args()
    rows = []
    for c in a.initial:
        for rate in a.rate:
            for law in ["activation", "kinetic"]:
                for du in a.du:
                    for row in curve(c, rate, law, a.end, du):
                        rows.append(
                            dict(
                                initial_kg_m3=c,
                                rate=rate,
                                law=law,
                                du=du,
                                **row,
                            )
                        )
    write(a.out, rows)
