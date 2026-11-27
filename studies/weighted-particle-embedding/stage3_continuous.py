"""Continuous reference for irreversible growth and mergers."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from coupled_events import KT
from scipy.integrate import solve_ivp
from stage2_population import brownian
from stage3_charge_modes import initial_charge
from stage3_spectra import dry_radius

RHO = 1400.0
COULOMB = 1.602176634e-19**2 / (4 * np.pi * 8.8541878128e-12 * KT)


def initial(n, ns, f, zmode):
    """Create the explicit dry population and finite vapor reservoirs."""
    i = np.arange(n)
    u = ((i + 1) * 0.618033988749895 + f * 0.0331) % 1
    r = dry_radius(u, f, np.where)
    mass = 4 * np.pi / 3 * RHO * r**3
    z = initial_charge(i, u, zmode, np.where).astype(int)
    j = np.arange(ns)
    diff = 5e-6 + 1.2e-5 * (1 + j) / (1 + ns)
    alpha = 0.08 + 0.91 * (1 + (j * 7) % ns) / (1 + ns)
    path = 3 * diff / (150 + 150 * (j + 1) / ns)
    g = (1 + j / ns) * 1e-8 / ns
    return mass, z, g, diff, alpha, path


def charged(r, m, z, i, j):
    """Call native Gopalakrishnan 2012 via its system-state interface."""
    from particula.dynamics import (
        get_coulomb_kernel_gopalakrishnan2012_via_system_state,
    )

    return get_coulomb_kernel_gopalakrishnan2012_via_system_state(
        r, m, z, 293.0, 101325.0
    )[i, j]


def _pair_hazards(rad, mm, z, i, j, law, charge, v):
    if law == "off" or not len(i):
        return np.zeros(len(i))
    if law == "constant":
        return np.full(len(i), 2e-15) / v
    if charge == "neutral":
        return brownian(rad, mm, i, j) / v
    return charged(rad, mm, z, i, j) / v


def forecast(
    n,
    ns,
    density,
    charge,
    family,
    seed,
    end,
    tol,
    law,
    method,
    *,
    first_clock=None,
):
    """Integrate growth and collision hazard up to every sampled merger."""
    tic = perf_counter()
    m, z, g0, diff, alpha, path = initial(n, ns, family, charge)
    g = g0.copy()
    log_g = np.zeros(ns)
    v = n / density
    dry = m.sum() / v
    rng = np.random.default_rng(seed)
    clock = rng.exponential()
    if first_clock is not None:
        clock = float(first_clock)
    events = calls = 0
    now = 0.0
    out = []
    for target in np.linspace(0, end, 5):
        while now < target - 1e-13 * end:
            base = m.copy()
            i, j = np.triu_indices(len(m), 1)

            def coefficients(mm):
                rad = np.cbrt(3 * mm / (4 * np.pi * RHO))
                kn = path / rad[:, None]
                k = (
                    4
                    * np.pi
                    * rad[:, None]
                    * diff
                    * (
                        0.75
                        * alpha
                        * (1 + kn)
                        / (kn**2 + (1 + 0.283 * alpha) * kn + 0.75 * alpha)
                    )
                )
                return rad, k

            def rhs(_, y, base=base, z=z, i=i, j=j):
                mm = base * y[: len(base)]
                gas = g0 * np.exp(y[len(base) : -1])
                rad, k = coefficients(mm)
                flux = k * gas
                return np.r_[
                    flux.sum(1) / base,
                    -k.sum(0) / v,
                    _pair_hazards(rad, mm, z, i, j, law, charge, v).sum(),
                ]

            def event(_, y, clock=clock):
                return y[-1] - clock

            event.terminal = True
            event.direction = 1
            sol = solve_ivp(
                rhs,
                (now, target),
                np.r_[np.ones(len(m)), log_g, 0.0],
                method=method,
                rtol=tol,
                atol=tol * 0.01,
                events=event if law != "off" and len(i) else None,
            )
            if not sol.success:
                raise ArithmeticError(sol.message)
            y = sol.y[:, -1]
            now = sol.t[-1]
            calls += sol.nfev
            m = base * y[: len(m)]
            log_g = y[len(m) : -1]
            g = g0 * np.exp(log_g)
            if sol.status == 1:
                weights = _pair_hazards(
                    np.cbrt(3 * m / (4 * np.pi * RHO)),
                    m,
                    z,
                    i,
                    j,
                    law,
                    charge,
                    v,
                )
                cdf = np.cumsum(weights)
                k = np.searchsorted(cdf, rng.random() * cdf[-1])
                m[i[k]] += m[j[k]]
                z[i[k]] += z[j[k]]
                m = np.delete(m, j[k])
                z = np.delete(z, j[k])
                clock = rng.exponential()
                events += 1
            else:
                clock -= y[-1]
        r = np.cbrt(3 * m / (4 * np.pi * RHO))
        gap = abs(m.sum() / v + g.sum() - dry - g0.sum()) / (dry + g0.sum())
        if gap > max(1e-9, 20 * tol) or np.min(g) < 0:
            raise ArithmeticError("Reference ledger failed")
        out.append(
            dict(
                n=len(m),
                gas=g.tolist(),
                particle=np.r_[dry, g0 - g].tolist(),
                bands=np.bincount(
                    np.searchsorted([35e-9, 70e-9, 120e-9], r, side="right"),
                    minlength=4,
                ).tolist(),
                charge_abs=int(abs(z).sum()),
                ledger=gap,
            )
        )
    return dict(
        n=n,
        species=ns,
        density=density,
        charge=charge,
        family=family,
        seed=seed,
        horizon=end,
        tol=tol,
        kernel=law,
        method=method,
        calls=calls,
        collisions=events,
        seconds=perf_counter() - tic,
        history=out,
    )


def work(job):
    """Write one completed continuous reference atomically."""
    folder, n, ns, d, z, f, seed, h, tol, law, method = job
    path = (
        folder
        / f"ode-{n}-{ns}-{d:g}-{z}-{f}-{seed}-{h:g}-{tol:g}-{law}-{method}.json"
    )
    if path.exists():
        return
    row = forecast(n, ns, d, z, f, seed, h, tol, law, method)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(row) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    for k, v, typ in [
        ("n", [16], int),
        ("s", [2], int),
        ("density", [1e12, 1e14], float),
        ("charge", ["neutral"], str),
        ("family", [102], int),
        ("h", [1.0], float),
        ("tol", [1e-9], float),
        ("law", ["physical"], str),
        ("method", ["DOP853"], str),
    ]:
        p.add_argument("--" + k, nargs="+", type=typ, default=v)
    p.add_argument("--seed", type=int, default=221001)
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    a.folder.mkdir(exist_ok=True, parents=True)
    jobs = product(
        [a.folder],
        a.n,
        a.s,
        a.density,
        a.charge,
        a.family,
        range(a.seed, a.seed + a.seeds),
        a.h,
        a.tol,
        a.law,
        a.method,
    )
    with ProcessPoolExecutor(a.workers) as pool:
        for i, _ in enumerate(pool.map(work, jobs, chunksize=4), 1):
            if i % 128 == 0:
                print(i, flush=True)
