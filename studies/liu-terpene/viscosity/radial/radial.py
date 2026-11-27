# %% Conservative moving spherical shells at measured macroscopic viscosity
"""Seven equal-molar-volume organics evaporate into a perfect gas sink."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from particula.dynamics import get_first_order_mass_transport_k
from particula.particles import (
    get_kelvin_radius,
    get_kelvin_term,
    get_vapor_transition_correction,
)
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse import lil_matrix
from scipy.special import expit

HERE = Path(__file__).resolve().parent
T = 297.15
KB = 1.380649e-23
R0 = 70.0e-9
RHO = 1200.0
CSTAR = 10.0 ** np.arange(-4, 3)
ALPHA = 1.0
DG = 6.0e-6
MW = 0.20
SIGMA = 0.03
MFP = 3 * DG / np.sqrt(8 * 8.314462618 * T / (np.pi * MW))
KR = float(get_kelvin_radius(SIGMA, RHO, MW, T))


def gas_speeds(radius, density=RHO, kelvin_radius=KR):
    """Pure-bin evaporation recession speeds, m/s, for a unit surface fraction."""
    kn = MFP / radius
    transition = (
        0.75
        * ALPHA
        * (1 + kn)
        / (kn**2 + kn + 0.283 * ALPHA * kn + 0.75 * ALPHA)
    )
    return (
        DG
        * transition
        / radius
        * np.exp(kelvin_radius / radius)
        * CSTAR
        * 1.0e-9
        / density
    )


def check_gas():
    radii = np.geomspace(0.4 * R0, R0, 47)
    beta = get_vapor_transition_correction(MFP / radii, ALPHA)
    coeff = get_first_order_mass_transport_k(radii, beta, DG) * get_kelvin_term(
        radii, KR
    )
    model = np.array([gas_speeds(r) for r in radii])
    oracle = (
        coeff[:, None]
        * CSTAR[None, :]
        * 1.0e-9
        / (4 * np.pi * radii[:, None] ** 2 * RHO)
    )
    np.testing.assert_allclose(model, oracle, rtol=3.0e-15, atol=0)
    return float(np.max(abs(model / oracle - 1)))


def inputs(packet):
    rows = list(csv.DictReader((packet / "model/results/fits.csv").open()))
    f = np.array(
        [
            float(r["fitted_fraction"])
            for r in rows
            if (r["model"], r["precursor"], r["rh"])
            == ("primary", "caryophyllene", "dry")
        ]
    )
    obs = [
        r
        for r in csv.DictReader((packet / "sources/observations.csv").open())
        if (r["precursor"], r["rh"]) == ("farnesene", "dry")
    ]
    t = np.array([float(r["time_min"]) for r in obs])
    y = np.array([float(r["vfr"]) for r in obs])
    assert len(f) == 7 and np.all(f > 0) and abs(f.sum() - 1) < 1.0e-8
    return f / f.sum(), t, y


def distribution(edges, f, thickness_nm, sign, initial_radius=R0):
    """Reallocate initial inventory between a fixed-thickness shell and its core."""
    V = np.diff(edges**3)
    if thickness_nm == 0:
        return V[:, None] * f
    interface = 1 - thickness_nm * 1.0e-9 / initial_radius
    outer = 1 - interface**3
    rank = sign * np.log10(CSTAR)

    def partition(mid):
        return 0.001 + 0.998 * expit(3.0 * (rank - mid))

    mid = brentq(lambda m: f @ partition(m) - outer, -15, 15)
    p = partition(mid)
    outer_v = np.maximum(
        0.0, edges[1:] ** 3 - np.maximum(edges[:-1] ** 3, interface**3)
    )
    inner_v = V - outer_v
    mass = outer_v[:, None] * f * p / outer + inner_v[:, None] * f * (1 - p) / (
        1 - outer
    )
    np.testing.assert_allclose(mass.sum(0), f, atol=3.0e-14)
    return mass


def surface(w, b, k):
    """Flux-continuous surface for equal-density species. No fitted shell resistance."""
    # s_i = b w_i / (b+k_i-u), where u is the recession speed.
    # Let z=b+k_min-u instead of subtracting nearly equal speeds.
    shifted = k - k[0]

    def residual(z):
        return np.sum(b * w / (shifted + z)) - 1.0

    lo = b * w[0] * 0.9999
    hi = b * np.sum(w) * 1.0001
    if lo <= 0 or residual(lo) < 0 or residual(hi) > 0:
        raise ArithmeticError(f"Invalid surface bracket {w=}, {b=}")
    z = brentq(residual, lo, hi, xtol=1e-32, rtol=1e-13)
    ws = b * w / (shifted + z)
    speed = k * ws
    return speed, speed.sum()


def run(
    f,
    t,
    D,
    n,
    thick=0.0,
    sign=0,
    rtol=2.0e-7,
    first_step=None,
    *,
    initial_radius=R0,
    gas_rates=None,
):
    """Finite-volume ALE mass balance in moving shells and seven escaped inventories."""
    gas_rates = gas_speeds if gas_rates is None else gas_rates
    edges = np.linspace(0, 1, n + 1)
    V = np.diff(edges**3)
    centers = 0.75 * np.diff(edges**4) / V
    initial = distribution(edges, f, thick, sign, initial_radius)
    # State blocks: extensive species masses per initial particle mass, radius
    # relative to initial radius, escaped inventory per initial particle mass.
    # The sparse pattern includes the shared recession speed driven by the
    # surface composition, and the current radius, in every interior equation.
    nr = 7 * n
    jac = lil_matrix((nr + 8, nr + 8), dtype=bool)
    jac[:, nr - 7 : nr + 1] = True
    for j in range(n):
        for v in range(7):
            for h in (j - 1, j, j + 1):
                if 0 <= h < n:
                    jac[7 * j + v, 7 * h + v] = True

    def rhs(t0, state):
        x = state[nr]
        rad = x * initial_radius
        w = state[:nr].reshape(n, 7) / (x**3 * V[:, None])
        # The dominant least-volatile component stays strictly positive.
        # Insignificant signed Newton perturbations in trace fractions do not
        # introduce artificial mass; no particle inventories are clipped.
        surf, u = surface(w[-1], D / (rad * (1 - centers[-1])), gas_rates(rad))
        area = 3 * x * x / initial_radius
        flux = np.zeros((n + 1, 7))
        flux[1:-1] = (
            area
            * edges[1:-1, None] ** 2
            * (
                D / (rad * np.diff(centers)[:, None]) * (w[:-1] - w[1:])
                + u * edges[1:-1, None] * w[:-1]
            )
        )
        flux[-1] = area * surf
        return np.r_[
            -np.diff(flux, axis=0).ravel(), -u / initial_radius, flux[-1]
        ]

    tout = np.unique(np.r_[0.0, t, 15.0, 200.0, 1200.0])
    ic = np.r_[initial.ravel(), 1.0, np.zeros(7)]
    sol = solve_ivp(
        rhs,
        (0, tout[-1] * 60.0),
        ic,
        method="BDF",
        t_eval=tout * 60.0,
        rtol=rtol,
        atol=np.r_[np.repeat(V * 2.0e-11, 7), 1.0e-11, np.full(7, 1.0e-11)],
        jac_sparsity=jac.tocsr(),
        first_step=first_step,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    masses = sol.y[:nr].reshape(n, 7, -1)
    escaped = sol.y[nr + 1 :]
    retained = masses.sum(axis=(0, 1))
    budget = np.max(abs(masses.sum(0) + escaped - f[:, None]))
    closure = np.max(abs(masses.sum(1) / (V[:, None] * sol.y[nr] ** 3) - 1.0))
    if budget > 2.0e-6 or closure > 4.0e-5 or masses.min() < -1.0e-9:
        raise ArithmeticError((budget, closure, masses.min()))
    return (
        tout,
        retained,
        dict(
            nfev=sol.nfev,
            calls_lu=sol.nlu,
            component_mass_budget=budget,
            local_density_error_fraction=closure,
            min_mass_fraction=float(masses.min()),
        ),
    )


def write(path, rows):
    with path.open("w") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def campaign(packet, out, stage):
    from itertools import product

    f, t, y = inputs(packet)
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{stage}.csv").exists():
        raise FileExistsError("Use a new output directory")
    tests = []
    if stage == "pilot":
        tests = [
            ("homogeneous", 1.4e8, 0.3, 32, 0.0, 0),
            ("homogeneous", 2.8e8, 0.6, 32, 0.0, 0),
        ]
    elif stage == "matrix":
        tests = [
            ("homogeneous", eta, a, 64, 0.0, 0)
            for eta, a in product((1.4e8, 2.8e8), (0.3, 0.6))
        ]
        for eta, a in ((1.4e8, 0.3), (2.8e8, 0.6)):
            for h, sign in product((2.0, 5.0, 10.0), (-1, 1)):
                tests.append(
                    (
                        "outer enriched" if sign == 1 else "outer depleted",
                        eta,
                        a,
                        64,
                        h,
                        sign,
                    )
                )
        tests.append(("rapid mix", 1.0e3, 0.45, 32, 0.0, 0))
    else:
        tests = [
            ("homogeneous", eta, a, n, 0.0, 0)
            for eta, a in ((1.4e8, 0.3), (2.8e8, 0.6))
            for n in (32, 96, 160)
        ]
        for eta, a in ((1.4e8, 0.3), (2.8e8, 0.6)):
            for h, sign in ((5.0, 1), (5.0, -1), (10.0, 1)):
                tests.append(
                    (
                        "outer enriched" if sign == 1 else "outer depleted",
                        eta,
                        a,
                        128,
                        h,
                        sign,
                    )
                )
    time0 = time.monotonic()
    raw = []
    metrics = []
    for case, eta, a, n, h, sgn in tests:
        T0 = time.monotonic()
        D = KB * T / (6 * np.pi * eta * a * 1.0e-9)
        tt, V, check = run(f, t, D, n, h, sgn)
        yhat = np.interp(t, tt, V)
        labels = dict(
            case=case,
            viscosity_Pas=eta,
            solute_radius_nm=a,
            D_m2_s=D,
            radii=n,
            shell_nm=h,
        )
        for ti, obs, calc in zip(t, y, yhat):
            raw.append(
                dict(
                    **labels,
                    t_min=ti,
                    observed=obs,
                    computed=calc,
                    signed_error=calc - obs,
                )
            )
        metrics.append(
            dict(
                **labels,
                vfr_15=np.interp(15.0, tt, V),
                vfr_200=np.interp(200.0, tt, V),
                vfr_1200=np.interp(1200.0, tt, V),
                day_rmse=float(
                    np.mean((yhat[t < 300] - y[t < 300]) ** 2) ** 0.5
                ),
                night_error=yhat[-1] - y[-1],
                **check,
                wall_s=time.monotonic() - T0,
            )
        )
        print(
            case,
            n,
            h,
            eta,
            a,
            "first/last",
            yhat[[0, -1]],
            "s",
            round(time.monotonic() - T0, 2),
            flush=True,
        )
        write(out / f"{stage}.csv", raw)
        write(out / f"{stage}-metrics.csv", metrics)
    (out / f"{stage}-runtime.json").write_text(
        json.dumps(
            dict(
                runs=len(tests),
                total_s=time.monotonic() - time0,
                gas_speed_error=check_gas(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("pilot", "matrix", "refine"))
    p.add_argument("packet", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    campaign(a.packet, a.out, a.stage)
