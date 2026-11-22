"""Closed O/H/S reduced chemistry for the 36 Zeuthen Appendix A outlet samples.

Species concentrations are on the inlet gas molar-density basis in ppm. Sulfur,
hydrogen and oxygen atomic inventories determine SO2, water and O2 exactly.
The radical subset omits atomic H and HOSO2. Forward/reverse constants use
one-bar equilibrium and all source forms except R11 wall loss are reversible.
The GPU advances four independent concentrations using a two-stage L-stable
Rosenbrock method with a local analytic Jacobian. No steady-state O assumption.
"""

from pathlib import Path

import numpy as np
import warp as wp

INPUT = Path(__file__).resolve().parents[1] / "inputs"
G = np.genfromtxt(INPUT / "dry_feeds.csv", delimiter=",", names=True)
T = G["temperature_K"]
PRESSURE_BAR = 1.05 * 1.01325
R = 8.31446261815324
C = PRESSURE_BAR * 1e5 / (R * T)
U = 1e-6 * C
OLD_F = 69 * np.exp(
    -63000 / (R * T)
    + (-5.125364149019666)
    + 1000 * (114.45626480078023) / R * (1 / 1100 - 1 / T)
)
# Table of reversible reactions. Rows = SO3,O,OH,HO2; 4..7=SO2,O2,H2O,unit.
REAGENTS = np.array(
    [
        [0, 7, 4, 1],
        [0, 1, 4, 5],
        [1, 1, 5, 7],
        [2, 2, 1, 6],
        [0, 2, 4, 3],
        [1, 3, 2, 5],
        [2, 3, 5, 6],
        [1, 7, 7, 7],
        [4, 7, 0, 7],
    ],
    dtype=np.int32,
)
NU = np.array(
    [
        [-1, 1, 0, 0],
        [-1, -1, 0, 0],
        [0, -2, 0, 0],
        [0, 1, -2, 0],
        [-1, 0, -1, 1],
        [0, -1, 1, -1],
        [0, 0, -1, -1],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
    ],
    dtype=float,
)
D = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
        [-0.5, -0.5, -0.25, -0.75],
        [0, 0, -0.5, -0.5],
        [0, 0, 0, 0],
    ],
    dtype=float,
)
thermo = np.genfromtxt(INPUT / "dry_equilibrium.csv", delimiter=",", names=True)
if not np.isin(T, thermo["temperature_K"]).all():
    raise ValueError("Missing dry equilibrium data for a measured temperature")
# The three experimental sweeps revisit temperatures in different orders.
lookup = {row["temperature_K"]: j for j, row in enumerate(thermo)}
thermo_row = [lookup[t] for t in T]
K1, K2, K3, K4, K5, K6, K7 = [thermo[f"K{i}"][thermo_row] for i in range(1, 8)]


def _inputs(x, stage, trial, old_core=False):
    """Stage adds current proposal. Positive trial selects its general form."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 40:
        raise ValueError("Expected (n,40) parameters")
    if stage < 0 or trial not in (0, 1):
        raise ValueError("Invalid stage or trial")
    zeros = np.zeros((len(x), 1), dtype=x.dtype)

    def z(i):
        return x[:, i - 1, None] if stage >= i else zeros

    def gen(i):
        return stage == i and trial == 1

    oxy = G["oxygen_percent"] * 0.01
    s3 = G["so3_in_ppm"]
    s = G["so2_in_ppm"] + s3
    # The affine map spans 30..150 about the 100 ppm estimate via a smooth
    # polynomial.
    h2o = 100 + (100 * z(7) if gen(7) else 60 * z(7) - 10 * z(7) ** 2)
    if old_core:
        h2o = h2o * 0
    n2 = C * (1 - oxy - 1e-6 * (s + h2o))
    k1 = 5.7e11 * np.exp(-40000 / T) * n2
    k2 = 7.8e5 * np.exp(-3065 / T)
    # Round 3 interpolates to the 2007 computed O-abstraction rate.
    target2 = 0.028 * T**2.57 * np.exp(-122300 / (R * T))
    k2 = k2 * np.exp(
        z(3) * ((5000 * (1 / 1273 - 1 / T)) if gen(3) else np.log(target2 / k2))
    )
    kinf = 3.7e5 * np.exp(-850 / T)
    pr = 2.9e15 * T**-3.58 * np.exp(-2620 / T) * n2 / kinf
    lfc = np.log10(0.57 * np.exp(-T / 371) + 0.43 * np.exp(-T / 7442))
    cc = -0.4 - 0.67 * lfc
    nn = 0.75 - 1.27 * lfc
    u = np.log10(pr) + cc
    bro = 10 ** (lfc / (1 + (u / (nn - 0.14 * u)) ** 2))
    kd_troe = kinf * pr / (1 + pr) * bro * K1
    k1 = k1 * np.exp(
        z(4) * (1000 * (1 / 1273 - 1 / T) if gen(4) else np.log(kd_troe / k1))
    )
    k1 = k1 * np.exp(z(1) * (2 if gen(1) else np.log(2.0)))
    k2 = k2 * np.exp(
        z(2) * (np.log(3.0) * (oxy / 0.04) if gen(2) else np.log(3.0))
    )
    # (m3/mol/s) for all bimolecular hazards after multiplication by [M].
    k3 = (
        1.2e5
        / T
        * (C + 14.4 * h2o * U)
        * np.exp(z(5) * (2 if gen(5) else np.log(2.0)))
    )
    k4 = (
        0.0357
        * T**2.4
        * np.exp(2110 * 4.184 / (R * T))
        * np.exp(z(6) * (2 if gen(6) else np.log(2.0)))
    )
    k5 = (
        0.048
        * T**2.46
        * np.exp(-114100 / (R * T))
        * np.exp(z(8) * (2 if gen(8) else np.log(3.0)))
    )
    k6 = 2e7 * np.exp(z(9) * (2 * (oxy / 0.04) if gen(9) else np.log(2.0)))
    k7 = (
        1.45e7 * np.exp(500 * 4.184 / (R * T))
        + 5e9 * np.exp(-17330 * 4.184 / (R * T))
    ) * np.exp(z(10) * (2 if gen(10) else np.log(2.0)))
    wall = z(11) * (20 if gen(11) else 20 * (T / 1273) ** 1.75)
    old = z(12) * (np.ones_like(T) * 0.01 if gen(12) else OLD_F)
    old_rev = old * 10 ** (4.678 - 4956 / T) / np.sqrt(PRESSURE_BAR * oxy)
    one = np.ones_like(k1)
    pairs = [
        (k1, k1 / K1 * U),
        (k2 * U, k2 / K2 * U),
        (k3 * U, k3 / K3),
        (k4 * U, k4 / K4 * U),
        (k5 * U, k5 / K5 * U),
        (k6 * U, k6 / K6 * U),
        (k7 * U, k7 / K7 * U),
        (wall, 0),
        (old, old_rev),
    ]
    coeff = np.stack([one * v for pair in pairs for v in pair], axis=-1)
    if old_core:
        coeff[..., 4:] = 0
    offset = np.stack(
        [
            one * 0,
            one * 0,
            one * 0,
            one * 0,
            s * one,
            (oxy * 1e6 + (0 if old_core else 0.5 * s3)) * one,
            h2o * one,
            one,
        ],
        axis=-1,
    )
    return coeff, offset


F = wp.float64
V = wp.vec4d
M = wp.mat44d
Sp = wp.types.vector(8, F)
Rates = wp.types.vector(18, F)
I4 = wp.types.vector(4, wp.int32)


@wp.func
def _state(y: V, offset: Sp, d: wp.array(dtype=V)):
    v = offset
    for j in range(8):
        v[j] = v[j] + wp.dot(d[j], y)
    return v


@wp.func
def _rhs(v: Sp, rc: Rates, idx: wp.array(dtype=I4), nu: wp.array(dtype=V)):
    f = V()
    for j in range(9):
        r = idx[j]
        flux = rc[2 * j] * v[r[0]] * v[r[1]] - rc[2 * j + 1] * v[r[2]] * v[r[3]]
        f += nu[j] * flux
    return f


@wp.func
def _matrix(
    v: Sp,
    rc: Rates,
    idx: wp.array(dtype=I4),
    nu: wp.array(dtype=V),
    d: wp.array(dtype=V),
):
    jacobian = M()
    for j in range(9):
        r = idx[j]
        g = rc[2 * j] * (v[r[1]] * d[r[0]] + v[r[0]] * d[r[1]]) - rc[
            2 * j + 1
        ] * (v[r[3]] * d[r[2]] + v[r[2]] * d[r[3]])
        jacobian += wp.outer(nu[j], g)
    return jacobian


@wp.kernel
def _solve(
    rc: wp.array(dtype=Rates),
    offset: wp.array(dtype=Sp),
    inputs: wp.array(dtype=wp.vec2d),
    idx: wp.array(dtype=I4),
    nu: wp.array(dtype=V),
    d: wp.array(dtype=V),
    steps: int,
    out: wp.array(dtype=V),
):
    i = wp.tid()
    y = V(inputs[i % 36][0], F(0), F(0), F(0))
    unit = inputs[i % 36][1] / F(steps * steps)
    for j in range(steps):
        dt = unit * F(2 * j + 1)
        v = _state(y, offset[i], d)
        jacobian = _matrix(v, rc[i], idx, nu, d)
        inv = wp.inverse(
            wp.identity(n=4, dtype=F) - F(0.2928932188134525) * dt * jacobian
        )
        k1 = inv * (dt * _rhs(v, rc[i], idx, nu))
        k2 = inv * (
            dt * _rhs(_state(y + k1, offset[i], d), rc[i], idx, nu) - F(2) * k1
        )
        y = y + F(1.5) * k1 + F(0.5) * k2
    out[i] = y


def _gpu(x, stage, trial, device, steps, old_core=False):
    if steps < 1:
        raise ValueError("Positive step count required")
    coeff, offset = _inputs(x, stage, trial, old_core)
    d = D.copy()
    if old_core:
        d[5] = 0

    def arr(v, dtype):
        return wp.array(np.ascontiguousarray(v), dtype=dtype, device=device)

    n = len(x) * len(G)
    out = wp.empty(n, dtype=V, device=device)
    wp.launch(
        _solve,
        n,
        [
            arr(coeff.reshape(n, 18), Rates),
            arr(offset.reshape(n, 8), Sp),
            arr(np.column_stack((G["so3_in_ppm"], G["residence_s"])), wp.vec2d),
            arr(REAGENTS, I4),
            arr(NU, V),
            arr(d, V),
            steps,
            out,
        ],
        device=device,
    )
    return out.numpy().reshape(len(x), len(G), 4)


def predict_dry(x, stage, trial, device="cuda:0", steps=128):
    """Absolute SO2 outlet in original 36-row order. x has shape (draws,40)."""
    y = _gpu(np.asarray(x, dtype=float), stage, trial, device, steps)
    inventory = G["so2_in_ppm"] + G["so3_in_ppm"]
    if (
        not np.all(np.isfinite(y))
        or np.min(y) < -1e-7
        or np.any(y[..., 0] > inventory + 1e-7)
    ):
        raise FloatingPointError(
            "Unresolved dry kinetics; increase integration steps"
        )
    return G["so2_in_ppm"] + G["so3_in_ppm"] - y[..., 0]
