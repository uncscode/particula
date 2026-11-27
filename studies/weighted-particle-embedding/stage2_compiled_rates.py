"""Accelerate continuous neutral trajectories with a compiled joint RHS."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from coupled_events import KT, MFP_AIR, MU_AIR
from numba import njit
from sparse_vapor_reference import ExplicitReservoir
from stage2_adaptive_kinetics import trained
from stage2_dilution import box
from stage2_population import collision_kernel, fixture, observable_names
from stage2_reference import reference


@njit(cache=True)
def transition_factor(kn, a, features, output_bias, col):
    """Evaluate the configured native, rational or tanh kinetic closure."""
    if features.shape[1] == 1:
        c, d, e = features[col, 0, :3]
        beta = (1 + kn) / (1 + (c / a + d) * kn + e / a * kn * kn)
    elif features.shape[1]:
        u = kn / (a + kn)
        v = kn / (1 + kn)
        z = np.log(a) / 3
        residual = output_bias[col]
        for j in range(features.shape[1]):
            w = features[col, j]
            residual += w[4] * np.tanh(
                (2 * u - 1) * w[0] + (2 * v - 1) * w[1] + z * w[2] + w[3]
            )
        beta = a / (a + kn) * np.exp(u * residual)
    else:
        beta = 0.75 * a * (1 + kn) / (kn * kn + (1 + 0.283 * a) * kn + 0.75 * a)
    return beta


@njit(cache=True)
def joint(
    y,
    dry,
    v0,
    mole0,
    density,
    molar,
    diffus,
    alpha,
    path,
    kelvin,
    sat,
    volume,
    features,
    output_bias,
    jump,
    offset,
):
    """Evaluate all particle, gas, and integrated hazard derivatives."""
    n, s = len(dry), len(sat)
    out = np.zeros_like(y)
    radius = np.empty(n)
    diffusion = np.empty(n)
    speed = np.empty(n)
    distance = np.empty(n)
    for row in range(n):
        vol = v0[row]
        mol = mole0[row]
        mass = dry[row]
        for col in range(s):
            q = max(0.0, y[row * s + col])
            vol += dry[row] * q / density[col]
            mol += q / molar[col]
            mass += dry[row] * q
        r = (3 * vol / (4 * np.pi)) ** (1 / 3)
        radius[row] = r
        for col in range(s):
            q = max(0.0, y[row * s + col])
            a = alpha[col]
            kn = path[col] / r
            beta = transition_factor(kn, a, features, output_bias, col)
            eq = np.exp(kelvin[col] / r) * sat[col] * q / (mol * molar[col])
            flux = (
                4
                * np.pi
                * r
                * diffus[col]
                * beta
                * (y[n * s + col] * sat[col] - eq)
            )
            out[row * s + col] = flux / dry[row]
            out[n * s + col] -= flux / (volume * sat[col])
        kn_air = MFP_AIR / r
        d = (
            KT
            * (1 + kn_air * (1.257 + 0.4 * np.exp(-1.1 / kn_air)))
            / (6 * np.pi * MU_AIR * r)
        )
        sp = np.sqrt(8 * KT / (np.pi * mass))
        length = 8 * d / (np.pi * sp)
        if jump.shape[0]:
            xx = np.log(length / (2 * r))
            correction = offset
            for j in range(len(jump)):
                correction += jump[j, 2] * np.tanh(
                    xx * jump[j, 0] / 3 + jump[j, 1]
                )
            g = length * (0.5 + 0.5 / (1 + np.exp(-xx - correction)))
        else:
            g = (
                (2 * r + length) ** 3 - (4 * r * r + length * length) ** 1.5
            ) / (6 * r * length) - 2 * r
        diffusion[row] = d
        speed[row] = sp
        distance[row] = g
    for i in range(n):
        for j in range(i + 1, n):
            rad = radius[i] + radius[j]
            d = diffusion[i] + diffusion[j]
            denom = rad / (rad + np.hypot(distance[i], distance[j])) + 4 * d / (
                rad * np.hypot(speed[i], speed[j])
            )
            out[-1] += 4 * np.pi * d * rad / (denom * volume)
    return out


class Factory:
    """Prepare immutable per-collision arrays for the compiled integrator."""

    def __init__(self, model):
        """Pack the same fitted coefficients as the tensor forecast."""
        self.model = model
        self.kernel = collision_kernel if model is None else model.pair
        if model is None:
            self.head = np.zeros((2, 0, 5))
            self.bias = np.zeros(2)
            self.jump = np.zeros((0, 3))
            self.offset = 0.0
        else:
            self.head = np.concatenate(
                [model.w, model.b[..., None], model.out[..., None]], axis=2
            )
            self.bias = model.offset
            w, b, v, c = model.jump
            self.jump = np.column_stack([w[:, 0], b, v[0]])
            self.offset = float(model.intercept + c[0])

    def reservoir(self, m, n, volume, spec):
        """Bind one unit-weight, neutral material box to the hot callback."""
        if np.any(n != 1) or spec.species != 2:
            raise ValueError(
                "Compiled experiment requires explicit two-vapor boxes"
            )
        model = ExplicitReservoir(m, n, volume, spec)
        local = model.local
        args = (
            local.dry,
            local.base_vol,
            local.base_moles,
            spec.density,
            spec.molar,
            spec.diffusivity,
            spec.accommodation,
            local.path,
            local.kelvin,
            spec.saturation,
            volume,
            self.head,
            self.bias,
            self.jump,
            self.offset,
        )
        model.joint_rates = lambda _, y: joint(y, *args)
        return model


@lru_cache(maxsize=2)
def factory(weights, method):
    """Read the trained constitutive functions once per process."""
    fac = Factory(None if method == "native" else trained(weights))
    if method == "rational":
        from stage2_transfer import restore

        b, _ = box(74001, 1, 10**11, 16, 900)
        net = restore(b.spec, weights / "conductance-rational-w0-separate.pt")
        fac.head = np.zeros((2, 1, 5))
        for i, label in enumerate(b.spec.names):
            unit = net.units["water" if label == "water" else "organic"]
            fac.head[i, 0, :3] = unit.coeff.detach().exp().numpy()
    return fac


def one(job):
    """Save a continuous realization and its integration diagnostics."""
    work, weights, method, k, tol, f, m, d, schedule, frames, legs, existing = (
        job
    )
    if schedule == "dilute":
        b, frames = box(f, m, d, 16, 900)
        label = "dilute-coupled"
        legs = 4
    else:
        b = fixture(f, m, 16, d, frames=frames, legs=legs)
        label = "coupled"
    if method == "native":
        name = (
            f"f{f}-m{m}-n16-d{d}-{label}-zero-r{k}-full"
            f"-t{tol:.0e}-LSODA-f{frames}-l{legs}.npz"
        )
    else:
        name = f"f{f}-m{m}-d{d}-r{k}-t{tol:.0e}.npz"
    file = work / name
    if file.exists() or (existing is not None and (existing / name).exists()):
        return
    fac = factory(weights, method)
    tic = perf_counter()
    obs, stats = reference(
        b, k, tol, reservoir=fac.reservoir, kernel=fac.kernel
    )
    np.savez_compressed(
        file,
        obs=obs,
        time=b.t,
        names=observable_names(b.spec),
        seconds=perf_counter() - tic,
        implementation="numba-joint-rhs",
        **stats,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("weights", type=Path)
    p.add_argument("--schedule", choices=["pulse", "dilute"], default="dilute")
    p.add_argument("--families", type=int, nargs="+", default=[74001])
    p.add_argument("--mix", type=int, nargs="+", default=[1])
    p.add_argument("--densities", type=int, nargs="+", default=[10**11])
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--legs", type=int, default=96)
    p.add_argument("--existing", type=Path)
    p.add_argument(
        "--model", choices=["native", "learned", "rational"], required=True
    )
    p.add_argument("--tol", nargs="+", type=float, default=[1e-9])
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    jobs = [
        (
            a.out,
            a.weights,
            a.model,
            k,
            t,
            f,
            m,
            d,
            a.schedule,
            a.frames,
            a.legs,
            a.existing,
        )
        for f, m, d, k, t in product(
            a.families, a.mix, a.densities, a.seeds, a.tol
        )
    ]
    one(jobs[0])
    with ProcessPoolExecutor(a.workers) as pool:
        for i, _ in enumerate(pool.map(one, jobs, chunksize=1), 1):
            if i % 64 == 0:
                print(i, len(jobs), flush=True)
