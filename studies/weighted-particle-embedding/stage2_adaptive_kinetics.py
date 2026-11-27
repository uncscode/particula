"""Integrate fitted gas exchange and Brownian hazards with adaptive LSODA."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from coupled_events import KT, MFP_AIR, MU_AIR
from scipy.special import expit
from sparse_vapor_reference import ExplicitReservoir
from stage2_dilution import box
from stage2_pair import load_pair
from stage2_population import observable_names, wet_radius
from stage2_reference import reference
from stage2_transfer import restore


class Laws:
    """Export the frozen tanh fits in double precision for solver callbacks."""

    def __init__(self, spec, weights):
        """Read the same water, organic and scalar jump checkpoint weights."""
        net = restore(spec, weights / "conductance-features-w128-separate.pt")
        pair = load_pair(weights / "jump-w4-s901.pt", "cpu")
        heads = [
            net.units["water" if k == "water" else "organic"]
            for k in spec.names
        ]

        def array(parameters):
            return np.stack([p.detach().numpy() for p in parameters])

        self.w = array([h.layer.weight for h in heads])
        self.b = array([h.layer.bias for h in heads])
        self.out = array([h.output.weight[0] for h in heads])
        self.offset = array([h.output.bias[0] for h in heads])
        self.jump = [p.detach().numpy() for p in pair.shape.parameters()]
        self.intercept = pair.offset.detach().numpy()

    def beta(self, kn, accommodation):
        """Evaluate the frozen positive conductance over all particle lanes."""
        a = np.maximum(accommodation, 1e-12)
        u = kn / (kn + a)
        v = kn / (kn + 1)
        hidden = np.tanh(
            self.b
            + (2 * u - 1)[..., None] * self.w[..., 0]
            + (2 * v - 1)[..., None] * self.w[..., 1]
            + (np.log(a) / 3)[:, None] * self.w[..., 2]
        )
        residual = np.sum(hidden * self.out, axis=-1) + self.offset
        return np.where(
            accommodation > 0, a / (a + kn) * np.exp(u * residual), 0
        )

    def pair(self, mass, charge, spec, kind, i, j):
        """Evaluate the fitted jump law on neutral wet particles."""
        if kind != "brownian" or np.any(charge):
            raise ValueError(
                "The fitted jump-distance model requires neutral Brownian pairs"
            )
        r = wet_radius(mass, spec)
        kn = MFP_AIR / r
        diff = (
            KT
            * (1 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn)))
            / (6 * np.pi * MU_AIR * r)
        )
        speed = np.sqrt(8 * KT / (np.pi * mass.sum(1)))
        length = 8 * diff / (np.pi * speed)
        logx = np.log(length / (2 * r))
        w, b, v, c = self.jump
        residual = np.tanh(logx[:, None] * w[:, 0] / 3 + b) @ v[0] + c[0]
        jump = length * (0.5 + 0.5 * expit(logx + self.intercept + residual))
        rad = r[i] + r[j]
        d = diff[i] + diff[j]
        denom = rad / (rad + np.hypot(jump[i], jump[j])) + 4 * d / (
            rad * np.hypot(speed[i], speed[j])
        )
        return 4 * np.pi * d * rad / denom

    def reservoir(self, m, n, volume, spec):
        """Use the same constitutive model for every adaptive RHS evaluation."""
        model = ExplicitReservoir(m, n, volume, spec)
        base = model.local
        s = base.c

        def rates(_, y):
            gas = y[-s:] * model.scale
            q = np.maximum(y[:-s].reshape(-1, s), 0)
            vol = base.base_vol + base.dry * (q / spec.density[:s]).sum(1)
            rr = np.cbrt(3 * vol / (4 * np.pi))[:, None]
            mol = base.base_moles[:, None] + (q / spec.molar[:s]).sum(
                1, keepdims=True
            )
            eq = (
                np.exp(base.kelvin / rr)
                * spec.saturation
                * q
                / (mol * spec.molar[:s])
            )
            gain = (
                4
                * np.pi
                * rr
                * spec.diffusivity
                * self.beta(base.path / rr, spec.accommodation)
            )
            dq = gain * (gas - eq) / base.dry[:, None]
            return np.r_[dq.ravel(), -(base.w @ dq) / model.scale]

        model.rates = rates
        return model


class RationalLaws(Laws):
    """Use the tensor-fitted rational parameters with the same jump network."""

    def __init__(self, spec, weights):
        """Restore the identifiable law independently for the CPU callback."""
        super().__init__(spec, weights)
        net = restore(spec, weights / "conductance-rational-w0-separate.pt")
        self.coeff = np.stack(
            [
                net.units["water" if k == "water" else "organic"]
                .coeff.detach()
                .exp()
                .numpy()
                for k in spec.names
            ]
        ).T

    def beta(self, kn, accommodation):
        """Evaluate the three fitted coefficients per material."""
        c, d, e = self.coeff
        a = np.maximum(accommodation, 1e-12)
        return np.where(
            accommodation > 0,
            (1 + kn) / (1 + (c / a + d) * kn + e / a * kn**2),
            0,
        )


@lru_cache(maxsize=2)
def trained(weights):
    """Restore the fixed two-vapor experiment once in each worker."""
    b, _ = box(69001, 0, 10**9, 16, 900)
    return Laws(b.spec, weights)


def run_case(task):
    """Cache one continuously integrated fitted forecast and realized cost."""
    f, m, d, k, tol, work, weights = task
    out = work / f"f{f}-m{m}-d{d}-r{k}-t{tol:.0e}.npz"
    if out.exists():
        return
    b, _ = box(f, m, d, 16, 900)
    model = trained(weights)
    start = perf_counter()
    obs, stats = reference(
        b,
        k,
        tol,
        True,
        "brownian",
        "LSODA",
        reservoir=model.reservoir,
        kernel=model.pair,
    )
    np.savez_compressed(out, obs=obs, seconds=perf_counter() - start, **stats)


def collect(a, keys, tol):
    """Package the trajectories in the same schema as the split forecasts."""
    obs, seconds, drift, rhs = [], [], [], []
    for f, m, d, k in keys:
        with np.load(a.cache / f"f{f}-m{m}-d{d}-r{k}-t{tol:.0e}.npz") as z:
            obs.append(z["obs"])
            seconds.append(float(z["seconds"]))
            drift.append(float(z["drift"]))
            rhs.append(int(z["evaluations"]))
    b, frames = box(*keys[0, :3], 16, 900)
    meta = dict(
        process="dilute-coupled",
        charge="zero",
        slots=16,
        frames=frames,
        legs=4,
        representation="full",
        scheme="coupled-LSODA",
        kernel="jump-w4-s901",
        transfer=getattr(a, "transfer", "conductance-features-w128-separate"),
        device="cpu",
        rtol=tol,
        mean_rhs=np.mean(rhs),
        plateau_seconds=900,
    )
    np.savez_compressed(
        a.out / f"adaptive-kinetics-n{len(a.seeds)}-t{tol:.0e}.npz",
        obs=np.asarray(obs),
        keys=keys,
        t=b.t,
        names=observable_names(b.spec),
        max_drift=max(drift),
        seconds=seconds,
        meta=json.dumps(meta),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "out", "weights"):
        p.add_argument(name, type=Path)
    p.add_argument("--mix", type=int, nargs="+", default=[0, 1])
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--families", type=int, nargs="+", default=[69001, 70001])
    p.add_argument("--densities", type=int, nargs="+", default=[10**9, 10**11])
    p.add_argument("--tol", type=float, nargs="+", default=[1e-7, 1e-9])
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    a.cache.mkdir(exist_ok=True, parents=True)
    a.out.mkdir(exist_ok=True, parents=True)
    keys = np.asarray(list(product(a.families, a.mix, a.densities, a.seeds)))
    for tol in a.tol:
        jobs = [(*key, tol, a.cache, a.weights) for key in keys]
        with ProcessPoolExecutor(a.workers) as pool:
            for index, _ in enumerate(pool.map(run_case, jobs, chunksize=1), 1):
                if index % 16 == 0:
                    print(tol, index, len(jobs), flush=True)
        collect(a, keys, tol)
