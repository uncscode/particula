"""Test which physical composition information a weighted group must retain."""

import argparse
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
from material_bank import mixture
from multivapor_reference import Rates
from scipy.integrate import solve_ivp
from sparse_vapor_reference import ExplicitReservoir, integrate
from stage2_population import fixture

BINS = (8, 16, 32, 64, 128, 256, 512, 1024)


def sorted_classes(mass, aware):
    """Sort dry mass separately within the observed salt-composition classes."""
    dry = mass[:, -2:].sum(1)
    salt = mass[:, -1] / dry
    split = aware and np.ptp(salt) > 1e-10
    group = (
        salt > (salt.min() + salt.max()) / 2
        if split
        else np.zeros(len(salt), bool)
    )
    classes = []
    for label in np.unique(group):
        ids = np.flatnonzero(group == label)
        ix = ids[np.argsort(dry[ids], kind="stable")]
        classes.append(mass[ix])
    return classes


def pack(classes, bins):
    """Preserve integral weights and every material's initial inventory."""
    mass, weights = [], []
    for m in classes:
        # Each class has either two or four copies at each matched dry size.
        repeats = 4 // len(classes)
        nd = len(m) // repeats
        if bins > nd:
            raise ValueError("More quantiles requested than distinct dry sizes")
        edges = (np.arange(bins + 1, dtype=np.int64) * nd // bins) * repeats
        w = np.diff(edges)
        mass.append(np.add.reduceat(m, edges[:-1], axis=0) / w[:, None])
        weights.append(w)
    return np.concatenate(mass), np.concatenate(weights)


def flux(m, w, v, g, spec, chunk=8192):
    """Directly sum positive and negative physical exchange over actual rows."""
    sums = np.zeros(spec.species)
    for start in range(0, len(w), chunk):
        ids = slice(start, start + chunk)
        model = Rates(m[ids], w[ids], v, spec)
        q = m[ids, : spec.species] / model.dry[:, None]
        sums += model.w @ model.local(q.ravel(), g).reshape(q.shape)
    return sums


def prepare(root, n, s, f, mix, d, c):
    """Build the individual inputs and save weighted physical states."""
    root.mkdir(parents=True, exist_ok=True)
    name = f"mix-{n}-{s}-{f}-{mix}-{d}-{c}"
    target = root / (name + ".npz")
    if target.exists():
        return
    spec = mixture(s, c)
    box = fixture(f, mix, n, d, "both", "zero", 8, 4, chemistry=spec)
    w = np.ones(n, np.int64)
    start = time.perf_counter()
    exact = flux(box.mass, w, box.volume, box.gas, spec)
    secs = time.perf_counter() - start
    bulk = box.mass.sum(0) / box.volume
    result = dict(
        n=n,
        s=s,
        f=f,
        mix=mix,
        d=d,
        c=c,
        levels=box.levels,
        masks=box.masks,
        t=box.t,
        v=box.volume,
        bulk=bulk,
        exact_flux=exact,
        exact_flux_s=secs,
    )
    for tag, aware in (("size", False), ("composition", True)):
        classes = sorted_classes(box.mass, aware)
        for b in BINS:
            if b > n // 4:
                continue
            m, wt = pack(classes, b)
            np.testing.assert_allclose(
                wt @ m / box.volume, bulk, rtol=3e-12, atol=1e-25
            )
            if wt.sum() != n:
                raise ArithmeticError("Physical count changed while packing")
            field = f"{tag}_{b}"
            result[field + "_mass"] = m
            result[field + "_w"] = wt
            result[field + "_flux"] = flux(m, wt, box.volume, box.gas, spec)
    if n <= 4096:
        result["unit_0_mass"] = box.mass
        result["unit_0_w"] = w
        result["unit_0_flux"] = exact
    tmp = target.with_suffix(".part")
    with tmp.open("wb") as out:
        np.savez_compressed(out, **result)
    tmp.replace(target)


def dop853(m, w, v, levels, masks, t, spec, tol):
    """Integrate physical rates with explicit gas and adaptive steps."""
    model = ExplicitReservoir(m, w, v, spec)
    q = (m[:, : spec.species] / model.local.dry[:, None]).ravel()
    g = levels[0].copy()
    starts = np.r_[0, np.flatnonzero(np.diff(t) == 0) + 1]
    wet, gas = [], []
    for leg, (i, j) in enumerate(
        zip(starts, np.r_[starts[1:], len(t)], strict=True)
    ):
        ts = t[i:j] - t[i]
        g = np.where(masks[leg], levels[leg], g)
        sol = solve_ivp(
            model.rates,
            (0, ts[-1]),
            np.r_[q, g / model.scale],
            method="DOP853",
            t_eval=ts,
            rtol=tol,
            atol=tol * 1e-3,
        )
        if not sol.success or sol.y.min() < -tol * 1e-2:
            raise ArithmeticError(sol.message)
        q = sol.y[: -spec.species, -1]
        g = sol.y[-spec.species :, -1] * model.scale
        qhist = sol.y[: -spec.species].T.reshape(len(ts), len(w), spec.species)
        bulk = np.einsum("i,tis->ts", model.local.w, qhist)
        ghist = sol.y[-spec.species :].T * model.scale
        np.testing.assert_allclose(
            bulk + ghist,
            np.broadcast_to(bulk[0] + ghist[0], bulk.shape),
            rtol=3e-11,
            atol=1e-16,
        )
        wet.append(bulk)
        gas.append(ghist)
    return np.concatenate(wet), np.concatenate(gas)


def advance(p, folder, b, method, tol, tag):
    """Save a full mixing-state pulse at one material and size resolution."""
    target = folder / f"{p.stem}-{tag}-{b}-{method}-{tol:g}.npz"
    if target.exists():
        return
    with np.load(p) as a:
        field = f"{tag}_{b}"
        if field + "_mass" not in a:
            return
        m, w = a[field + "_mass"], a[field + "_w"]
        spec = mixture(int(a["s"]), int(a["c"]))
        tic = time.perf_counter()
        solver = dop853 if method == "DOP853" else integrate
        wet, gas = solver(
            m, w, float(a["v"]), a["levels"], a["masks"], a["t"], spec, tol
        )
        result = {
            k: a[k]
            for k in (
                "n",
                "s",
                "f",
                "mix",
                "d",
                "c",
                "v",
                "t",
                "bulk",
                "exact_flux",
                "exact_flux_s",
            )
        }
        result.update(
            wet=wet,
            gas=gas,
            group_flux=a[field + "_flux"],
            groups=len(w),
            dry_bins=b,
            grouping=tag,
            method=method,
            tolerance=tol,
            elapsed_s=time.perf_counter() - tic,
        )
    tmp = target.with_suffix(".part")
    with tmp.open("wb") as out:
        np.savez_compressed(out, **result)
    tmp.replace(target)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "run"])
    p.add_argument("inputs", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--n", nargs="+", type=int, default=[1000000])
    p.add_argument("--s", nargs="+", type=int, default=[2, 8, 32])
    p.add_argument("--family", nargs="+", type=int, default=[85001, 86011])
    p.add_argument("--density", nargs="+", type=int, default=[1, 8])
    p.add_argument("--chemistry", type=int, default=9177)
    p.add_argument("--bins", nargs="+", type=int, default=list(BINS))
    p.add_argument("--grouping", nargs="+", default=["size", "composition"])
    p.add_argument("--method", choices=["DOP853", "BDF"], default="DOP853")
    p.add_argument("--tol", type=float, default=2e-8)
    p.add_argument(
        "--until",
        required=True,
        type=lambda x: datetime.fromisoformat(x).timestamp(),
    )
    a = p.parse_args()
    if a.action == "prepare":
        for n, s, f, d, m in product(a.n, a.s, a.family, a.density, (0, 1)):
            if time.time() >= a.until:
                break
            prepare(a.inputs, n, s, f, m, d, a.chemistry)
    else:
        if a.output is None:
            p.error("Specify a trajectory output directory")
        a.output.mkdir(parents=True, exist_ok=True)
        for b, n, s, f, d, tag, m in product(
            a.bins, a.n, a.s, a.family, a.density, a.grouping, (0, 1)
        ):
            if time.time() >= a.until:
                break
            src = a.inputs / f"mix-{n}-{s}-{f}-{m}-{d}-{a.chemistry}.npz"
            if src.exists():
                advance(src, a.output, b, a.method, a.tol, tag)
