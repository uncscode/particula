"""Integrate independent water histories at every native particle slot."""

import argparse
from pathlib import Path

import numpy as np
from projection_study import digest
from scipy.integrate import solve_ivp
from stiff_exchange import rhs
from water_constants import KELVIN, MEAN_FREE_PATH, SATURATION


def jacobian(t, ratio, dry, kappa, weight, total):
    """Closed-reservoir derivative of the water-to-dry mass ratio rate."""
    wet = np.maximum(ratio, 0)
    cap = 1000 / 1400 * kappa
    activity = wet / (wet + cap)
    r = np.cbrt(3 * dry * (1 / 1400 + wet / 1000) / (4 * np.pi))
    kn = MEAN_FREE_PATH / r
    b, c = 1 + 0.283 * 0.2, 0.75 * 0.2
    denom = kn**2 + b * kn + c
    k = 4 * np.pi * r * 2.4e-5 * c * (1 + kn) / denom / dry
    kelvin = SATURATION * np.exp(KELVIN / r)
    drive = total - weight @ ratio - kelvin * activity
    rprime = dry / (4 * np.pi * 1000 * r**2)
    kpower = 1 - kn / (1 + kn) + kn * (2 * kn + b) / denom
    surface_prime = kelvin * (
        cap / (wet + cap) ** 2 - activity * KELVIN * rprime / r**2
    )
    diag = k * (drive * kpower * rprime / r - surface_prime) * (ratio >= 0)
    return np.diag(diag) - np.outer(k, weight)


def box(mass, counts, volume, time, supplied, rtol):
    """Resolve consecutive plateaus and carry each particle's hydration."""
    valid = counts > 0
    if not valid.any():
        return (
            np.zeros_like(time),
            np.repeat(
                supplied,
                np.diff(
                    np.r_[0, np.flatnonzero(np.diff(time) == 0) + 1, len(time)]
                ),
            ),
            0,
        )
    m = mass[valid]
    dry = m[:, 1:].sum(-1)
    kappa = (0.03 * m[:, 1] + 0.75 * m[:, 2]) / dry
    weight = dry * counts[valid] / volume
    ratio = m[:, 0] / dry
    starts = np.r_[0, np.flatnonzero(np.diff(time) == 0) + 1]
    ends = np.r_[starts[1:], len(time)]
    waters, gases, work = [], [], 0
    for start, stop, gas in zip(starts, ends, supplied, strict=True):
        total = gas + weight @ ratio
        ts = time[start:stop] - time[start]
        solution = solve_ivp(
            rhs,
            (0, ts[-1]),
            ratio,
            method="BDF",
            t_eval=ts,
            args=(dry, kappa, weight, total),
            jac=jacobian,
            rtol=rtol,
            atol=rtol * 1e-3,
        )
        if not solution.success or not np.isfinite(solution.y).all():
            raise ArithmeticError(solution.message)
        w = weight @ solution.y
        g = total - w
        if solution.y.min() < -rtol * 1e-2 or g.min() < -1e-14:
            raise ArithmeticError("Negative continuous inventory")
        ratio = solution.y[:, -1]
        work += solution.nfev
        waters.append(w)
        gases.append(g)
    return np.concatenate(waters), np.concatenate(gases), work


def run(root, out, blocks, tolerances, first, count, source=None):
    """Publish per-shard results after replaying every prescribed reset."""
    folder = source or root / f"repeated/b{blocks}/data/multicycle/validation"
    files = sorted(folder.glob("families-*.npz"))[first : first + count]
    if not files:
        raise ValueError("No input histories")
    out.mkdir(parents=True, exist_ok=True)
    for p in files:
        sha = digest(p)
        with np.load(p) as a:
            time, mass, counts = a["time"], a["mass"][0], a["counts"]
            starts = np.r_[0, np.flatnonzero(np.diff(time) == 0) + 1]
            supplied = a["gas"][starts].T
            for rtol in tolerances:
                dest = out / f"{p.stem}-r{rtol:.0e}.npz"
                if dest.exists():
                    with np.load(dest) as d:
                        if (
                            sha != d["source_sha256"].item()
                            or d["rtol"] != rtol
                        ):
                            raise ValueError(
                                "Existing reference inputs changed"
                            )
                    continue
                output = [
                    box(m, n, float(a["volume"]), time, g, rtol)
                    for m, n, g in zip(mass, counts, supplied, strict=True)
                ]
                water, gas, work = map(np.stack, zip(*output, strict=True))
                tmp = dest.with_suffix(".tmp.npz")
                np.savez_compressed(
                    tmp,
                    water=water,
                    gas=gas,
                    ids=a["ids"],
                    time=time,
                    evaluations=work,
                    rtol=rtol,
                    source_sha256=sha,
                )
                tmp.replace(dest)
                print(dest.name, int(work.sum()), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--tolerances", type=float, nargs="+", default=[1e-8, 1e-10])
    p.add_argument("--source", type=Path)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--count", type=int, default=16)
    run(**vars(p.parse_args()))
