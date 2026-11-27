"""Independent stiff reference for two vapors coupled by finite gas and
activity.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from dual_materials import (
    ACCOMMODATION,
    DENSITY,
    DIFFUSIVITY,
    MOLAR,
    SATURATED,
    SURFACE_DENSITY,
    SURFACE_TENSION,
    TEMPERATURE,
    R,
)
from scipy.integrate import solve_ivp
from water_constants import MEAN_FREE_PATH

PATH = MEAN_FREE_PATH * np.sqrt(MOLAR[0] / MOLAR[:2])
KELVIN = 2 * SURFACE_TENSION * MOLAR[:2] / (SURFACE_DENSITY * R * TEMPERATURE)


def rate(ratio, fixed, weight, total, initial, separate, derivatives=False):
    """Calculate the native ideal activities, transition transport and their
    Jacobian.
    """
    dry = fixed.sum(1)
    q = ratio.reshape(-1, 2)
    gas = total - weight @ q
    base_volume = (fixed / DENSITY[2:]).sum(1)
    base_moles = (fixed / MOLAR[2:]).sum(1)
    result = np.empty_like(q)
    local = np.zeros((len(q), 2, 2))
    gain = np.empty_like(q)
    for s in range(2):
        values = (
            q
            if not separate
            else np.column_stack(
                [q[:, t] if t == s else initial[:, t] for t in range(2)]
            )
        )
        # Clip trial states. The integration checks every accepted inventory.
        values = values.clip(0)
        vol = base_volume + dry * (values / DENSITY[:2]).sum(1)
        radius = np.cbrt(3 * vol / (4 * np.pi))
        moles = base_moles / dry + (values / MOLAR[:2]).sum(1)
        activity = values[:, s] / (MOLAR[s] * moles)
        kn, alpha = PATH[s] / radius, ACCOMMODATION[s]
        denom = kn * kn + (1 + 0.283 * alpha) * kn + 0.75 * alpha
        gain[:, s] = (
            4
            * np.pi
            * radius
            * DIFFUSIVITY[s]
            * 0.75
            * alpha
            * (1 + kn)
            / (denom * dry)
        )
        sat = SATURATED[s] * np.exp(KELVIN[s] / radius)
        drive = gas[s] - sat * activity
        result[:, s] = gain[:, s] * drive
        if derivatives:
            power = (
                1 - kn / (1 + kn) + kn * (2 * kn + 1 + 0.283 * alpha) / denom
            )
            for t in range(2):
                if separate and s != t:
                    continue
                dr = dry / (4 * np.pi * radius**2 * DENSITY[t])
                dactivity = (
                    (1 if s == t else 0) / MOLAR[s] - activity / MOLAR[t]
                ) / moles
                local[:, s, t] = (
                    gain[:, s]
                    * (
                        drive * power * dr / radius
                        - sat
                        * (dactivity - activity * KELVIN[s] * dr / radius**2)
                    )
                    * (q[:, t] >= 0)
                )
    if not derivatives:
        return result.ravel()
    jac = np.zeros((len(q), 2, len(q), 2))
    i = np.arange(len(q))
    jac[i, :, i, :] = local
    for s in range(2):
        jac[:, s, :, s] -= np.outer(gain[:, s], weight)
    return jac.reshape(q.size, q.size)


def box(
    mass, counts, volume, times, forcing, tolerance, separate=False, mask=None
):
    """Integrate both volatile masses through prescribed gas resets."""
    mask = np.ones_like(forcing, dtype=bool) if mask is None else mask
    forcing = np.array(forcing, copy=True)
    active = counts > 0
    if not active.any():
        starts = np.r_[0, np.flatnonzero(np.diff(times) == 0) + 1, len(times)]
        for i in range(1, len(forcing)):
            forcing[i] = np.where(mask[i], forcing[i], forcing[i - 1])
        return (
            np.zeros((len(times), 2)),
            np.repeat(forcing, np.diff(starts), axis=0),
            0,
        )
    m = mass[active]
    fixed = m[:, 2:]
    dry = fixed.sum(1)
    initial = m[:, :2] / dry[:, None]
    y = initial.ravel()
    weight = counts[active] * dry / volume
    start = np.r_[0, np.flatnonzero(np.diff(times) == 0) + 1]
    wout, gout, work = [], [], 0
    for leg, (j, end, gas) in enumerate(
        zip(start, np.r_[start[1:], len(times)], forcing, strict=True)
    ):
        ts = times[j:end] - times[j]
        if leg:
            gas = np.where(mask[leg], gas, gout[-1][-1])
        total = gas + weight @ y.reshape(-1, 2)
        args = (fixed, weight, total, initial, separate)
        sol = solve_ivp(
            lambda t, x, args=args: rate(x, *args),
            (0, ts[-1]),
            y,
            t_eval=ts,
            method="BDF",
            jac=lambda t, x, args=args: rate(x, *args, derivatives=True),
            rtol=tolerance,
            atol=tolerance * 1e-3,
        )
        if not sol.success or sol.y.min() < -tolerance * 1e-2:
            raise ArithmeticError(sol.message)
        bulk = np.einsum(
            "i,its->ts",
            weight,
            sol.y.T.reshape(len(ts), len(m), 2).transpose(1, 0, 2),
        )
        vapor = total - bulk
        if vapor.min() < -1e-14 or not np.isfinite(vapor).all():
            raise ArithmeticError("Invalid independent two-vapor reservoir")
        wout.append(bulk)
        gout.append(vapor)
        y = sol.y[:, -1]
        work += sol.nfev
    return np.concatenate(wout), np.concatenate(gout), work


def run(source, out, tolerances, separate):
    """Save independent reference solutions with source fingerprints."""
    out.mkdir(parents=True, exist_ok=True)
    for file in sorted(source.glob("families-*.npz")):
        with np.load(file) as a:
            for tol in tolerances:
                label = "isolated" if separate else "coupled"
                dest = out / f"{file.stem}-r{tol:.0e}-{label}.npz"
                sha = hashlib.sha256(file.read_bytes()).hexdigest()
                if dest.exists():
                    with np.load(dest) as b:
                        if b["source_sha256"] != sha:
                            raise ValueError("Two-vapor inputs changed")
                    continue
                masks = (
                    a["reset_mask"]
                    if "reset_mask" in a
                    else np.ones_like(a["levels"], dtype=bool)
                )
                values = [
                    box(
                        mi,
                        ni,
                        float(a["volume"]),
                        a["time"],
                        levels,
                        tol,
                        separate,
                        mask,
                    )
                    for mi, ni, levels, mask in zip(
                        a["mass0"],
                        a["counts"],
                        a["levels"].repeat(2, 0),
                        masks.repeat(2, 0),
                        strict=True,
                    )
                ]
                w, g, steps = map(np.stack, zip(*values, strict=True))
                pending = dest.with_suffix(".tmp.npz")
                np.savez_compressed(
                    pending,
                    water=w,
                    gas=g,
                    steps=steps,
                    ids=a["ids"],
                    time=a["time"],
                    tolerance=tol,
                    separate=separate,
                    source_sha256=sha,
                )
                pending.replace(dest)
                print(dest.name, steps.sum(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--tolerances", type=float, nargs="+", default=[1e-7, 1e-9])
    p.add_argument("--separate", action="store_true")
    run(**vars(p.parse_args()))
