"""Independent stiff equations for an arbitrary number of volatile materials."""

import argparse
from pathlib import Path

import numpy as np
from dual_materials import SURFACE_DENSITY, SURFACE_TENSION, TEMPERATURE, R
from projection_study import digest
from scipy.integrate import solve_ivp
from vapor_properties import materials
from water_constants import MEAN_FREE_PATH


class Rates:
    """Evaluate local transfer and the full shared-reservoir Jacobian."""

    def __init__(self, mass, counts, volume, spec):
        """Prepare dry particle constants independent of the solver steps."""
        s = spec.species
        self.c = s
        self.spec = spec
        self.dry = mass[:, s:].sum(1)
        self.w = counts * self.dry / volume
        self.base_vol = (mass[:, s:] / spec.density[s:]).sum(1)
        self.base_moles = (mass[:, s:] / spec.molar[s:]).sum(1) / self.dry
        self.path = MEAN_FREE_PATH * np.sqrt(0.01801528 / spec.molar[:s])
        self.kelvin = (
            2
            * SURFACE_TENSION
            * spec.molar[:s]
            / (SURFACE_DENSITY * R * TEMPERATURE)
        )

    def local(self, y, gas, jac=False):
        """Evaluate per-particle rates or the local Jacobian and gas gain."""
        s = self.c
        q = y.reshape(-1, s)
        z = q.clip(0)
        p = self.spec
        volume = self.base_vol + self.dry * (z / p.density[:s]).sum(1)
        radius = np.cbrt(3 * volume / (4 * np.pi))[:, None]
        mol = self.base_moles[:, None] + (z / p.molar[:s]).sum(1, keepdims=True)
        activity = z / (mol * p.molar[:s])
        ke = np.exp(self.kelvin / radius) * p.saturation
        kn = self.path / radius
        denom = (
            kn**2 + (1 + 0.283 * p.accommodation) * kn + 0.75 * p.accommodation
        )
        gain = (
            4
            * np.pi
            * radius
            * p.diffusivity
            * 0.75
            * p.accommodation
            * (1 + kn)
            / (denom * self.dry[:, None])
        )
        drive = gas - ke * activity
        if not jac:
            return (gain * drive).ravel()
        power = (
            1
            - kn / (1 + kn)
            + (2 * kn + 1 + 0.283 * p.accommodation) * kn / denom
        )
        dr = self.dry[:, None] / (4 * np.pi * radius**2 * p.density[:s])
        dx = (
            np.eye(s) / p.molar[:s] - activity[:, :, None] / p.molar[:s]
        ) / mol[:, :, None]
        term = (
            drive * power / radius + ke * activity * self.kelvin / radius**2
        )[:, :, None] * dr[:, None]
        local = (
            gain[:, :, None] * (term - ke[:, :, None] * dx) * (q >= 0)[:, None]
        )
        return local, gain

    def evaluate(self, y, total, jac=False):
        """Return the rate or dense derivative after eliminating the gas."""
        s = self.c
        q = y.reshape(-1, s)
        gas = total - self.w @ q
        value = self.local(y, gas, jac)
        if not jac:
            return value
        local, gain = value
        j = np.zeros((len(q), s, len(q), s))
        ix = np.arange(len(q))
        j[ix, :, ix, :] = local
        for i in range(s):
            j[:, i, :, i] -= np.outer(gain[:, i], self.w)
        return j.reshape(y.size, y.size)


def integrate(m, n, volume, levels, mask, t, spec, tol):
    """Carry all reservoirs across partially prescribed vapor changes."""
    valid = n > 0
    m, n = m[valid], n[valid]
    rates = Rates(m, n, volume, spec)
    q = (m[:, : spec.species] / rates.dry[:, None]).ravel()
    starts = np.r_[0, np.flatnonzero(np.diff(t) == 0) + 1]
    wet, gases = [], []
    for leg, (i, j) in enumerate(
        zip(starts, np.r_[starts[1:], len(t)], strict=True)
    ):
        gas = (
            levels[leg]
            if leg == 0
            else np.where(mask[leg], levels[leg], gases[-1][-1])
        )
        total = gas + rates.w @ q.reshape(-1, spec.species)
        ts = t[i:j] - t[i]
        sol = solve_ivp(
            lambda _, y, total=total: rates.evaluate(y, total),
            (0, ts[-1]),
            q,
            method="BDF",
            jac=lambda _, y, total=total: rates.evaluate(y, total, True),
            t_eval=ts,
            rtol=tol,
            atol=tol * 1e-3,
        )
        if not sol.success or sol.y.min() < -tol * 1e-2:
            raise ArithmeticError(sol.message)
        bulk = np.einsum(
            "i,tis->ts", rates.w, sol.y.T.reshape(len(ts), len(m), spec.species)
        )
        g = total - bulk
        if g.min() < -1e-14 or not np.isfinite(g).all():
            raise ArithmeticError("Invalid vapor reservoir")
        wet.append(bulk)
        gases.append(g)
        q = sol.y[:, -1]
    return np.concatenate(wet), np.concatenate(gases)


def run(
    source, out, count, tolerances, spec=None, limit=None, explicit_gas=False
):
    """Save independent trajectories at both integration tolerances."""
    if explicit_gas:
        from sparse_vapor_reference import integrate as solve
    else:
        solve = integrate
    formulation = (
        "explicit-gas-sparse" if explicit_gas else "eliminated-gas-dense"
    )
    out.mkdir(parents=True, exist_ok=True)
    spec = materials(count) if spec is None else spec
    if spec.species != count or (limit is not None and limit < 1):
        raise ValueError("Invalid reference cohort")
    processed = 0
    for file in sorted(source.glob("families-*.npz")):
        if limit is not None and processed >= limit:
            break
        with np.load(file) as a:
            ids = a["ids"][: None if limit is None else limit - processed]
            processed += len(ids)
            if "pressure_Pa" in a:
                np.testing.assert_array_equal(a["pressure_Pa"], spec.pressure)
            for tol in tolerances:
                dst = out / f"{file.stem}-r{tol:.0e}-coupled.npz"
                sha = digest(file)
                if dst.exists():
                    with np.load(dst) as old:
                        np.testing.assert_array_equal(old["ids"], ids)
                        previous = (
                            str(old["formulation"])
                            if "formulation" in old
                            else "eliminated-gas-dense"
                        )
                        if (
                            previous != formulation
                            or old["source_sha256"] != sha
                        ):
                            raise ValueError("Changed native source")
                    continue
                pairs = [
                    solve(
                        m,
                        n,
                        float(a["volume"]),
                        levels,
                        mask,
                        a["time"],
                        spec,
                        tol,
                    )
                    for m, n, levels, mask in zip(
                        a["mass0"][: 2 * len(ids)],
                        a["counts"][: 2 * len(ids)],
                        a["levels"][: len(ids)].repeat(2, 0),
                        a["reset_mask"][: len(ids)].repeat(2, 0),
                        strict=True,
                    )
                ]
                wet, gas = map(np.stack, zip(*pairs, strict=True))
                tmp = dst.with_suffix(".tmp.npz")
                np.savez_compressed(
                    tmp,
                    water=wet,
                    gas=gas,
                    source_sha256=sha,
                    formulation=formulation,
                    tolerance=tol,
                    ids=ids,
                    time=a["time"],
                    species=spec.names,
                )
                tmp.replace(dst)
                print(dst.name, flush=True)

    if processed == 0 or (limit is not None and processed != limit):
        raise ValueError("Incomplete native cohort for BDF integration")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("count", type=int)
    p.add_argument("--explicit-gas", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--tolerances", type=float, nargs="+", default=[1e-7, 1e-9])
    run(**vars(p.parse_args()))
