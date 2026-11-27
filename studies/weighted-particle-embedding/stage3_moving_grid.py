"""Independent growth-following size grid and charge-resolved coagulation."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from stage2_population import brownian
from stage3_charge_modes import charge_mode
from stage3_continuous import RHO, charged, initial
from stage3_spectra import radius_domain


class MovingGrid:
    """Evolve each mass pivot by growth and redistribute mergers by volume."""

    def __init__(
        self, n, s, d, zmode, f, bins, qmax, *, unit_charge_grid=False
    ):
        """Initialize moving volume pivots and indexed collision pairs."""
        m, z, self.g0, diff, alpha, self.path = initial(n, s, f, zmode)
        self.n, self.d, self.b, self.zmode = n, d, bins, zmode
        self.strength = 1.0
        self.base = (4 * np.pi / 3 * RHO) * np.geomspace(
            *radius_domain(m, RHO), bins
        ) ** 3
        pattern, magnitude = charge_mode(zmode)
        self.unit = 1 if unit_charge_grid else max(1, magnitude)
        self.q = (
            np.arange(-qmax, qmax + 1)
            if unit_charge_grid
            else np.array([0])
            if zmode == "neutral"
            else np.arange(0 if pattern == "like" else -qmax, qmax + 1)
        )
        self.nq = len(self.q)
        self.kn_factor = 1 + 0.283 * alpha
        self.alpha = 0.75 * alpha
        self.pref = 4 * np.pi * diff * self.alpha
        hi = np.searchsorted(self.base, m)
        frac = (m - self.base[hi - 1]) / np.diff(self.base)[hi - 1]
        qi = z // self.unit - self.q[0]
        shape = self.b * self.nq
        self.v0 = (
            np.bincount(hi * self.nq + qi, frac, minlength=shape)
            + np.bincount((hi - 1) * self.nq + qi, 1 - frac, minlength=shape)
        ) / n
        self.dry = d * m.mean()
        self.charge0 = z.mean()
        self.prepare_pairs()

    def prepare_pairs(self):
        """Index the native all-pairs size and charge operator."""
        self.i, self.j = np.triu_indices(self.b * self.nq)
        self.ib, self.jb = self.i // self.nq, self.j // self.nq
        charge = np.tile(self.q, self.b)
        qsum = charge[self.i] + charge[self.j]
        self.valid_charge = (qsum >= self.q[0]) & (qsum <= self.q[-1])
        self.qi = np.clip(qsum - self.q[0], 0, self.nq - 1)
        self.charges = charge * self.unit
        self.pair_scale = self.d * np.where(self.i == self.j, 0.5, 1.0)

    def unpack(self, y):
        """Read pivot masses, bin probabilities and finite gas densities."""
        n = self.b
        end = n + len(self.v0)
        return self.base * y[:n], y[n:end], self.g0 * np.exp(y[end:-1])

    def rhs(self, _, y):
        """Growth follows characteristics while collisions deposit locally."""
        pivot, v, gas = self.unpack(y)
        rad = np.cbrt(pivot / (4 * np.pi / 3 * RHO))
        kn = self.path / rad[:, None]
        cond = (
            rad[:, None]
            * self.pref
            * (1 + kn)
            / (kn * kn + self.kn_factor * kn + self.alpha)
        )
        w = v.reshape(self.b, self.nq).sum(1)
        if self.strength == 0:
            return np.r_[
                (cond @ gas) / self.base,
                np.zeros_like(v),
                -self.d * (w @ cond),
                0.0,
            ]
        dv, lost = self.collision_tendency(rad, pivot, v)
        return np.r_[(cond @ gas) / self.base, dv, -self.d * (w @ cond), lost]

    def collision_tendency(self, rad, pivot, v):
        """Deposit resolved binary mergers and return the truncated hazard."""
        if self.nq == 1:
            kernel = brownian(rad, pivot, self.i, self.j)
        else:
            kernel = charged(
                np.repeat(rad, self.nq),
                np.repeat(pivot, self.nq),
                self.charges,
                self.i,
                self.j,
            )
        rate = self.strength * self.pair_scale * kernel * v[self.i] * v[self.j]
        merged = pivot[self.ib] + pivot[self.jb]
        inside = (merged <= pivot[-1]) & self.valid_charge
        lost = rate[~inside].sum()
        rate = np.where(inside, rate, 0.0)
        hi = np.clip(np.searchsorted(pivot, merged), 1, self.b - 1)
        frac = (merged - pivot[hi - 1]) / np.diff(pivot)[hi - 1]
        target = hi * self.nq + self.qi
        dv = np.bincount(target, rate * frac, minlength=len(v)) + np.bincount(
            target - self.nq, rate * (1 - frac), minlength=len(v)
        )
        dv -= np.bincount(self.i, rate, minlength=len(v)) + np.bincount(
            self.j, rate, minlength=len(v)
        )
        return dv, lost

    def observe(self, y, *, expected_charge=None):
        """Measure inventories, truncation and positivity."""
        pivot, v, g = self.unpack(y)
        by = v.reshape(self.b, self.nq)
        w = by.sum(1)
        rad = np.cbrt(pivot / (4 * np.pi / 3 * RHO))
        target = self.charge0 if expected_charge is None else expected_charge
        charge_error = abs(by.sum(0) @ self.q * self.unit - target)
        ledger = abs(
            self.d * (w @ pivot) + g.sum() - self.dry - self.g0.sum()
        ) / (self.dry + self.g0.sum())
        if (
            np.any(np.diff(pivot) <= 0)
            or not np.isfinite(y).all()
            or min(w.min(), v.min()) < -1e-8
            or max(ledger, charge_error) > 2e-7
        ):
            raise ArithmeticError(
                "Moving grid lost ordering, positivity, charge or mass"
            )
        bands = np.bincount(
            np.searchsorted([35e-9, 70e-9, 120e-9], rad, side="right"),
            w,
            minlength=4,
        )
        return dict(
            n=self.n * w.sum(),
            gas=g.tolist(),
            particle=np.r_[self.dry, self.g0 - g].tolist(),
            bands=(self.n * bands).tolist(),
            charge_abs=self.n * (by.sum(0) @ abs(self.q)) * self.unit,
            net_charge_error_per_initial_particle=charge_error,
            ledger=ledger,
            terminal_fraction=w[-1],
            omitted_mergers_per_initial_particle=y[-1],
        )


def work(job):
    """Save each refinement and numerical failure."""
    folder, n, s, d, z, f, h, b, q, tol = job
    path = folder / f"moving-{n}-{s}-{d:g}-{z}-{f}-{h:g}-{b}-{q}-{tol:g}.json"
    if path.exists():
        return
    row = dict(
        n=n,
        species=s,
        density=d,
        charge=z,
        family=f,
        horizon=h,
        bins=b,
        charge_extent=q,
        tol=tol,
        kernel="physical",
    )
    tic = perf_counter()
    try:
        model = MovingGrid(n, s, d, z, f, b, q)
        sol = solve_ivp(
            model.rhs,
            (0, h),
            np.r_[np.ones(b), model.v0, np.zeros(s + 1)],
            rtol=tol,
            atol=tol * 1e-5,
            t_eval=np.linspace(0, h, 5),
            method="DOP853",
        )
        if not sol.success:
            raise ArithmeticError(sol.message)
        hist = [model.observe(y) for y in sol.y.T]
        row.update(
            status="completed",
            history=hist,
            nfev=sol.nfev,
            truncation_ok=bool(
                max(
                    max(
                        v["omitted_mergers_per_initial_particle"],
                        v["terminal_fraction"],
                    )
                    for v in hist
                )
                < 1e-7
            ),
        )
    except (ArithmeticError, ValueError) as e:
        row.update(status="failed", why=str(e))
    row["seconds"] = perf_counter() - tic
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(row) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    for k, typ, vals in [
        ("n", int, [1000000]),
        ("s", int, [2, 32]),
        ("density", float, [1e12]),
        ("charge", str, ["neutral"]),
        ("family", int, [101, 307, 411, 512]),
        ("h", float, [1.0]),
        ("bins", int, [48, 96, 192]),
        ("qmax", int, [3]),
        ("tol", float, [2e-8]),
    ]:
        p.add_argument("--" + k, type=typ, nargs="+", default=vals)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    jobs = product(
        [a.folder],
        a.n,
        a.s,
        a.density,
        a.charge,
        a.family,
        a.h,
        a.bins,
        a.qmax,
        a.tol,
    )
    with ProcessPoolExecutor(a.workers) as pool:
        for i, _ in enumerate(pool.map(work, jobs), 1):
            if i % 10 == 0:
                print(i, flush=True)
