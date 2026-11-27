"""Continuously couple a finite bipolar ion supply to a growing aerosol."""

import argparse
import json
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from stage3_ion_clocks import ION_MASS, PRESSURE, RECOMB, T, mobility_attachment
from stage3_moving_grid import MovingGrid

MOBILITY = (1.4e-4, 1.9e-4)
POLICIES = ("live", "fixed", "no-collisions")


def attach(radius, charge, sign):
    """Evaluate the native mobility attachment coefficient in cubic metres/s."""
    lane = int(sign < 0)
    mass = (4 * np.pi / 3 * 1400) * radius**3
    return mobility_attachment(
        radius, mass, charge, T, PRESSURE, ION_MASS[lane], MOBILITY[lane], sign
    )


class ChargingGrid:
    """Transport aerosol occupation along integer charge and wet volume."""

    def __init__(
        self, n, s, z, f, bins, qmax, policy, *, grid_class=MovingGrid
    ):
        """Initialize charge states and one free ion per sign per host."""
        self.base = grid_class(
            n, s, 1e12, z, f, bins, qmax, unit_charge_grid=True
        )
        self.base.strength = float(policy != "no-collisions")
        self.policy = policy
        b = self.base
        self.y0 = np.r_[np.ones(b.b), b.v0, np.zeros(s + 1), 1, 1, 0, 0, 0, 0]

    def rhs(self, t, y):
        """Advance gas growth, collisions, ion capture and recombination."""
        b = self.base
        base = y[:-6]
        fp, fn = np.maximum(0, y[-6:-4])
        deriv = b.rhs(t, base)
        recomb = RECOMB * b.d * fp * fn
        if self.policy == "fixed":
            return np.r_[deriv, -recomb, -recomb, recomb, 0, 0, 0]
        pivot, v, _ = b.unpack(base)
        rad = np.cbrt(pivot / (4 * np.pi / 3 * 1400))[:, None]
        by = v.reshape(b.b, b.nq)
        plus = b.d * fp * attach(rad, b.q, 1) * by
        minus = b.d * fn * attach(rad, b.q, -1) * by
        blocked = plus[:, -1].sum() + minus[:, 0].sum()
        plus[:, -1] = 0
        minus[:, 0] = 0
        dv = deriv[b.b : b.b + v.size].reshape(by.shape)
        dv -= plus + minus
        dv[:, 1:] += plus[:, :-1]
        dv[:, :-1] += minus[:, 1:]
        ap, an = plus.sum(), minus.sum()
        return np.r_[deriv, -recomb - ap, -recomb - an, recomb, ap, an, blocked]

    def observe(self, y):
        """Recover aerosol and ion budgets independently across all phases."""
        b = self.base
        fp, fn, rec, ap, an, blocked = y[-6:]
        h = b.observe(y[:-6], expected_charge=b.charge0 - fp + fn)
        _, v, _ = b.unpack(y[:-6])
        counts = v.reshape(b.b, b.nq).sum(0)
        drift = max(abs(fp + ap + rec - 1), abs(fn + an + rec - 1))
        if min(fp, fn, rec, ap, an) < -1e-8 or drift > 1e-8:
            raise ArithmeticError("Continuous ion budget failed")
        h.update(
            free_positive=fp * b.n,
            free_negative=fn * b.n,
            recombined=rec * b.n,
            attached_positive=ap * b.n,
            attached_negative=an * b.n,
            charged_count=(counts[b.q != 0].sum() * b.n),
            charge_hist=(
                np.bincount(np.clip(b.q + 9, 0, 18), counts, minlength=19) * b.n
            ).tolist(),
            mean_abs_charge=np.dot(counts, abs(b.q)) / counts.sum(),
            ion_ledger=drift,
            omitted_attachments_per_initial_particle=blocked,
        )
        return h


def solve(
    n,
    s,
    z,
    f,
    bins,
    qmax,
    policy,
    horizon=1.0,
    tol=2e-8,
    *,
    grid_class=MovingGrid,
):
    """Integrate the joint physics through five observations."""
    m = ChargingGrid(n, s, z, f, bins, qmax, policy, grid_class=grid_class)
    sol = solve_ivp(
        m.rhs,
        (0, horizon),
        m.y0,
        method="DOP853",
        rtol=tol,
        atol=tol * 1e-5,
        t_eval=np.linspace(0, horizon, 5),
    )
    if not sol.success:
        raise ArithmeticError(sol.message)
    h = [m.observe(v) for v in sol.y.T]
    lost = max(
        max(
            r["omitted_mergers_per_initial_particle"],
            r["omitted_attachments_per_initial_particle"],
            r["terminal_fraction"],
        )
        for r in h
    )
    return dict(
        history=h,
        nfev=sol.nfev,
        truncation_ok=bool(lost < 1e-6),
        max_truncated_fraction=lost,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--n", type=int, nargs="+", default=[1000000])
    p.add_argument("--species", type=int, nargs="+", default=[32])
    p.add_argument("--charge", nargs="+", default=["neutral", "bipolar4"])
    p.add_argument("--family", type=int, default=307)
    p.add_argument("--bins", nargs="+", type=int, default=[24, 48, 72])
    p.add_argument("--qmax", nargs="+", type=int, default=[6, 8])
    p.add_argument(
        "--policy", nargs="+", choices=POLICIES, default=list(POLICIES)
    )
    p.add_argument("--tol", nargs="+", type=float, default=[2e-8])
    p.add_argument(
        "--until",
        type=lambda x: datetime.fromisoformat(x).timestamp(),
        required=True,
    )
    a = p.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    for bins, qmax, n, s, z, mode, tol in product(
        a.bins, a.qmax, a.n, a.species, a.charge, a.policy, a.tol
    ):
        if time.time() >= a.until:
            break
        path = (
            a.folder
            / f"grid-{n}-{s}-{z}-{a.family}-{mode}-{bins}-{qmax}-{tol}.json"
        )
        if path.exists():
            continue
        args = dict(
            n=n,
            species=s,
            charge=z,
            family=a.family,
            policy=mode,
            bins=bins,
            charge_extent=qmax,
            tol=tol,
            density=1e12,
            horizon=1.0,
        )
        start = time.perf_counter()
        try:
            r = args | solve(n, s, z, a.family, bins, qmax, mode, tol=tol)
            r["status"] = "completed"
        except (ArithmeticError, RuntimeError, ValueError) as err:
            r = args | dict(status="failed", why=str(err))
        r["elapsed_s"] = time.perf_counter() - start
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(r) + "\n")
        tmp.replace(path)
        print(path.name, r["status"], r["elapsed_s"], flush=True)
