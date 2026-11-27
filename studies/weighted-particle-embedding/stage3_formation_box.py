"""Competition for two vapors and coagulation during new particle formation."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from particula.gas import get_molecule_mean_free_path
from scipy.constants import Avogadro
from scipy.integrate import solve_ivp
from stage2_population import brownian
from stage3_birth_growth import D0, MOLAR, RHO


class FormationBox:
    """Keep newborn-only and background-containing lineages separate."""

    def __init__(
        self,
        bins,
        precursor,
        organic,
        background,
        rbg,
        rate,
        law,
        collision,
        advection="upwind",
    ):
        """Build the source, gas reservoirs and two-lineage population grid."""
        self.b = bins
        self.c0 = np.array([precursor, organic])
        self.rate = rate
        self.law = law
        self.collision = collision
        self.advection = advection
        self.base_number = 1e11
        self.r = np.geomspace(D0 / 2, 0.6e-6, bins)
        self.pivot = RHO * 4 * np.pi / 3 * self.r**3
        self.seed = self.pivot[0]
        self.w = np.zeros((2, bins))
        hi = np.searchsorted(self.r, rbg * 1e-9)
        mass = 4 * np.pi / 3 * RHO * (rbg * 1e-9) ** 3
        upper = (mass - self.pivot[hi - 1]) / (
            self.pivot[hi] - self.pivot[hi - 1]
        )
        self.w[1, hi - 1 : hi + 1] = (
            background / self.base_number * np.array([1 - upper, upper])
        )
        self.dry = background * mass
        diff = np.array([1e-5, 0.65e-5])
        alpha = np.array([1.0, 0.55])
        path = np.asarray(
            get_molecule_mean_free_path(
                molar_mass=np.array([MOLAR, 0.18]),
                temperature=293.0,
                pressure=101325.0,
            )
        )
        kn = path / self.r[:, None]
        self.k = (
            4
            * np.pi
            * self.r[:, None]
            * diff
            * 0.75
            * alpha
            * (1 + kn)
            / (kn**2 + (1 + 0.283 * alpha) * kn + 0.75 * alpha)
        )
        self.mass_width = np.diff(self.pivot)
        if advection == "limited":
            log_width = np.log(self.pivot[1] / self.pivot[0])
            face_mass = np.sqrt(self.pivot[:-1] * self.pivot[1:])
            face_k = np.sqrt(self.k[:-1] * self.k[1:])
            self.face_rate = face_k / (face_mass[:, None] * log_width)
        elif advection != "upwind":
            raise ValueError(f"Unknown growth advection {advection}")
        if collision != "off":
            i, j = np.triu_indices(bins * 2)
            ib, jb = i % bins, j % bins
            include = (
                (i // bins != j // bins)
                if collision == "scavenging"
                else np.ones(len(i), bool)
            )
            self.i, self.j = i[include], j[include]
            ib, jb = ib[include], jb[include]
            self.ij = ib, jb
            self.beta = (
                brownian(self.r, self.pivot, ib, jb)
                * self.base_number
                * np.where(self.i == self.j, 0.5, 1.0)
            )
            target_mass = self.pivot[ib] + self.pivot[jb]
            h = np.clip(np.searchsorted(self.pivot, target_mass), 1, bins - 1)
            self.f = (target_mass - self.pivot[h - 1]) / np.diff(self.pivot)[
                h - 1
            ]
            self.dest = h + bins * (self.j // bins)
            self.cross = self.i // bins != self.j // bins
            self.valid = target_mass <= self.pivot[-1]

    def growth_flux(self, v, gas):
        """Transport number along mass pivots while preserving gas uptake."""
        if self.advection == "upwind":
            return v[:, :-1] * (self.k[:-1] @ gas) / self.mass_width
        delta = np.diff(v, axis=1)
        left, right = delta[:, :-1], delta[:, 1:]
        slope = np.where(
            left * right > 0,
            np.sign(left) * np.minimum(abs(left), abs(right)),
            0,
        )
        face = v[:, :-1].copy()
        face[:, 1:] += 0.5 * slope
        intended = v[:, :-1] @ self.k[:-1]
        inferred = face @ (self.face_rate * self.mass_width[:, None])
        scale = np.divide(
            intended, inferred, out=np.zeros_like(intended), where=inferred != 0
        )
        return face * ((scale * gas) @ self.face_rate.T)

    def rhs(self, _, y):
        """Compute births, growth, merger losses and precursor uptake."""
        b = self.b
        v = y[: 2 * b].reshape(2, b)
        c = self.c0 * np.exp(y[2 * b : 2 * b + 2])
        scale = self.base_number
        # J/C remains finite when the closed reservoir is nearly exhausted.
        per_mass = (
            self.rate
            * Avogadro
            / MOLAR
            * (1 if self.law == "activation" else c[0] / 1e-9)
        )
        source = c[0] * per_mass / scale
        grow = self.growth_flux(v, c)
        dv = np.zeros_like(v)
        dv[:, :-1] -= grow
        dv[:, 1:] += grow
        dv[0, 0] += source
        dv = dv.ravel()
        total = cross = overflow = 0.0
        if self.collision != "off":
            w = v.ravel()
            rate = self.beta * w[self.i] * w[self.j]
            overflow = rate[~self.valid].sum()
            rate = np.where(self.valid, rate, 0.0)
            total = rate.sum()
            cross = rate[self.cross].sum()
            dv += np.bincount(
                self.dest, rate * self.f, minlength=2 * b
            ) + np.bincount(self.dest - 1, rate * (1 - self.f), minlength=2 * b)
            dv -= np.bincount(self.i, rate, minlength=2 * b) + np.bincount(
                self.j, rate, minlength=2 * b
            )
        dlog = -scale * v[:, :-1].sum(0) @ self.k[:-1]
        dlog[0] -= per_mass * self.seed
        return np.r_[dv, dlog, source, total, cross, overflow]

    def run(self, end, tol):
        """Integrate and audit number and phase transfer."""
        b = self.b
        end = int(end)
        sol = solve_ivp(
            self.rhs,
            (0, end),
            np.r_[self.w.ravel(), np.zeros(6)],
            t_eval=np.unique(
                np.r_[
                    0,
                    np.geomspace(0.05, end, 40),
                    np.arange(0, end + 1, 2),
                    end,
                ]
            ),
            method="DOP853",
            rtol=tol,
            atol=tol * 1e-5,
        )
        if not sol.success:
            raise ArithmeticError(sol.message)
        history = []
        for t, y in zip(sol.t, sol.y.T, strict=True):
            v = y[: 2 * b].reshape(2, b) * self.base_number
            c = self.c0 * np.exp(y[2 * b : 2 * b + 2])
            birth, merges, cross, overflow = y[-4:] * self.base_number
            particle = v.sum(0) @ self.pivot
            gap = abs(particle + c.sum() - self.dry - self.c0.sum()) / max(
                1e-30, self.dry + self.c0.sum()
            )
            number = abs(
                v.sum() - (self.w.sum() * self.base_number + birth - merges)
            ) / max(1.0, v.sum())
            if max(number, gap) > 2e-6 or v.min() < -max(1e-3, 1e-8 * v.sum()):
                raise ArithmeticError(
                    f"Birth balance error {number}, {gap}, {v.min()}"
                )
            # Integrate a piecewise-constant distribution in log diameter
            # within the reporting bins only; mass pivots keep full precision.
            ld = np.log(2e9 * self.r)
            step = ld[1] - ld[0]
            counts = [
                v[0] @ np.clip((ld + 0.5 * step - np.log(d)) / step, 0, 1)
                for d in (5, 10)
            ]
            history.append(
                dict(
                    t=t,
                    number_m3=v.sum(),
                    newborn_only_m3=v[0].sum(),
                    above_5nm_new_m3=counts[0],
                    above_10nm_new_m3=counts[1],
                    background_hosts_m3=v[1].sum(),
                    precursor_gas_kg_m3=c[0],
                    organic_gas_kg_m3=c[1],
                    precursor_particle_kg_m3=self.c0[0] - c[0],
                    organic_particle_kg_m3=self.c0[1] - c[1],
                    births_m3=birth,
                    mergers_m3=merges,
                    capture_events_m3=cross,
                    ledger=gap,
                    count_balance=number,
                    terminal_fraction=v[:, -1].sum() / max(v.sum(), 1.0),
                    missed_mergers_fraction=overflow / max(v.sum(), 1.0),
                )
            )
        return history


def run_case(job):
    """Save each experiment and any failure at the chosen size cutoff."""
    folder, b, p, o, bg, rbg, rate, law, k, end, tol, advection = job
    suffix = "" if advection == "upwind" else "-" + advection
    path = folder / (
        f"formation-{b}-{p:g}-{o:g}-{bg:g}-{rbg:g}-"
        f"{rate:g}-{law}-{k}-{end}-{tol:g}{suffix}.json"
    )
    if path.exists():
        return
    row = dict(
        bins=b,
        precursor=p,
        organic=o,
        background=bg,
        background_radius_nm=rbg,
        rate=rate,
        law=law,
        collision=k,
        horizon=end,
        tol=tol,
        growth_advection=advection,
    )
    tic = perf_counter()
    try:
        history = FormationBox(b, p, o, bg, rbg, rate, law, k, advection).run(
            end, tol
        )
        row.update(
            status="completed",
            history=history,
            truncation_ok=bool(
                max(
                    max(v["terminal_fraction"], v["missed_mergers_fraction"])
                    for v in history
                )
                < 1e-7
            ),
        )
    except (ArithmeticError, ValueError) as error:
        row.update(status="failed", why=str(error))
    row["seconds"] = perf_counter() - tic
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(row) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    for k, typ, v in [
        ("bins", int, [48, 96]),
        ("precursor", float, [1e-9]),
        ("organic", float, [0.0, 2e-9]),
        ("background", float, [0, 1e9, 1e10, 1e11]),
        ("radius", float, [60.0]),
        ("rate", float, [1e-7]),
        ("laws", str, ["activation", "kinetic"]),
        ("collision", str, ["physical", "scavenging", "off"]),
        ("end", int, [128]),
        ("tol", float, [2e-8]),
    ]:
        p.add_argument("--" + k, type=typ, nargs="+", default=v)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--advection", choices=("upwind", "limited"), default="upwind"
    )
    a = p.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    jobs = product(
        [a.folder],
        a.bins,
        a.precursor,
        a.organic,
        a.background,
        a.radius,
        a.rate,
        a.laws,
        a.collision,
        a.end,
        a.tol,
        [a.advection],
    )
    with ProcessPoolExecutor(a.workers) as pool:
        for i, _ in enumerate(pool.map(run_case, jobs), 1):
            if i % 5 == 0:
                print(i, flush=True)
