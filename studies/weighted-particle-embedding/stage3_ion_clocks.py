"""Finite-box charge exchange and aerosol collisions with exact event laws."""

import argparse
import json
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from particula.dynamics import (
    get_coulomb_kernel_gopalakrishnan2012_via_system_state,
)
from particula.dynamics.ions.coagulation import (
    NEGATIVE_ION_MASS,
    POSITIVE_ION_MASS,
)
from particula.dynamics.ions.mobility_attachment import mobility_attachment
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply

ION_MASS = np.array([POSITIVE_ION_MASS, NEGATIVE_ION_MASS])
ION_VOLUME = 4 * np.pi / 3 * (0.25e-9) ** 3
T = 293.0
PRESSURE = 101325.0
DRY_RHO = 1400.0
RECOMB = 1.7e-12
SIZES = {"newborn": (2.5, 60.0), "mixed": (20.0, 95.0), "similar": (65.0, 80.0)}
CHARGES = {"neutral": (0, 0), "like": (2, 2), "opposite": (-2, 2)}


class IonBox:
    """Enumerate core identities and finite positive and negative ions."""

    def __init__(self, shape, pattern, positive, negative):
        """Set the dry cores, signed charges and finite ion supplies."""
        self.radius = np.array(SIZES[shape]) * 1e-9
        self.volume = 4 * np.pi / 3 * self.radius**3
        self.base_q = np.array(CHARGES[pattern])
        self.ions = np.array([positive, negative])
        self.first = (((1, 0, 0), (2, 0, 0)), positive, negative)
        # Cache each physical rate for this finite box only.
        self.geometry = lru_cache(None)(self._geometry)
        self.attachment = lru_cache(None)(self._attachment)
        self.coag = lru_cache(None)(self._coag)

    def _geometry(self, host):
        core, plus, minus = host
        idx = np.array([bool(core & 1), bool(core & 2)])
        volume = self.volume[idx].sum()
        mass = volume * DRY_RHO + plus * ION_MASS[0] + minus * ION_MASS[1]
        radius = np.cbrt(
            (volume + (plus + minus) * ION_VOLUME) * 3 / (4 * np.pi)
        )
        return radius, mass, self.base_q[idx].sum() + plus - minus

    def _attachment(self, host, sign):
        r, m, z = self.geometry(host)
        lane = int(sign < 0)
        return float(
            mobility_attachment(
                r,
                m,
                z,
                T,
                PRESSURE,
                ION_MASS[lane],
                (1.4e-4, 1.9e-4)[lane],
                sign,
            )
        )

    def _coag(self, hosts):
        r, m, z = np.array([self.geometry(host) for host in hosts]).T
        return float(
            get_coulomb_kernel_gopalakrishnan2012_via_system_state(
                r, m, z, T, PRESSURE
            )[0, 1]
        )

    def edges(self, state):
        """Emit rate coefficients, multiplicities and successor states."""
        hosts, pos, neg = state
        if len(hosts) == 2:
            merged = tuple(map(sum, zip(*hosts, strict=True)))
            yield 1, self.coag(hosts), ((merged,), pos, neg)
        if pos and neg:
            yield 2, pos * neg * RECOMB, (hosts, pos - 1, neg - 1)
        for i, host in enumerate(hosts):
            core, p, n = host
            for sign, available in [(1, pos), (-1, neg)]:
                if available:
                    dest = (core, p + int(sign == 1), n + int(sign == -1))
                    yield (
                        0,
                        available * self.attachment(host, sign),
                        (
                            hosts[:i] + (dest,) + hosts[i + 1 :],
                            pos - int(sign == 1),
                            neg - int(sign == -1),
                        ),
                    )

    def matrices(self):
        """Build column-oriented generators per unit inverse box volume."""
        idx = {self.first: 0}
        queue = deque([self.first])
        triples = [[], [], []]
        while queue:
            state = queue.popleft()
            src = idx[state]
            for process, rate, dest in self.edges(state):
                if dest not in idx:
                    idx[dest] = len(idx)
                    queue.append(dest)
                triples[process].extend(
                    ((idx[dest], src, rate), (src, src, -rate))
                )
        states = list(idx)
        operators = []
        for values in triples:
            if values:
                r, c, v = zip(*values, strict=True)
            else:
                r, c, v = [], [], []
            operators.append(
                coo_matrix((v, (r, c)), shape=(len(idx), len(idx))).tocsc()
            )
        return states, *operators

    def observable(self, state):
        """Count aerosols and ion partitions after verifying the ledgers."""
        hosts, p, n = state
        attached = np.array([h[1:] for h in hosts]).sum(0)
        inert = self.ions - np.array([p, n]) - attached
        charge = np.array([self.geometry(h)[2] for h in hosts])
        if (
            inert[0] != inert[1]
            or min(inert) < 0
            or charge.sum() + p - n
            != self.base_q.sum() + self.ions[0] - self.ions[1]
        ):
            raise ArithmeticError("Finite-box ion or charge ledger failed")
        return [
            len(hosts),
            p,
            n,
            *attached,
            inert[0],
            abs(charge).sum(),
            (charge != 0).sum(),
            int(len(hosts) == 1),
        ]


NAMES = [
    "aerosol_count",
    "free_positive",
    "free_negative",
    "attached_positive",
    "attached_negative",
    "recombined_pairs",
    "aerosol_unsigned_charge",
    "charged_aerosols",
    "prob_merged",
]


def solve(shape, pattern, pos, neg, density, times, substeps):
    """Compare fixed charge, collision omission and split clocks."""
    box = IonBox(shape, pattern, pos, neg)
    states, attach, collide, recombine = box.matrices()
    vol = 2 / density
    attach, collide, recombine = (
        (attach + recombine) / vol,
        collide / vol,
        recombine / vol,
    )
    outputs = np.array([box.observable(s) for s in states])
    p0 = np.zeros(len(states))
    p0[0] = 1
    rows = []
    for t in times:
        for mode, steps in [
            ("coupled", 1),
            ("fixed-charge", 1),
            ("no-aerosol-collision", 1),
            *(("strang", i) for i in substeps),
        ]:
            tic = perf_counter()
            if mode == "strang":
                a = attach * (t / (2 * steps))
                c = collide * (t / steps)
                p = expm_multiply(a, p0)
                for i in range(steps):
                    p = expm_multiply(c, p)
                    p = expm_multiply(a if i == steps - 1 else 2 * a, p)
            else:
                gen = {
                    "coupled": attach + collide,
                    "fixed-charge": collide + recombine,
                    "no-aerosol-collision": attach,
                }[mode]
                p = expm_multiply(t * gen, p0)
            if abs(p.sum() - 1) > 1e-10 or p.min() < -1e-12:
                raise ArithmeticError("Probability failed to close")
            rows.append(
                dict(
                    shape=shape,
                    pattern=pattern,
                    positive=pos,
                    negative=neg,
                    density=density,
                    horizon_s=t,
                    policy=mode,
                    steps=steps,
                    states=len(p),
                    seconds=perf_counter() - tic,
                    **dict(zip(NAMES, p @ outputs, strict=True)),
                )
            )
    return rows


def job(item):
    """Archive all clock comparisons for one box."""
    folder, shape, pattern, pos, neg, d, times, sub = item
    p = folder / f"ion-{shape}-{pattern}-{pos}-{neg}-{d:g}.json"
    if p.exists():
        return
    rows = solve(shape, pattern, pos, neg, d, times, sub)
    tmp = p.with_suffix(".part")
    tmp.write_text(json.dumps(rows) + "\n")
    tmp.replace(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--shapes", nargs="+", default=list(SIZES))
    parser.add_argument("--patterns", nargs="+", default=list(CHARGES))
    parser.add_argument(
        "--ions", nargs="+", default=["0,0", "1,1", "2,2", "2,0", "0,2"]
    )
    parser.add_argument(
        "--density", type=float, nargs="+", default=[1e9, 1e10, 1e11]
    )
    parser.add_argument(
        "--times", type=float, nargs="+", default=[1.0, 10.0, 60.0, 300.0]
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--workers", type=int, default=2)
    a = parser.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    jobs = (
        (a.folder, shape, pat, *map(int, ions.split(",")), d, a.times, a.steps)
        for shape, pat, ions, d in product(
            a.shapes, a.patterns, a.ions, a.density
        )
    )
    with ProcessPoolExecutor(a.workers) as pool:
        for i, _ in enumerate(pool.map(job, jobs), 1):
            if i % 5 == 0:
                print(i, flush=True)
