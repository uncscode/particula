"""Track individual births, vapor uptake and Brownian mergers."""

import argparse
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter, time

import numpy as np
from particula.gas import get_molecule_mean_free_path
from scipy.constants import Avogadro
from stage2_population import brownian
from stage3_birth_growth import D0, MOLAR, MOLECULES_PER_CLUSTER, RHO

MASS_FACTOR = (4 * np.pi / 3) * RHO
CLUSTER = MOLECULES_PER_CLUSTER * MOLAR / Avogadro
VAPORS = np.array([1e-9, 2e-9])
DIFF = np.array([1e-5, 0.65e-5])
ALPHA = np.array([1.0, 0.55])
PATH = np.asarray(
    get_molecule_mean_free_path(
        molar_mass=np.array([MOLAR, 0.18]), temperature=293.0, pressure=101325.0
    )
)


class Source:
    """Consume a shared unit-rate Poisson event stream in integrated hazard."""

    def __init__(self, seed):
        """Initialize the independent source clock."""
        self.rng = np.random.default_rng(seed)
        self.exposure = 0.0
        self.last = 0.0
        self.events = np.empty(0)
        self.offset = 0

    def count(self, demand):
        """Advance continuous source exposure and count crossed event times."""
        self.exposure += demand
        n = 0
        while True:
            if self.offset == len(self.events):
                self.events = (
                    self.last + self.rng.exponential(size=2048).cumsum()
                )
                self.last = self.events[-1]
                self.offset = 0
            stop = np.searchsorted(self.events, self.exposure, side="right")
            n += stop - self.offset
            self.offset = stop
            if stop < len(self.events):
                return n


class Box:
    """Use one row per particle with a dry host marker and two condensables."""

    def __init__(self, hosts, seed):
        """Place dry hosts and two vapors in a finite volume."""
        self.volume = hosts / 1e10
        self.hosts = hosts
        self.mass = np.zeros((hosts, 3))
        self.mass[:, 0] = MASS_FACTOR * (60e-9) ** 3
        self.dry = self.mass[:, 0].sum()
        self.gas = VAPORS.copy()
        events, sources = np.random.SeedSequence(seed).spawn(2)
        self.rng = np.random.default_rng(events)
        self.source = Source(sources)
        self.births = self.mergers = self.captures = 0
        self.max_p = 0.0

    def transfer(self, mass, dt):
        """Allocate the frozen-geometry gas debit among particles."""
        radius = np.cbrt(mass / MASS_FACTOR)[:, None]
        kn = PATH / radius
        k = (
            4
            * np.pi
            * radius
            * DIFF
            * 0.75
            * ALPHA
            * (1 + kn)
            / (kn * kn + (1 + 0.283 * ALPHA) * kn + 0.75 * ALPHA)
        )
        total = k.sum(axis=0)
        take = self.gas * -np.expm1(-dt * total / self.volume)
        return k * (self.volume * take / total), take

    def grow(self, dt):
        """Update geometry at a half step, then commit each finite gas flux."""
        total = self.mass.sum(axis=1)
        predictor, _ = self.transfer(total, dt / 2)
        flux, take = self.transfer(total + predictor.sum(axis=1), dt)
        self.mass[:, 1:] += flux
        self.gas -= take

    def inject(self, dt):
        """Create integer clusters from the live activation-rate reservoir."""
        prop = self.gas[0] * Avogadro / MOLAR * 1e-7 * self.volume
        n = self.source.count(prop * dt)
        if n:
            debit = n * CLUSTER / self.volume
            if debit > self.gas[0]:
                raise ArithmeticError("Poisson proposal exhausts the precursor")
            particles = np.zeros((n, 3))
            particles[:, 1] = CLUSTER
            self.mass = np.concatenate((self.mass, particles))
            self.gas[0] -= debit
            self.births += n

    def collide(self, dt, policy):
        """Apply Brownian rates to inclusion-weighted disjoint pairs."""
        n = len(self.mass)
        if policy == "off" or n < 2:
            return
        pairs = self.rng.permutation(n)[: 2 * (n // 2)].reshape(-1, 2)
        i, j = pairs.T
        wet = self.mass.sum(axis=1)
        rr = np.cbrt(wet / MASS_FACTOR)
        p = brownian(rr, wet, i, j) * (
            (n * (n - 1) / 2) / len(i) * dt / self.volume
        )
        cross = (self.mass[i, 0] == 0) != (self.mass[j, 0] == 0)
        if policy == "scavenging":
            p[~cross] = 0
        peak = p.max(initial=0.0)
        if not np.isfinite(peak) or peak > 1 or (p < 0).any():
            raise ArithmeticError(f"Invalid pair probability {peak}")
        self.max_p = max(self.max_p, peak)
        keep = self.rng.random(len(i)) < p
        self.captures += int(cross[keep].sum())
        i, j = i[keep], j[keep]
        if len(i):
            self.mass[i] += self.mass[j]
            self.mass = np.delete(self.mass, j, axis=0)
        self.mergers += len(i)

    def observe(self, t):
        """Measure inventories, particle lineage and detection diameter."""
        if len(self.mass) != self.hosts + self.births - self.mergers:
            raise ArithmeticError("Number balance failed")
        inventory = self.mass[:, 1:].sum(axis=0) / self.volume
        gap = max(
            abs(self.mass[:, 0].sum() / self.dry - 1),
            max(abs((inventory + self.gas) / VAPORS - 1)),
        )
        if not np.isfinite(gap) or gap > 1e-10 or np.min(self.gas) < 0:
            raise ArithmeticError(f"Material balance failed {gap}")
        newborn = self.mass[:, 0] == 0
        sizes = 2e9 * np.cbrt(self.mass[newborn].sum(axis=1) / MASS_FACTOR)
        return dict(
            time_s=t,
            number_m3=len(self.mass) / self.volume,
            newborn_only_m3=newborn.sum() / self.volume,
            above_5nm_new_m3=(sizes >= 5).sum() / self.volume,
            above_10nm_new_m3=(sizes >= 10).sum() / self.volume,
            background_hosts_m3=(~newborn).sum() / self.volume,
            births_m3=self.births / self.volume,
            capture_events_m3=self.captures / self.volume,
            precursor_gas_kg_m3=self.gas[0],
            organic_gas_kg_m3=self.gas[1],
            precursor_particle_kg_m3=inventory[0],
            organic_particle_kg_m3=inventory[1],
            max_ledger=gap,
        )


def run(hosts, seed, policy, interval, growth, collision_ratio, end):
    """Center source and collision on independent regular time meshes."""
    start = perf_counter()
    system = Box(hosts, seed)
    dt = interval / (2 * collision_ratio)
    ticks = round(end / dt)
    if abs(ticks * dt - end) > 1e-12 or end % 2:
        raise ValueError("Observation horizon must be aligned to two seconds")
    history = [system.observe(0)]
    for k in range(ticks):
        for _ in range(growth):
            system.grow(dt / growth)
        # Birth and collision occur at the midpoint of their own intervals.
        if (k + 1) % (2 * collision_ratio) == collision_ratio:
            system.inject(interval)
        if (k + 1) % 2 == 1:
            system.collide(2 * dt, policy)
        if (k + 1) % round(2 / dt) == 0:
            history.append(system.observe((k + 1) * dt))
    return dict(
        hosts=hosts,
        seed=seed,
        collision=policy,
        birth_interval=interval,
        collision_interval=2 * dt,
        growth_interval=dt / growth,
        end=end,
        onset_diameter_nm=1e9 * D0,
        max_probability=system.max_p,
        runtime_s=perf_counter() - start,
        history=history,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--hosts", nargs="+", type=int, default=[64, 256, 1024])
    p.add_argument(
        "--interval", nargs="+", type=float, default=[1.0, 0.5, 0.25]
    )
    p.add_argument("--collision-ratio", nargs="+", type=int, default=[1, 2])
    p.add_argument("--growth", nargs="+", type=int, default=[1])
    p.add_argument(
        "--policies", nargs="+", default=["physical", "scavenging", "off"]
    )
    p.add_argument("--seeds", type=int, default=256)
    p.add_argument("--first", type=int, default=443001)
    p.add_argument("--end", type=int, default=128)
    p.add_argument(
        "--until",
        type=lambda v: datetime.fromisoformat(v).timestamp(),
        required=True,
    )
    a = p.parse_args()
    a.folder.mkdir(exist_ok=True, parents=True)
    for seed in range(a.first, a.first + a.seeds):
        for cfg in product(
            a.hosts, a.interval, a.collision_ratio, a.growth, a.policies
        ):
            if time() >= a.until:
                sys.exit(0)
            hosts, interval, ratio, growth, method = cfg
            path = a.folder / (
                "-".join(map(str, (*cfg, a.end, seed))) + ".json"
            )
            if path.exists():
                continue
            row = run(hosts, seed, method, interval, growth, ratio, a.end)
            tmp = path.with_suffix(".part")
            tmp.write_text(json.dumps(row) + "\n")
            tmp.replace(path)
            print(path.name, row["runtime_s"], flush=True)
