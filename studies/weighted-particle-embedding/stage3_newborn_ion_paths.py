"""Follow dilute newborn tracers through repeated finite-ion attachment."""

import argparse
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter, time

import numpy as np
from scipy.interpolate import PchipInterpolator
from stage3_birth_growth import D0, RHO
from stage3_charging_grid import MOBILITY
from stage3_explicit_formation import CLUSTER, MASS_FACTOR
from stage3_ion_clocks import (
    ION_MASS,
    ION_VOLUME,
    PRESSURE,
    T,
    mobility_attachment,
)
from stage3_newborn_charge import FIELDS, uptake
from stage3_newborn_finite_ions import Host, kernel_table

CLOCKS = (
    (0.5, 96, 9),
    (0.25, 144, 14),
    (0.125, 192, 18),
    (0.0625, 256, 24),
    (0.03125, 320, 30),
)
CUT = MASS_FACTOR * (np.array([5.0, 10.0]) * 0.5e-9) ** 3
INITIAL_SEED = 551001


class Medium:
    """Bind the ion reservoirs, host charge and native tracer coefficients."""

    def __init__(self, ambient, mode, ratio, level):
        """Integrate the prescribed host state and tabulate time midpoints."""
        h = ambient["history"]
        t = np.array([r["t"] for r in h])
        vals = np.array([[r[k] for k in FIELDS] for r in h])
        self.field = PchipInterpolator(t, vals, axis=0)
        self.bg = ambient["background"]
        self.dt, nr, nh = CLOCKS[level]
        self.end = t[-1]
        host = Host(self.field, self.bg, mode, ratio, 32, 2e-9, self.end)
        self.q = host.q
        self.edges = np.linspace(0, self.end, round(self.end / self.dt) + 1)
        mid = (self.edges[:-1] + self.edges[1:]) / 2
        self.environment = self.field(mid)
        self.prob, self.ions = host.at(mid)
        self.rad = np.geomspace(D0 / 2 * (1 - 1e-8), 150e-9, nr)
        self.lograd = np.log(self.rad)
        self.hosts = (
            np.linspace(
                vals[:, 2].min() * (1 - 1e-10),
                vals[:, 2].max() * (1 + 1e-10),
                nh,
            )
            * 1e-9
        )
        self.host_idx = np.searchsorted(
            self.hosts, self.environment[:, 2] * 1e-9
        ).clip(1, nh - 1)
        j = self.host_idx
        self.host_frac = (self.environment[:, 2] * 1e-9 - self.hosts[j - 1]) / (
            self.hosts[j] - self.hosts[j - 1]
        )
        self.cache = {}
        times = np.linspace(0, self.end, 16385)
        births = self.field(times)[:, 4]
        if abs(births - np.maximum.accumulate(births)).max() > 8 * np.spacing(
            vals[-1, 4]
        ):
            raise ArithmeticError("Cumulative births decreased")
        u, ix = np.unique(
            np.maximum.accumulate(births) / vals[-1, 4], return_index=True
        )
        self.inverse = u, times[ix]
        self.host_drift = host.error

    def tables(self, pos, neg):
        """Retain the actual ion masses and volumes for each charge history."""
        key = (pos, neg)
        if key not in self.cache:
            q = 1 + pos - neg
            extra = (
                pos * ION_MASS[0]
                + neg * ION_MASS[1]
                - (pos + neg) * RHO * ION_VOLUME
            )
            tab = kernel_table(self.rad, self.hosts, self.q, q, extra)
            capture = np.empty((len(self.prob), len(self.rad)))
            for j in np.unique(self.host_idx):
                indices = np.flatnonzero(self.host_idx == j)
                f = self.host_frac[indices, None]
                capture[indices] = (
                    (1 - f) * (self.prob[indices] @ tab[j - 1].T)
                    + f * (self.prob[indices] @ tab[j].T)
                ) * self.bg
            mass = MASS_FACTOR * self.rad**3 + extra
            ions = [
                mobility_attachment(
                    self.rad,
                    mass,
                    np.full(len(mass), q),
                    T,
                    PRESSURE,
                    ION_MASS[i],
                    MOBILITY[i],
                    sign,
                )
                for i, sign in enumerate((1, -1))
            ]
            self.cache[key] = capture, np.array(ions)
        return self.cache[key]

    def rates(self, tick, mass, pos, neg):
        """Interpolate capture and attachment rates in each live state."""
        r = np.log(np.cbrt(mass / MASS_FACTOR))
        if r.min() < self.lograd[0] or r.max() > self.lograd[-1]:
            raise ArithmeticError("Tracer radius outside the native rate table")
        rates = np.empty((len(mass), 3))
        code = pos * 128 + neg
        for k in np.unique(code):
            select = code == k
            cap, att = self.tables(int(k // 128), int(k % 128))
            rays = r[select]
            rates[select, 0] = np.interp(rays, self.lograd, cap[tick])
            for i in (0, 1):
                rates[select, i + 1] = (
                    np.interp(rays, self.lograd, att[i])
                    * self.ions[tick, i]
                    * self.bg
                )
        return rates


def simulate(model, seed, size):
    """Sample birth times and exponential hazards, integrating each event."""
    start = perf_counter()
    rng = np.random.default_rng(seed)
    u = (np.arange(size) + rng.random(size)) / size
    birth = np.interp(u, *model.inverse)
    exp = rng.exponential(size=(64, size))
    mark = rng.random((64, size))
    mass = np.full(size, CLUSTER)
    vapor = np.zeros(size)
    pos = np.zeros(size, dtype=int)
    neg = pos.copy()
    hazard = np.zeros(size)
    remain = exp[0].copy()
    arrivals = np.zeros((size, 2))
    arriving_pos = np.zeros_like(arrivals)
    arriving_neg = np.zeros_like(arrivals)
    markers = np.zeros((size, 2), dtype=bool)
    hmax = 0.0
    for tick, (lo, hi) in enumerate(
        zip(model.edges[:-1], model.edges[1:], strict=True)
    ):
        n = np.searchsorted(birth, hi, side="left")
        idx = np.arange(n)
        left = hi - np.maximum(lo, birth[:n])
        gas = model.environment[tick, :2]
        while len(idx):
            current = mass[idx]
            mid = current + left * 0.5 * uptake(current, gas)
            rates = model.rates(tick, mid, pos[idx], neg[idx])
            attachment = rates[:, 1:].sum(axis=1)
            spent = left * attachment
            hit = spent >= remain[idx]
            hmax = max(hmax, float(spent.max()))
            fraction = np.ones_like(left)
            fraction[hit] = remain[idx[hit]] / spent[hit]
            dt = left * fraction
            growth = dt * uptake(current + dt * 0.5 * uptake(current, gas), gas)
            cap = dt * rates[:, 0]
            before = hazard[idx]
            for j, threshold in enumerate(CUT):
                crossed = (current < threshold) & (
                    current + growth >= threshold
                )
                if crossed.any():
                    ids = idx[crossed]
                    phase = (threshold - current[crossed]) / growth[crossed]
                    value = np.exp(-before[crossed] - cap[crossed] * phase)
                    arrivals[ids, j] = value
                    arriving_pos[ids, j] = value * pos[ids]
                    arriving_neg[ids, j] = value * neg[ids]
                    markers[ids, j] = True
            mass[idx] += growth
            vapor[idx] += growth
            hazard[idx] += cap
            remain[idx] -= np.minimum(spent, remain[idx])
            ids = idx[hit]
            if len(ids):
                events = pos[ids] + neg[ids]
                if np.max(events) + 1 >= len(exp):
                    raise ArithmeticError(
                        "Exceeded the recorded ion event stream"
                    )
                choose = mark[events, ids] < rates[hit, 1] / attachment[hit]
                pos[ids] += choose
                neg[ids] += ~choose
                mass[ids] += RHO * ION_VOLUME
                remain[ids] = exp[events + 1, ids]
                for j, threshold in enumerate(CUT):
                    jump = (mass[ids] >= threshold) & ~markers[ids, j]
                    ii = ids[jump]
                    value = np.exp(-hazard[ii])
                    arrivals[ii, j] = value
                    arriving_pos[ii, j] = value * pos[ii]
                    arriving_neg[ii, j] = value * neg[ii]
                    markers[ii, j] = True
            left = (left - dt)[hit]
            idx = ids
    ion_vol = RHO * ION_VOLUME * (pos + neg)
    actual = mass - ion_vol + pos * ION_MASS[0] + neg * ION_MASS[1]
    ledger = np.max(
        abs(actual - (CLUSTER + vapor + pos * ION_MASS[0] + neg * ION_MASS[1]))
        / actual
    )
    if ledger > 1e-10 or not np.array_equal(mass[:, None] >= CUT, markers):
        raise ArithmeticError("Tracer material or crossing ledger failed")
    return [
        dict(
            diameter_nm=d,
            arrivals_per_1000=1000 * arrivals[:, j].mean(),
            fraction_with_any_ion=float(np.mean(pos + neg > 0)),
            fraction_with_two_or_more_ions=float(np.mean(pos + neg >= 2)),
            positive_ions_per_arrival=arriving_pos[:, j].sum()
            / arrivals[:, j].sum()
            if arrivals[:, j].any()
            else None,
            negative_ions_per_arrival=arriving_neg[:, j].sum()
            / arrivals[:, j].sum()
            if arrivals[:, j].any()
            else None,
            maximum_ions=int((pos + neg).max()),
            max_proposed_integrated_rate=hmax,
            max_tracer_relative_ledger=ledger,
            max_host_ledger_per_host=model.host_drift,
            wall_s=perf_counter() - start,
        )
        for j, d in enumerate((5, 10))
    ]


def run(root, until, level, batches, size):
    """Interleave fresh stratified birth batches across finite ion exposures."""
    dest = root / "newborn-ion-paths"
    dest.mkdir(parents=True, exist_ok=True)
    src = (
        root
        / "formation-cohorts"
        / (
            "cohorts-1e-09-2e-09-1e+10-60-1e-07-"
            "activation-scavenging-0.0625-128.json"
        )
    )
    raw = json.loads(src.read_text())

    @lru_cache(maxsize=12)
    def medium(mode, ratio):
        return Medium(raw, mode, ratio, level)

    cases = (
        ("opposite", 4),
        ("opposite", 1),
        ("bipolar", 4),
        ("same", 4),
        ("neutral", 4),
        ("opposite", 0),
        ("same", 0),
        ("bipolar", 1),
        ("neutral", 1),
    )
    for rep in range(batches):
        for mode, ratio in cases:
            if time() >= until:
                return
            seed = INITIAL_SEED + rep
            path = dest / f"paths-{level}-{size}-{mode}-{ratio}-{seed}.json"
            if path.exists():
                continue
            rows = simulate(medium(mode, ratio), seed, size)
            for r in rows:
                r.update(
                    level=level,
                    time_step_s=CLOCKS[level][0],
                    background_m3=1e10,
                    initial_host_charge=mode,
                    initial_ions_each_sign_per_host=ratio,
                    batch_seed=seed,
                    tracers_in_batch=size,
                )
            tmp = path.with_suffix(".part")
            tmp.write_text(json.dumps(rows) + "\n")
            tmp.replace(path)
            print(path.name, rows[-1]["wall_s"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument(
        "--level", type=int, choices=range(len(CLOCKS)), required=True
    )
    p.add_argument("--tracers", type=int, default=512)
    p.add_argument("--batches", type=int, default=128)
    p.add_argument(
        "--until",
        required=True,
        type=lambda x: datetime.fromisoformat(x).timestamp(),
    )
    a = p.parse_args()
    run(a.cache, a.until, a.level, a.batches, a.tracers)
