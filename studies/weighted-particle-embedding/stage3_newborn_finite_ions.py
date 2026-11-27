"""Bound newborn arrivals under finite ions and evolving host charge."""

import argparse
import json
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter, time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from stage3_birth_growth import D0
from stage3_charging_grid import MOBILITY
from stage3_continuous import charged
from stage3_explicit_formation import CLUSTER, MASS_FACTOR
from stage3_ion_clocks import (
    ION_MASS,
    ION_VOLUME,
    PRESSURE,
    RECOMB,
    T,
    mobility_attachment,
)
from stage3_newborn_charge import FIELDS, uptake

REGIMES = {
    "neutral": (0,),
    "opposite": (-16,),
    "same": (16,),
    "bipolar": (-16, 16),
}
LEVELS = (
    (0.25, 80, 8, 24, 1e-7),
    (0.125, 128, 12, 32, 2e-9),
    (0.0625, 192, 18, 36, 5e-10),
)


def attachment(radius, charge, sign):
    """Evaluate the native small-ion mobility rate at the actual dry density."""
    i = int(sign < 0)
    return mobility_attachment(
        radius,
        MASS_FACTOR * radius**3,
        charge,
        T,
        PRESSURE,
        ION_MASS[i],
        MOBILITY[i],
        sign,
    )


class Host:
    """Track every host charge and finite ion reservoir per initial host."""

    def __init__(self, field, density, mode, ratio, qmax, tol, end):
        """Integrate charge changes at the prescribed growing host radius."""
        self.q = np.arange(-qmax, qmax + 1)
        size = len(self.q)
        initial = np.zeros(size + 6)
        for z in REGIMES[mode]:
            initial[z + qmax] += 1 / len(REGIMES[mode])
        initial[size : size + 2] = ratio
        self.initial = initial[:size].copy()

        def rhs(t, y):
            prob = np.maximum(y[:size], 0)
            pos, neg = np.maximum(y[size : size + 2], 0)
            r = float(field(t)[2]) * 1e-9
            plus = density * pos * attachment(r, self.q, 1) * prob
            minus = density * neg * attachment(r, self.q, -1) * prob
            blocked = plus[-1] + minus[0]
            plus[-1] = minus[0] = 0
            d = -(plus + minus)
            d[1:] += plus[:-1]
            d[:-1] += minus[1:]
            ap, an = plus.sum(), minus.sum()
            recombined = RECOMB * density * pos * neg
            return np.r_[
                d,
                -recombined - ap,
                -recombined - an,
                ap,
                an,
                recombined,
                blocked,
            ]

        sol = solve_ivp(
            rhs,
            (0, end),
            initial,
            method="DOP853",
            rtol=tol,
            atol=tol * 1e-3,
            dense_output=True,
        )
        if not sol.success:
            raise ArithmeticError(sol.message)
        self.solution = sol.sol
        pop = sol.y[:size]
        fp, fn, ap, an, rec, blocked = sol.y[size:]
        self.error = max(
            abs(pop.sum(axis=0) - 1).max(),
            abs(fp + ap + rec - ratio).max(),
            abs(fn + an + rec - ratio).max(),
            abs((self.q @ pop + fp - fn) - self.q @ self.initial).max(),
        )
        self.truncation = blocked[-1]
        if (
            pop.min() < -2e-7
            or min(fp.min(), fn.min()) < -2e-7
            or self.error > 5e-7
        ):
            raise ArithmeticError(
                "Host charge, population or ion inventory failed"
            )
        self.output = dict(
            host_final_mean_charge=float(self.q @ pop[:, -1]),
            host_final_neutral_fraction=pop[qmax, -1],
            free_positive_per_host=fp[-1],
            free_negative_per_host=fn[-1],
            attached_positive_per_host=ap[-1],
            attached_negative_per_host=an[-1],
            recombined_pairs_per_host=rec[-1],
            host_ion_ledger_max_per_host=self.error,
            blocked_host_attachments_per_host=self.truncation,
            max_host_ion_mass_fraction=(
                ap[-1] * ION_MASS[0] + an[-1] * ION_MASS[1]
            )
            / (MASS_FACTOR * (field(0)[2] * 1e-9) ** 3),
            max_host_mean_ion_volume_fraction=(ap[-1] + an[-1])
            * ION_VOLUME
            / (4 * np.pi / 3 * (field(0)[2] * 1e-9) ** 3),
        )

    def at(self, times):
        """Return normalized host fractions and the two free-ion inventories."""
        y = self.solution(times)
        p = np.maximum(y[: len(self.q)].T, 0)
        p /= p.sum(axis=1)[:, None]
        return p, np.maximum(y[len(self.q) : len(self.q) + 2].T, 0)


def kernel_table(radii, hosts, q, tracer_charge=1, extra_mass=0.0):
    """Use the native pair matrix to tabulate only tracer–host coefficients."""
    a = np.arange(len(radii))[:, None]
    b = len(radii) + np.arange(len(q))[None, :]
    z = np.r_[np.full(len(radii), tracer_charge), q]
    values = []
    for host in hosts:
        rr = np.r_[radii, np.full(len(q), host)]
        mass = MASS_FACTOR * rr**3
        mass[: len(radii)] += extra_mass
        k = charged(rr, mass, z, a, b)
        if not np.isfinite(k).all() or (k < 0).any():
            raise ArithmeticError("Invalid native tracer capture rate")
        values.append(k)
    return np.array(values)


def compete(survival, capture, ion, dt):
    """Advance first-event survival and probability of an ion arriving first."""
    total = capture + ion
    event = survival * -np.expm1(-dt * total)
    share = np.divide(ion, total, out=np.zeros_like(total), where=total > 0)
    return survival - event, event * share


def trial(ambient, mode, ratio, level):
    """Bracket arrivals under unrestricted behavior following a first ion."""
    tic = perf_counter()
    dt, nr, nh, qmax, tol = LEVELS[level]
    hist = ambient["history"]
    times = np.array([h["t"] for h in hist])
    vals = np.array([[h[k] for k in FIELDS] for h in hist])
    field = PchipInterpolator(times, vals, axis=0)
    bg = ambient["background"]
    if not np.allclose(vals[:, 3], bg, rtol=1e-12):
        raise ValueError("Finite-ion experiment requires conserved host number")
    host = Host(field, bg, mode, ratio, qmax, tol, times[-1])
    n = round(times[-1] / dt)
    micro = dt / 2
    edges = np.arange(2 * n + 1) * micro
    centers = (edges[1:] + edges[:-1]) / 2
    env = field(centers)
    hprob, free = host.at(centers)
    births = field(edges)[:, 4]
    monotone = np.maximum.accumulate(np.minimum(births, births[-1]))
    if np.max(abs(monotone - births)) > 8 * np.spacing(births[-1]):
        raise ArithmeticError("Prescribed births decreased")
    weights = np.diff(monotone[::2])
    rad = np.geomspace(D0 / 2 * (1 - 1e-8), 150e-9, nr)
    lograd = np.log(rad)
    hosts = (
        np.linspace(
            vals[:, 2].min() * (1 - 1e-10), vals[:, 2].max() * (1 + 1e-10), nh
        )
        * 1e-9
    )
    table = kernel_table(rad, hosts, host.q)
    att = np.column_stack([attachment(rad, np.ones(nr), s) for s in (1, -1)])
    mass = np.full(n, CLUSTER)
    survive = np.ones(n)
    no_ion = np.ones(n)
    first_ion = np.zeros(n)
    diameter = np.array([5.0, 10.0])
    cut = MASS_FACTOR * (diameter * 0.5e-9) ** 3
    lower, fixed, ambiguous = np.zeros((3, 2))
    for tick, v in enumerate(env):
        born = (tick + 1) // 2
        if not born:
            continue
        sl = slice(0, born)
        current = mass[sl]
        mid = current + micro / 2 * uptake(current, v[:2])
        delta = micro * uptake(mid, v[:2])
        rr = np.log(np.cbrt(mid / MASS_FACTOR))
        if rr.min() < lograd[0] or rr.max() > lograd[-1]:
            raise ArithmeticError("Tracer radius outside rate table")
        rh = v[2] * 1e-9
        j = np.searchsorted(hosts, rh).clip(1, nh - 1)
        frac = (rh - hosts[j - 1]) / (hosts[j] - hosts[j - 1])
        rates = (1 - frac) * (table[j - 1] @ hprob[tick]) + frac * (
            table[j] @ hprob[tick]
        )
        capture = bg * np.interp(rr, lograd, rates)
        ion = bg * np.interp(rr, lograd, att @ free[tick])
        for k, threshold in enumerate(cut):
            hit = (current < threshold) & (current + delta >= threshold)
            if not hit.any():
                continue
            part = micro * (threshold - current[hit]) / delta[hit]
            end, added = compete(survive[sl][hit], capture[hit], ion[hit], part)
            w = weights[sl][hit]
            lower[k] += w @ end
            ambiguous[k] += w @ (first_ion[sl][hit] + added)
            fixed[k] += w @ (no_ion[sl][hit] * np.exp(-capture[hit] * part))
        survive[sl], added = compete(survive[sl], capture, ion, micro)
        first_ion[sl] += added
        no_ion[sl] *= np.exp(-capture * micro)
        current += delta
    for k, threshold in enumerate(cut):
        unhit = mass < threshold
        ambiguous[k] += weights[unhit] @ first_ion[unhit]
    scale = 1000 / weights.sum()
    roundoff = 3e-12 * weights.sum()
    if np.any(lower > fixed + roundoff) or np.any(
        fixed > lower + ambiguous + roundoff
    ):
        raise ArithmeticError(
            "Fixed tracer falls outside its first-event bounds"
        )
    return [
        dict(
            background_m3=bg,
            initial_host_charge=mode,
            initial_ions_each_sign_per_host=ratio,
            diameter_nm=d,
            level=level,
            trace_interval_s=dt,
            tracer_radius_points=nr,
            host_radius_points=nh,
            host_charge_extent=qmax,
            host_rtol=tol,
            per_1000_lower=scale * lower[k],
            per_1000_upper=scale * (lower[k] + ambiguous[k]),
            per_1000_first_ion_uncertainty=scale * ambiguous[k],
            per_1000_fixed_tracer_live_hosts=scale * fixed[k],
            births_m3=weights.sum(),
            wall_s=perf_counter() - tic,
            **host.output,
        )
        for k, d in enumerate(diameter)
    ]


def run(cache, until, level):
    """Run each registered exposure once at the requested numerical setting."""
    dest = cache / "newborn-finite-ions"
    dest.mkdir(parents=True, exist_ok=True)
    for bg, ratio, mode in product(
        (1e10, 1e9, 1e11),
        (0, 1, 0.1, 4),
        ("opposite", "same", "bipolar", "neutral"),
    ):
        if time() >= until:
            return
        source_dt = 0.25 if bg == 1e9 else 0.0625
        src = (
            cache
            / "formation-cohorts"
            / (
                f"cohorts-1e-09-2e-09-{bg:g}-60-1e-07-"
                f"activation-scavenging-{source_dt:g}-128.json"
            )
        )
        target = dest / f"ion-{bg:g}-{ratio}-{mode}-{level}.json"
        if target.exists() or not src.exists():
            continue
        rows = trial(json.loads(src.read_text()), mode, ratio, level)
        tmp = target.with_suffix(".part")
        tmp.write_text(json.dumps(rows) + "\n")
        tmp.replace(target)
        print(target.name, rows[-1]["wall_s"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument(
        "--level", type=int, required=True, choices=range(len(LEVELS))
    )
    p.add_argument(
        "--until",
        required=True,
        type=lambda x: datetime.fromisoformat(x).timestamp(),
    )
    args = p.parse_args()
    run(args.cache, args.until, args.level)
