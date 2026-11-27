"""Follow dilute charged newborn tracers through 5 and 10 nm."""

import argparse
import json
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter, time

import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from stage2_population import brownian
from stage3_birth_growth import D0
from stage3_continuous import charged
from stage3_explicit_formation import ALPHA, CLUSTER, DIFF, MASS_FACTOR, PATH

PURE = (
    "Fuchs neutral",
    "Zero charge",
    "Same +4",
    "Opposite −4",
    "Same +16",
    "Opposite −16",
)
HOSTS = (0, 0, 4, -4, 16, -16)
MODES = (*PURE, "Mixed ±4", "Mixed ±16", "No capture")
FIELDS = (
    "precursor_gas_kg_m3",
    "organic_gas_kg_m3",
    "background_radius_nm",
    "background_hosts_m3",
    "births_m3",
)


def physical_rates(radius, host):
    """Return native Coulomb kernels and the reference Fuchs neutral kernel."""
    r = np.r_[radius, host]
    m = MASS_FACTOR * r**3
    i = np.arange(len(radius))
    j = np.full_like(i, len(radius))
    rates = [brownian(r, m, i, j)]
    for q in HOSTS[1:]:
        z = np.r_[np.ones(len(radius)), q]
        rates.append(charged(r, m, z, i, j))
    return np.array(rates).T


class Rates:
    """Interpolate the physical kernels in wet tracer and host radius."""

    def __init__(self, maximum_host, bins):
        """Tabulate native coefficients over the actual host radius range."""
        radii = np.geomspace(D0 / 2 * (1 - 1e-8), 150e-9, bins)
        hosts = np.geomspace(
            59.999e-9, max(maximum_host * 1.00001, 60.001e-9), max(6, bins // 8)
        )
        coefficients = np.array([physical_rates(radii, h) for h in hosts])
        if not (coefficients > 0).all():
            raise ArithmeticError("Nonpositive prescribed pair rate")
        self.table = RegularGridInterpolator(
            (np.log(hosts), np.log(radii)), np.log(coefficients)
        )
        self.hosts = hosts
        self.radii = radii

    def __call__(self, radius, host):
        """Average same- and opposite-sign rates for bipolar hosts."""
        coordinates = np.column_stack(
            (np.full(len(radius), np.log(host)), np.log(radius))
        )
        k = np.exp(self.table(coordinates))
        return np.column_stack(
            (
                k,
                (k[:, 2] + k[:, 3]) / 2,
                (k[:, 4] + k[:, 5]) / 2,
                np.zeros(len(k)),
            )
        )

    def audit(self):
        """Check withheld tracer and host sizes against native coefficients."""
        rr = np.geomspace(self.radii[0] * 1.001, self.radii[-1] * 0.999, 37)
        errors = []
        for rh in np.geomspace(self.hosts[0], self.hosts[-1], 8)[1:-1]:
            direct = physical_rates(rr, rh)
            errors.append(
                np.max(abs(self(rr, rh)[:, : len(PURE)] / direct - 1), axis=0)
            )
        return np.max(errors, axis=0)


def uptake(mass, gas):
    """Return irreversible two-species mass uptake per tracer in kg/s."""
    r = np.cbrt(mass / MASS_FACTOR)[:, None]
    kn = PATH / r
    k = (
        4
        * np.pi
        * r
        * DIFF
        * 0.75
        * ALPHA
        * (1 + kn)
        / (kn**2 + (1 + 0.283 * ALPHA) * kn + 0.75 * ALPHA)
    )
    return k @ gas


def crossings(a, md, hazard, change, weights, mcut, arrivals, eligible):
    """Credit a lineage at its first physical diameter crossing."""
    for j, c in enumerate(mcut):
        hit = (a < c) & (a + md >= c)
        if hit.any():
            fraction = (c - a[hit]) / md[hit]
            surv = np.exp(-hazard[hit] - change[hit] * fraction[:, None])
            eligible[j] += weights[hit].sum()
            arrivals[j] += weights[hit] @ surv


def simulate(ambient, dt, bins, stride):
    """Accumulate tracer births and first-passage survival."""
    start = perf_counter()
    h = ambient["history"][::stride]
    if h[-1]["t"] != ambient["horizon"]:
        h = [*h, ambient["history"][-1]]
    t = np.array([z["t"] for z in h])
    ambient_fields = np.array([[z[k] for k in FIELDS] for z in h])
    interp = PchipInterpolator(t, ambient_fields, axis=0)
    rates = Rates(ambient_fields[:, 2].max() * 1e-9, bins)
    n = round(t[-1] / dt)
    micro = dt / 2
    times = np.arange(2 * n + 1) * micro
    centers = interp(times[:-1] + micro / 2)
    endpoints = interp(times)
    source = endpoints[:, 4]
    # At an exhausted source, interpolation can oscillate by one float ulp.
    monotone = np.maximum.accumulate(np.minimum(source, h[-1]["births_m3"]))
    if np.max(abs(monotone - source)) > 8 * np.spacing(h[-1]["births_m3"]):
        raise ArithmeticError("Interpolated cumulative births decreased")
    endpoints[:, 4] = monotone
    weights = np.zeros(n)
    m = np.full(n, CLUSTER)
    hazard = np.zeros((n, len(MODES)))
    markers = np.array([5.0, 10.0])
    mcut = MASS_FACTOR * (markers * 0.5e-9) ** 3
    arrivals = np.zeros((2, len(MODES)))
    eligible = np.zeros(2)
    born = 0
    for tick, v in enumerate(centers):
        if tick % 2:
            # Midpoint births use the actual cumulative source across this step.
            weights[born] = endpoints[tick + 1, 4] - endpoints[tick - 1, 4]
            born += 1
        if not born:
            continue
        a = m[:born]
        mid = a + micro / 2 * uptake(a, v[:2])
        md = micro * uptake(mid, v[:2])
        change = micro * v[3] * rates(np.cbrt(mid / MASS_FACTOR), v[2] * 1e-9)
        crossings(
            a,
            md,
            hazard[:born],
            change,
            weights[:born],
            mcut,
            arrivals,
            eligible,
        )
        hazard[:born] += change
        a += md
    if abs(weights.sum() - h[-1]["births_m3"]) > 1e-11 * h[-1][
        "births_m3"
    ] or np.any(weights < 0):
        raise ArithmeticError("Prescribed birth weights did not close")
    if not np.allclose(arrivals[:, -1], eligible, rtol=2e-13):
        raise ArithmeticError("No-capture first-passage count failed")
    surviving = weights[:, None] * np.exp(-hazard)
    audit = rates.audit()
    audit = np.r_[audit, max(audit[2:4]), max(audit[4:6]), 0.0]
    out = []
    for j, diameter in enumerate(markers):
        select = m >= mcut[j]
        terminal = surviving[select].sum(0)
        native = h[-1][f"above_{int(diameter)}nm_new_m3"]
        for k, label in enumerate(MODES):
            out.append(
                dict(
                    background_m3=ambient["background"],
                    source_interval_s=ambient["dt"],
                    ambient_grid_s=2 * stride,
                    trace_interval_s=dt,
                    rate_radius_bins=bins,
                    horizon_s=t[-1],
                    diameter_nm=diameter,
                    mode=label,
                    births_m3=weights.sum(),
                    eligible_m3=eligible[j],
                    arrivals_m3=arrivals[j, k],
                    arrivals_per_1000_formed=1000
                    * arrivals[j, k]
                    / weights.sum(),
                    percent_reaching_diameter_if_age_eligible=100
                    * arrivals[j, k]
                    / eligible[j]
                    if eligible[j]
                    else None,
                    survivors_above_diameter_at_end_m3=terminal[k],
                    equivalent_neutral_reference_m3=native
                    if label == "Fuchs neutral"
                    else None,
                    neutral_reconstruction_pct=100
                    * (terminal[k] - native)
                    / max(native, 1.0)
                    if label == "Fuchs neutral"
                    else None,
                    withheld_pair_rate_max_error_pct=100 * audit[k],
                    wall_s=perf_counter() - start,
                )
            )
    return out


def main(cache, until):
    """Refine tracer passage at three background concentrations."""
    folder = cache / "newborn-charge"
    folder.mkdir(parents=True, exist_ok=True)
    for dt, bins, stride in (
        (0.5, 48, 1),
        (0.25, 96, 1),
        (0.125, 96, 1),
        (0.125, 192, 1),
        (0.125, 192, 2),
        (0.0625, 192, 1),
        (0.0625, 192, 2),
        (0.0625, 96, 1),
    ):
        for bg, coarse in product((1e9, 1e10, 1e11), (False, True)):
            if time() >= until:
                return
            source_dt = (0.25 if bg == 1e9 else 0.0625) * (2 if coarse else 1)
            source = (
                cache
                / "formation-cohorts"
                / (
                    f"cohorts-1e-09-2e-09-{bg:g}-60-1e-07-"
                    f"activation-scavenging-{source_dt:g}-128.json"
                )
            )
            if not source.exists():
                continue
            path = (
                folder
                / f"trace-{bg:g}-{source_dt:g}-{dt:g}-{bins}-{stride}.json"
            )
            if path.exists():
                continue
            rows = simulate(json.loads(source.read_text()), dt, bins, stride)
            tmp = path.with_suffix(".part")
            tmp.write_text(json.dumps(rows) + "\n")
            tmp.replace(path)
            print(path.name, rows[-1]["wall_s"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument(
        "--until",
        required=True,
        type=lambda v: datetime.fromisoformat(v).timestamp(),
    )
    a = p.parse_args()
    main(a.cache, a.until)
