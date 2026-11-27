"""Integrate a continuous Brownian collision clock during vapor exchange."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from stiff_exchange import rhs

# Particula air at 293 K and 101325 Pa, checked against the owning kernel.
MU_AIR = 1.8126042751588682e-5
MFP_AIR = 6.502269421801076e-8
KT = 1.380649e-23 * 293.0


def mobility(dry, water_ratio):
    """Compute the wet-particle inputs to the Brownian transition kernel."""
    wet = np.maximum(water_ratio, 0)
    radius = np.cbrt(3 * dry * (1 / 1400 + wet / 1000) / (4 * np.pi))
    kn = MFP_AIR / radius
    slip = 1 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))
    diffusivity = KT * slip / (6 * np.pi * MU_AIR * radius)
    speed = np.sqrt(8 * KT / (np.pi * dry * (1 + wet)))
    length = 8 * diffusivity / (np.pi * speed)
    collection = (
        (2 * radius + length) ** 3 - (4 * radius**2 + length**2) ** 1.5
    ) / (6 * radius * length) - 2 * radius
    return np.stack((radius, diffusivity, collection, speed))


def pair_rates(left, right):
    """Return Kij in m³/s for each corresponding pair of wet particles."""
    r, d, _, _ = left + right
    g = np.hypot(left[2], right[2])
    v = np.hypot(left[3], right[3])
    return 4 * np.pi * d * r / (r / (r + g) + 4 * d / (r * v))


class CollisionClock:
    """Evolve node water and cumulative hazard up to a collision or reset."""

    def __init__(self, mass, effective_volume, total, exchange, coagulate):
        """Preserve dry material and box volume between collisions."""
        self.dry = mass[:, 1:].sum(1)
        self.kappa = (0.03 * mass[:, 1] + 0.75 * mass[:, 2]) / self.dry
        self.volume = effective_volume
        self.weight = self.dry / effective_volume
        self.total = total
        self.exchange = exchange
        self.i, self.j = np.triu_indices(len(mass), 1)
        self.coagulate = coagulate and bool(len(self.i))

    def rates(self, y):
        """Exclude self collisions and account for equal represented counts."""
        a = mobility(self.dry, y)
        return pair_rates(a[:, self.i], a[:, self.j]) / self.volume

    def function(self, t, y):
        """Integrate water exchange and the changing total collision rate."""
        ratio = y[:-1]
        flux = (
            rhs(t, ratio, self.dry, self.kappa, self.weight, self.total)
            if self.exchange
            else np.zeros_like(ratio)
        )
        rate = self.rates(ratio).sum() if self.coagulate else 0.0
        return np.r_[flux, rate]

    def jacobian(self, t, y):
        """Build the BDF Jacobian with two vectorized endpoint perturbations."""
        x = y[:-1]
        dx = 1e-7 * np.maximum(np.abs(x), 0.1)
        jac = np.zeros((len(y), len(y)))
        if self.exchange:
            current = rhs(t, x, self.dry, self.kappa, self.weight, self.total)
            perturbed = x[:, None] + np.diag(dx)
            f = rhs(
                t,
                perturbed,
                self.dry[:, None],
                self.kappa[:, None],
                self.weight,
                self.total,
            )
            jac[:-1, :-1] = (f - current[:, None]) / dx
        if self.coagulate:
            a = mobility(self.dry, x)
            updated = mobility(self.dry, x + dx)
            left, right = a[:, self.i], a[:, self.j]
            reference = pair_rates(left, right)
            extra_i = pair_rates(updated[:, self.i], right) - reference
            extra_j = pair_rates(left, updated[:, self.j]) - reference
            gradient = np.bincount(self.i, extra_i, len(x)) + np.bincount(
                self.j, extra_j, len(x)
            )
            jac[-1, :-1] = gradient / (dx * self.volume)
        return jac


def history(
    initial, volume, gases, ends, rng, rtol, exchange=True, coagulate=True
):
    """Integrate one closed leg at a time, preserving the residual clock."""
    mass = initial.copy()
    dry_total = mass[:, 1:].sum(0)
    count0 = len(mass)
    start = 0.0
    remaining = rng.exponential()
    events = []
    rows = []
    for gas, end in zip(gases, ends, strict=True):
        total = gas + mass[:, 0].sum() / volume
        budget = 0.0
        while start < end:
            clock = CollisionClock(mass, volume, total, exchange, coagulate)
            y0 = np.r_[mass[:, 0] / clock.dry, 0.0]

            def crossing(_, y, threshold=remaining):
                return y[-1] - threshold

            crossing.terminal = True
            crossing.direction = 1
            sol = solve_ivp(
                clock.function,
                (start, end),
                y0,
                method="BDF",
                jac=clock.jacobian,
                rtol=rtol,
                atol=rtol * 1e-3,
                events=crossing if clock.coagulate else None,
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            water = clock.weight @ sol.y[:-1]
            if sol.y[:-1].min() < -1e-11 or (total - water).min() < -1e-13:
                raise FloatingPointError("Invalid continuous water solution")
            end_y = sol.y[:, -1]
            mass[:, 0] = end_y[:-1] * clock.dry
            gas = total - mass[:, 0].sum() / volume
            start = sol.t[-1]
            if sol.status != 1:
                remaining -= end_y[-1]
                if remaining <= 0:
                    raise ArithmeticError("Unprocessed collision before reset")
                break
            rates = clock.rates(end_y[:-1])
            cdf = np.cumsum(rates)
            which = np.searchsorted(cdf, rng.random() * cdf[-1], side="right")
            i, j = clock.i[which], clock.j[which]
            events.append((float(start), int(i), int(j)))
            residual = abs(end_y[-1] - remaining)
            if residual > max(1e-9, remaining * 1e-7):
                raise ArithmeticError("Unresolved collision clock crossing")
            mass[i] += mass[j]
            mass = np.delete(mass, j, 0)
            budget = max(budget, abs(gas + mass[:, 0].sum() / volume - total))
            remaining = rng.exponential()
        np.testing.assert_allclose(mass[:, 1:].sum(0), dry_total, rtol=2e-13)
        diameter = 2e6 * np.cbrt(
            3 * (mass[:, 0] / 1000 + mass[:, 1:].sum(1) / 1400) / (4 * np.pi)
        )
        rows.append(
            dict(
                number_fraction=len(mass) / count0,
                mean_wet_diameter_um=diameter.mean(),
                water_dry_ratio=mass[:, 0].sum() / dry_total.sum(),
                gas_kg_m3=gas,
                water_kg_m3=mass[:, 0].sum() / volume,
                budget_residual_kg_m3=budget,
                total_collisions=len(events),
            )
        )
    return rows, events


def run(source, directory, seed, first, groups, rtol):
    """Save individual box results and collision times after each family."""
    with np.load(source) as f:
        data = {k: f[k] for k in f.files}
    if np.any(data["ids"] >= 700000):
        raise ValueError(
            "Use the independent validation inputs for this reference"
        )
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    slots = data["mass"].shape[1]
    directory.mkdir(exist_ok=True, parents=True)
    selected = np.unique(data["ids"])[first : first + groups]
    if len(selected) != groups:
        raise ValueError("Family index outside validation input")
    for family in selected:
        output = directory / f"k{slots}-f{family}-s{seed}-r{rtol:.0e}.csv"
        event_file = output.with_suffix(".json")
        if output.exists() and event_file.exists():
            metadata = json.loads(event_file.read_text())
            if metadata["input_sha256"] != sha or metadata["rtol"] != rtol:
                raise ValueError("Reference configuration mismatch")
            continue
        rows, paths = [], []
        for box in np.flatnonzero(data["ids"] == family):
            m = data["mass"][box]
            c = data["counts"][box]
            keep = c > 0
            if not np.all(c[keep] == c[keep][0]):
                raise ValueError(
                    "Continuous collision clock assumes equal counts"
                )
            density, mix = (
                int(data["density_multiplier"][box]),
                int(data["mix"][box]),
            )
            rng = np.random.default_rng(
                [663101, slots, int(family), density, mix, seed]
            )
            stats, events = history(
                m[keep],
                float(data["volume"]) / c[keep][0],
                [data["gas"][box], data["reset"][box]],
                [data["time"][8], data["time"][-1]],
                rng,
                rtol,
            )
            for leg, values in zip(("wet_end", "dry_end"), stats, strict=True):
                rows.append(
                    dict(
                        family=family,
                        density_multiplier=density,
                        mix=mix,
                        leg=leg,
                        slots=slots,
                        seed=seed,
                        rtol=rtol,
                        **values,
                    )
                )
            paths.append(dict(density=density, mix=mix, events=events))
        tmp = output.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
            wr.writeheader()
            wr.writerows(rows)
        pending = event_file.with_suffix(".json.tmp")
        pending.write_text(
            json.dumps(dict(input_sha256=sha, rtol=rtol, paths=paths))
        )
        pending.replace(event_file)
        tmp.replace(output)
        print(output, sum(len(p["events"]) for p in paths), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("directory", type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--groups", type=int, default=6)
    p.add_argument("--rtol", type=float, default=1e-7)
    run(**vars(p.parse_args()))
