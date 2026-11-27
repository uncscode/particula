"""Test positive exponential midpoint exchange on explicit particle boxes."""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from itertools import count
from pathlib import Path

import torch
from stage3_adaptive_box import advance
from stage3_one_box import Box


class Midpoint(Box):
    """Predict wet radius at the middle of each joint finite-reservoir step."""

    def _transfer(self, radius, j, dt):
        k = self.conductance(radius, j)
        sumk = k.sum(0)
        take = self.gas[j] * (-torch.expm1(-dt * sumk / self.volume))
        return k * (take * self.volume / sumk), take

    def deposit(self, start, stop, increment):
        """Retain the mass transferred to each particle and species."""
        self.mass[:, start + 1 : stop + 1] += increment

    def vapor(self, duration, vmax, tile=4):
        """Use an independent wet-radius clock for midpoint conductances."""
        remaining = duration
        while remaining > duration * 1e-13:
            radius, mass = self.geometry()
            h = remaining
            if vmax:
                rmin = radius.min().reshape(1)
                rate = float(
                    (self.conductance(rmin, slice(None)) * self.gas).sum()
                    / mass.min()
                )
                if rate:
                    h = min(h, 3 * math.log1p(vmax) / rate)
            half_mass = mass.clone()
            for start in range(0, self.species, tile):
                j = slice(start, min(self.species, start + tile))
                increment, _ = self._transfer(radius, j, h / 2)
                half_mass += increment.sum(-1)
            middle = (half_mass * (3 / (4 * math.pi * 1400))).pow(1 / 3)
            for start in range(0, self.species, tile):
                stop = min(self.species, start + tile)
                j = slice(start, stop)
                increment, take = self._transfer(middle, j, h)
                self.deposit(start, stop, increment)
                self.gas[j] -= take
            remaining -= h
            self.vsteps += 1
            if self.vsteps > 100000:
                raise ArithmeticError("Vapor trial budget exhausted")


def jobs():
    """Interleave confirmation cases and matched vapor methods by seed."""
    for rep in count():
        f = (307, 411, 512)[rep % 3]
        for n, s, d, z, h in [
            (16, 2, 1e12, "neutral", 1.0),
            (64, 32, 1e13, "bipolar", 1.0),
            (4096, 32, 1e13, "segregated", 1.0),
            (65536, 2, 1e12, "neutral", 10.0),
            (1000000, 32, 1e12, "neutral", 1.0),
            (1000000, 32, 1e12, "bipolar", 1.0),
            (1000000, 2, 1e13, "neutral", 1.0),
            (16384, 8, 1e9, "neutral", 3600.0),
        ]:
            for label, k, cb, gb, vb, fixed in [
                ("midpoint", "physical", 0.02, 0.015, 0.004, 0),
                ("midpoint", "physical", 0.1, 0.08, 0.02, 0),
                ("midpoint", "learned", 0.1, 0.08, 0.02, 0),
                ("midpoint", "off", 0.1, 0.08, 0.02, 0),
                ("first-order", "physical", 0.1, 0.08, 0.02, 0),
                ("midpoint", "physical", 0.1, 0.08, 0.02, 2),
            ]:
                yield (
                    label,
                    n,
                    s,
                    d,
                    z,
                    f,
                    271001 + rep,
                    k,
                    h,
                    cb,
                    gb,
                    vb,
                    fixed,
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument(
        "--until",
        required=True,
        type=lambda x: datetime.fromisoformat(x).timestamp(),
    )
    p.add_argument("--seed-limit", type=int)
    a = p.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for ix, (method, *j) in enumerate(jobs()):
        if time.time() >= a.until or (
            a.seed_limit and j[5] >= 271001 + a.seed_limit
        ):
            break
        target = a.folder / (method + "-" + "-".join(map(str, j)) + ".json")
        if target.exists():
            continue
        start = datetime.now(timezone.utc).isoformat()
        try:
            row = advance(
                *j[:-1],
                fixed=j[-1],
                box_class=Midpoint if method == "midpoint" else Box,
            )
            row["status"] = "completed"
        except (ArithmeticError, RuntimeError) as error:
            row = dict(
                n=j[0],
                species=j[1],
                density=j[2],
                charge=j[3],
                family=j[4],
                seed=j[5],
                kernel=j[6],
                horizon=j[7],
                steps=j[-1],
                vmax=j[-2],
                collision_budget=j[-4],
                growth_budget=j[-3],
                status="failed",
                why=str(error),
            )
        row.update(
            panel="midpoint-confirm",
            vapor_method=method,
            start=start,
            end=datetime.now(timezone.utc).isoformat(),
        )
        tmp = target.with_suffix(".part")
        tmp.write_text(json.dumps(row) + "\n")
        tmp.replace(target)
        if ix % 24 == 0:
            print(ix, target.name, row["status"], flush=True)
