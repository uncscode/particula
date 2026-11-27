"""Advance finite ions, multivapor uptake and stochastic aerosol collisions."""

import argparse
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from itertools import count
from pathlib import Path
from time import perf_counter, time

import numpy as np
import torch
from stage3_adaptive_box import collision_interval
from stage3_charging_grid import POLICIES, attach
from stage3_ion_clocks import ION_MASS, RECOMB
from stage3_midpoint import Midpoint


@lru_cache(maxsize=4)
def coefficients(points=1024, qmax=32):
    """Build the native ion lookup over integer charge and log wet radius."""
    rad = np.geomspace(1e-9, 5e-6, points)
    charge = np.arange(-qmax, qmax + 1)[:, None]
    return np.stack([attach(rad[None, :], charge, sign) for sign in (1, -1)])


class ChargingBox(Midpoint):
    """Evolve one real charge state for each explicit aerosol particle."""

    def init_ions(self, policy, seed):
        """Start with one finite free ion of each sign per initial aerosol."""
        self.initial_q = self.q0
        self.policy = policy
        self.free = [self.n0, self.n0]
        self.attached = [0, 0]
        self.recombined = 0
        self.ion_steps = 0
        self.limited = 0
        self.ion_rng = torch.Generator(device=self.device).manual_seed(
            seed + 1000000
        )
        self.table = torch.as_tensor(
            coefficients(), device=self.device, dtype=self.dtype
        )
        if policy != "fixed":
            self.charge_mode = "dynamic"

    def ion_coefficients(self, r):
        """Interpolate native ion rates for the current particles."""
        qmax = (self.table.shape[1] - 1) // 2
        if (
            int(self.z.abs().max()) > qmax
            or float(r.min()) < 1e-9
            or float(r.max()) > 5e-6
        ):
            raise ArithmeticError("Particle outside the ion coefficient table")
        x = torch.log(r / 1e-9) * ((self.table.shape[2] - 1) / math.log(5000))
        i = x.floor().long().clamp(0, self.table.shape[2] - 2)
        frac = x - i
        left = self.table[:, self.z + qmax, i]
        return torch.lerp(left, self.table[:, self.z + qmax, i + 1], frac)

    def apply_attachments(self, mask, sign):
        """Distribute the remaining finite ions over requesting particles."""
        lane = int(sign < 0)
        wanted = int(mask.sum())
        nhit = min(self.free[lane], wanted)
        if nhit < wanted:
            self.limited += wanted - nhit
            ids = torch.where(mask)[0]
            take = torch.randperm(
                wanted, device=self.device, generator=self.ion_rng
            )[:nhit]
            self.z[ids[take]] += sign
        else:
            self.z.add_(mask, alpha=sign)
        self.free[lane] -= nhit
        self.attached[lane] += nhit

    def ions(self, duration, budget):
        """Bound depletion and per-particle hazards on the ion clock."""
        left = duration
        r, _ = self.geometry()
        while left > duration * 1e-13:
            fp, fn = self.free
            if max(fp, fn) == 0:
                break
            live = self.policy != "fixed"
            scale = 1 / self.volume
            if live:
                kp, kn = self.ion_coefficients(r)
                pp, pn = kp * (fp * scale), kn * (fn * scale)
                worst = max(
                    float((pp + pn).max()),
                    (float(kp.sum()) + RECOMB * fn) * scale,
                    (float(kn.sum()) + RECOMB * fp) * scale,
                )
            else:
                worst = RECOMB * max(fp, fn) * scale
            h = min(left, budget / max(worst, 1e-300))
            if fp and fn:
                small, large = min(fp, fn), max(fp, fn)
                num = torch.tensor(
                    float(small), dtype=self.dtype, device=self.device
                )
                prob = torch.full_like(num, h * RECOMB * large * scale)
                rec = int(torch.binomial(num, prob, generator=self.ion_rng))
                self.recombined += rec
                self.free[0] -= rec
                self.free[1] -= rec
            if live:
                u = torch.rand(
                    len(r),
                    generator=self.ion_rng,
                    device=self.device,
                    dtype=self.dtype,
                )
                plus = u < h * pp
                minus = (~plus) & (u < h * (pp + pn))
                self.apply_attachments(plus, 1)
                self.apply_attachments(minus, -1)
            self.ion_steps += 1
            left -= h
            if self.ion_steps > 100000:
                raise ArithmeticError("Ion interval budget exhausted")

    def measure_ions(self):
        """Close signed charge and both finite ion stocks independently."""
        pos, neg = self.attached
        self.q0 = self.initial_q + pos - neg
        if any(
            f + a + self.recombined != self.n0
            for f, a in zip(self.free, self.attached, strict=True)
        ):
            raise ArithmeticError("Lost ion material")
        if int(self.z.sum()) + self.free[0] - self.free[1] != self.initial_q:
            raise ArithmeticError("Lost total electric charge")
        h = super().measure()
        tracer_mass = (pos * ION_MASS[0] + neg * ION_MASS[1]) / self.volume
        h.update(
            free_positive=self.free[0],
            free_negative=self.free[1],
            attached_positive=pos,
            attached_negative=neg,
            recombined=self.recombined,
            charged_count=int(torch.count_nonzero(self.z)),
            charge_hist=torch.bincount(
                (self.z + 9).clamp(0, 18), minlength=19
            ).tolist(),
            mean_abs_charge=float(self.z.abs().double().mean()),
            ion_tracer_mass_fraction=tracer_mass / sum(h["particle"]),
        )
        return h


@torch.inference_mode()
def forecast(
    n,
    s,
    z,
    f,
    seed,
    policy,
    ib,
    device="cuda",
    horizon=1.0,
    ion_order="outside",
):
    """Measure one complete live-charge, fixed-charge or collision-free path."""
    if ion_order not in ("outside", "inside"):
        raise ValueError("Unsupported ion order")
    gpu = str(device).startswith("cuda")
    if gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    k = "off" if policy == "no-collisions" else "physical"
    b = ChargingBox(n, s, 1e12, z, device, f, seed, k)
    b.init_ions(policy, seed)
    hist = [b.measure_ions()]
    if gpu:
        torch.cuda.synchronize()
    setup = perf_counter() - start
    tic = perf_counter()
    for _ in range(4):
        left = horizon / 4
        while left > horizon * 1e-13:
            dt = collision_interval(b, left, 0.05, 0.04, horizon / 16)
            dt = min(dt, horizon / 16)
            if ion_order == "outside":
                b.ions(dt / 2, ib)
                b.vapor(dt / 2, 0.01)
            else:
                b.vapor(dt / 2, 0.01)
                b.ions(dt / 2, ib)
            b.collide(dt, exponential=True)
            if ion_order == "outside":
                b.vapor(dt / 2, 0.01)
                b.ions(dt / 2, ib)
            else:
                b.ions(dt / 2, ib)
                b.vapor(dt / 2, 0.01)
            left -= dt
        hist.append(b.measure_ions())
    if gpu:
        torch.cuda.synchronize()
    duration = perf_counter() - tic
    return dict(
        status="completed",
        n=n,
        species=s,
        charge=z,
        seed=seed,
        family=f,
        density=1e12,
        horizon=horizon,
        history=hist,
        policy=policy,
        ion_budget=ib,
        ion_order=ion_order,
        device=device,
        actual_ion_steps=b.ion_steps,
        actual_collision_steps=b.colsteps,
        actual_vapor_steps=b.vsteps,
        limited_attachments=b.limited,
        collisions=b.collisions,
        setup_s=setup,
        advance_s=duration,
        total_s=duration + setup,
        cuda_peak_bytes=torch.cuda.max_memory_allocated() if gpu else 0,
    )


def jobs():
    """Alternate sizes and initial charge with independent event seeds."""
    for rep in count():
        rng = np.random.default_rng(520001 + rep)
        first = [
            (1000000, 32, z, mode, 0.08, "outside")
            for z in ("neutral", "bipolar4")
            for mode in POLICIES
        ]
        rest = [
            (n, s, z, "live", ib, order)
            for n, s in ((4096, 2), (65536, 8), (1000000, 32))
            for z in ("neutral", "bipolar4")
            for ib, order in ((0.025, "outside"), (0.08, "inside"))
        ]
        for row in (first, rest):
            for i in rng.permutation(len(row)):
                yield (*row[i], 520001 + rep)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument(
        "--until",
        required=True,
        type=lambda x: datetime.fromisoformat(x).timestamp(),
    )
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for n, s, z, mode, ib, order, seed in jobs():
        if time() >= a.until:
            break
        path = a.folder / f"charge-{n}-{s}-{z}-{mode}-{ib}-{order}-{seed}.json"
        if path.exists():
            continue
        start = datetime.now(timezone.utc).isoformat()
        try:
            row = forecast(
                n, s, z, 307, seed, mode, ib, a.device, ion_order=order
            )
        except (ValueError, ArithmeticError, RuntimeError) as err:
            row = dict(
                status="failed",
                why=str(err),
                n=n,
                species=s,
                charge=z,
                policy=mode,
                seed=seed,
                ion_budget=ib,
                ion_order=order,
            )
        row.update(start=start, end=datetime.now(timezone.utc).isoformat())
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(row) + "\n")
        tmp.replace(path)
