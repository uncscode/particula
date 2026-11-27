"""Compare independent vapor and collision clocks on the Stage 2 box physics."""

import argparse
import json
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stage2_population import fixture, observable_names
from stage2_split import Split


class Multirate(Split):
    """Embedded vapor step-doubling and a geometry-limited collision clock."""

    def vapor_for(self, state, duration):
        """Refine vapor transfer until the embedded estimates agree."""
        remaining = duration
        h = duration
        while remaining > 1e-14 * duration:
            h = min(h, remaining)
            coarse = state.clone()
            fine = state.clone()
            self.vapor(coarse, h, 1)
            self.vapor(fine, h, 2)
            s = self.spec.species
            scale = (
                self.initial_inventory[:, None, :s]
                * state.volume[:, None, None]
                / self.initial_slots
            )
            particle = (
                (fine.mass[..., :s] - coarse.mass[..., :s]).abs()
                / (fine.mass[..., :s].abs() + scale * 1e-7).clamp_min(1e-35)
            ).amax()
            gas = (
                (fine.gas - coarse.gas).abs()
                / (
                    fine.gas.abs() + self.initial_inventory[:, :s] * 1e-7
                ).clamp_min(1e-23)
            ).amax()
            err = max(float(gas), float(particle))
            self.work["vapor_trials"] += 1
            if err <= self.vtol:
                state.mass, state.gas, state.vapor_memory = (
                    fine.mass,
                    fine.gas,
                    fine.vapor_memory,
                )
                remaining -= h
                self.work["vapor_accepted"] += 1
            else:
                self.work["vapor_rejected"] += 1
            h *= min(
                2.5, max(0.2, 0.85 * (self.vtol / max(err, 1e-30)) ** (1 / 4))
            )
            if h < 1e-13 * duration:
                raise ArithmeticError(
                    "Vapor error control exhausted the step size"
                )

    def event_step(self, state, duration, idx):
        """Locate the earliest crossing on a locally refined hazard clock."""
        remaining = duration
        h = remaining
        while remaining > 1e-14 * duration:
            h = min(h, remaining)
            trial = state.clone()
            self.vapor_for(trial, h)
            if self.kernel == "off":
                state.mass, state.gas, state.vapor_memory = (
                    trial.mass,
                    trial.gas,
                    trial.vapor_memory,
                )
                return
            start = self.pair_rates(state)[0] / state.volume[:, None]
            end = self.pair_rates(trial)[0] / state.volume[:, None]
            h0, h1 = start.sum(-1), end.sum(-1)
            change = float(
                ((start - end).abs().sum(-1) / h0.clamp_min(1e-60)).max()
            )
            if change > self.ctol:
                self.work["collision_rejected"] += 1
                h *= max(0.15, 0.8 * self.ctol / change)
                if h < 1e-13 * duration:
                    raise ArithmeticError("Event clock exhausted the step size")
                continue
            whole = 0.5 * h * (h0 + h1)
            occurs = (whole >= state.remaining) & (whole > 0)
            fraction = 1.0
            if bool(occurs.any()):
                row = torch.where(occurs)[0]
                debt = state.remaining[row] / h
                disc = (
                    h0[row].square() + 2 * (h1[row] - h0[row]) * debt
                ).clamp_min(0.0)
                roots = 2 * debt / (h0[row] + disc.sqrt()).clamp_min(1e-300)
                first = int(roots.argmin())
                fraction = min(1.0, max(0.0, float(roots[first])))
                event_row = row[first : first + 1]
                if fraction < 1:
                    trial = state.clone()
                    self.vapor_for(trial, h * fraction)
            trial.remaining = state.remaining - h * fraction * (
                h0 + 0.5 * fraction * (h1 - h0)
            )
            if bool(occurs.any()):
                cdf = (
                    self.pair_rates(trial)[0] / trial.volume[:, None]
                ).cumsum(-1)
                self.merge_selected(trial, event_row, idx, cdf)
            if bool((trial.remaining < -1e-10).any()):
                raise ArithmeticError(
                    "Another event preceded selected crossing"
                )
            for key, val in vars(trial).items():
                setattr(state, key, val)
            remaining -= h * fraction
            self.work["collision_accepted"] += 1
            h *= min(2.5, max(1.0, 0.85 * self.ctol / max(change, 1e-30)))

    def step(self, state, dt, idx, *args):
        """Advance one observation interval with independent process clocks."""
        if self.event_aware:
            self.event_step(state, dt, idx)
            return
        remaining = dt
        h = dt
        while remaining > 1e-14 * dt:
            h = min(h, remaining)
            proposal = state.clone()
            self.vapor_for(proposal, h / 2)
            change = 0.0
            if self.kernel != "off":
                old = self.pair_rates(state)[0]
                mid = self.pair_rates(proposal)[0]
                # Include changes in the distribution of pair rates.
                change = float(
                    (
                        (old - mid).abs().sum(-1) / old.sum(-1).clamp_min(1e-60)
                    ).max()
                )
            if change > self.ctol:
                self.work["collision_rejected"] += 1
                h *= max(0.15, 0.8 * self.ctol / change)
                if h < 1e-13 * dt:
                    raise ArithmeticError(
                        "Collision geometry controller exhausted the step size"
                    )
                continue
            self.coag(proposal, h, idx)
            self.vapor_for(proposal, h / 2)
            for key, value in vars(proposal).items():
                setattr(state, key, value)
            self.work["collision_accepted"] += 1
            remaining -= h
            h *= min(2.5, max(1.0, 0.85 * self.ctol / max(change, 1e-30)))


def run(a, driver_type=Multirate, chemistry=None, experiment=None):
    """Save matched trajectories and actual accepted and rejected work."""
    keys = list(
        product(
            a.families, a.mix, a.densities, range(a.start, a.start + a.seeds)
        )
    )
    a.folder.mkdir(parents=True, exist_ok=True)
    tag = (
        f"{a.process}-{a.charge}-{Path(a.kernel).stem}-n{a.slots}"
        f"-v{a.vtol:g}-c{a.ctol:g}"
    )
    if experiment:
        tag = experiment + "-" + tag
    if a.event_aware:
        tag += "-localized"
    if a.plateau:
        tag += f"-plateau{a.plateau}"
    target = a.folder / (tag + ".npz")
    if target.exists():
        with np.load(target) as old:
            np.testing.assert_array_equal(keys, old["keys"])
        return
    outputs, durations, work = [], [], []
    max_drift = 0.0
    for first in range(0, len(keys), a.batch):
        block = keys[first : first + a.batch]
        frames = 8
        boxes = [
            fixture(
                f, m, a.slots, d, a.process, a.charge, 8, 4, chemistry=chemistry
            )
            for f, m, d, _ in block
        ]
        if a.plateau:
            from stage2_dilution import box

            boxes = []
            for f, m, d, _ in block:
                b, frames = box(f, m, d, a.slots, a.plateau)
                if a.charge != "zero":
                    raise ValueError("Dilute fixture is neutral")
                boxes.append(b)
        tic = perf_counter()
        driver = driver_type(
            boxes,
            [k[-1] for k in block],
            getattr(a, "transfer", "radau-3-4"),
            a.kernel,
            a.device,
        )
        driver.event_aware = a.event_aware
        driver.options = a
        s = driver.start()
        driver.initial_inventory = s.mass.sum(1) / s.volume[:, None]
        driver.initial_inventory[:, : driver.spec.species] += s.gas
        driver.vtol, driver.ctol = a.vtol, a.ctol
        driver.work = dict(
            vapor_trials=0,
            vapor_accepted=0,
            vapor_rejected=0,
            collision_accepted=0,
            collision_rejected=0,
        )
        obs, drift, _ = driver.rollout()
        max_drift = max(max_drift, drift)
        outputs.append(obs)
        work.append(driver.work)
        durations.append(perf_counter() - tic)
    meta = dict(
        process=getattr(
            a,
            "reference_process",
            "dilute-" + a.process if a.plateau else a.process,
        ),
        charge=a.charge,
        slots=a.slots,
        frames=frames,
        legs=4,
        representation="full",
        scheme="multirate-localized" if a.event_aware else "multirate",
        kernel=Path(a.kernel).stem,
        transfer=Path(getattr(a, "transfer", "radau-3-4")).stem,
        device=a.device,
        vtol=a.vtol,
        ctol=a.ctol,
        batch=a.batch,
    )
    if experiment:
        meta["experiment"] = experiment
    tmp = target.with_suffix(".part.npz")
    np.savez_compressed(
        tmp,
        keys=keys,
        t=boxes[0].t,
        names=observable_names(boxes[0].spec),
        obs=np.concatenate(outputs),
        max_drift=max_drift,
        seconds=durations,
        meta=json.dumps(meta),
        work=json.dumps(work),
    )
    tmp.replace(target)
    print(tag, len(keys), sum(durations), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--families", nargs="+", type=int, default=[81001, 81002])
    p.add_argument("--mix", nargs="+", type=int, default=[0, 1])
    p.add_argument("--densities", nargs="+", type=int, default=[1, 8])
    p.add_argument("--start", type=int, default=211001)
    p.add_argument("--seeds", type=int, default=64)
    p.add_argument("--slots", type=int, default=16)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--vtol", type=float, default=0.003)
    p.add_argument("--ctol", type=float, default=0.02)
    p.add_argument("--device", default="cpu")
    p.add_argument("--kernel", default="brownian")
    p.add_argument("--charge", default="zero")
    p.add_argument("--process", default="coupled")
    p.add_argument("--event-aware", action="store_true")
    p.add_argument("--plateau", type=int, default=0)
    torch.set_num_threads(1)
    with torch.inference_mode():
        run(p.parse_args())
