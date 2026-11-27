"""Advance large particle boxes on independent process clocks."""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stage3_one_box import Box


def collision_interval(box, left, collision_budget, growth_budget, max_dt):
    """Estimate a collision clock from the current population and reservoirs.

    The deterministic probe includes extremes of mass and signed charge. The
    disjoint matching used for actual collisions is independently randomized.
    """
    if box.kernel == "off" or len(box.mass) < 2:
        return left
    radius, mass = box.geometry()
    n = len(mass)
    ids = torch.linspace(0, n - 1, min(24, n), device=box.device).long()
    ids = torch.cat(
        (
            ids,
            torch.stack(
                (
                    radius.argmin(),
                    radius.argmax(),
                    box.z.argmin(),
                    box.z.argmax(),
                )
            ),
        )
    ).unique()
    local = torch.triu_indices(len(ids), len(ids), 1, device=box.device)
    kmax = float(box.rates(radius[ids], mass[ids], box.z[ids], local).max())
    exposure_rate = kmax * (n * (n - 1) / 2) / (n // 2) / box.volume
    smallest = radius.min().reshape(1)
    growth = float(
        (box.conductance(smallest, slice(None)) * box.gas).sum() / mass.min()
    )
    return min(
        left,
        max_dt,
        collision_budget / max(exposure_rate, 1e-300),
        3 * math.log1p(growth_budget) / max(growth, 1e-300),
    )


@torch.inference_mode()
def advance(
    n,
    ns,
    density,
    z,
    f,
    seed,
    k,
    horizon,
    cb,
    gb,
    vb,
    device="cuda",
    fixed=0,
    box_class=Box,
    exponential=True,
):
    """Advance one box and retain independent clock and inventory evidence."""
    gpu = str(device).startswith("cuda")
    if gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    tic = perf_counter()
    b = box_class(n, ns, density, z, device, f, seed, k)
    hist = [b.measure()]
    if gpu:
        torch.cuda.synchronize()
    after = perf_counter()
    steps = []
    for _ in range(4):
        left = horizon / 4
        while left > horizon * 1e-13:
            dt = (
                min(left, horizon / (4 * fixed))
                if fixed
                else collision_interval(b, left, cb, gb, horizon / 4)
            )
            b.vapor(dt / 2, vb)
            b.collide(dt, exponential=exponential)
            b.vapor(dt / 2, vb)
            left -= dt
            steps.append(dt)
            if len(steps) > 100000:
                raise ArithmeticError("Collision time grid exceeded budget")
        hist.append(b.measure())
    if gpu:
        torch.cuda.synchronize()
    return dict(
        n=n,
        species=ns,
        density=density,
        charge=z,
        family=f,
        seed=seed,
        kernel=k,
        horizon=horizon,
        steps=fixed,
        vmax=vb,
        collision_budget=cb,
        growth_budget=gb,
        device=device,
        actual_collision_steps=b.colsteps,
        actual_vapor_steps=b.vsteps,
        minimum_step=min(steps),
        maximum_step=max(steps),
        mean_step=np.mean(steps),
        max_probability=b.max_probability,
        setup_s=after - tic,
        advance_s=perf_counter() - after,
        total_s=perf_counter() - tic,
        pairs=b.pairs,
        history=hist,
        cuda_peak_allocated=torch.cuda.max_memory_allocated() if gpu else 0,
    )


def jobs():
    """Alternate size, charge and elapsed time before adding seeds."""
    for rep in count():
        f = (101, 102, 205, 308)[rep % 4]
        seed = 261001 + rep
        for n, s, d, z, h in [
            (65536, 2, 1e12, "neutral", 1.0),
            (1000000, 32, 1e12, "neutral", 1.0),
            (1000000, 32, 1e12, "segregated", 1.0),
            (4096, 32, 1e14, "segregated", 1.0),
            (65536, 2, 1e13, "bipolar", 1.0),
            (16384, 8, 1e12, "neutral", 60.0),
            (65536, 32, 1e12, "neutral", 10.0),
            (1000000, 2, 1e11, "bipolar", 10.0),
            (16384, 8, 1e9, "neutral", 3600.0),
            (65536, 8, 1e11, "bipolar", 60.0),
        ]:
            for k, cb, gb, vb, fixed in [
                ("physical", 0.025, 0.025, 0.004, 0),
                ("physical", 0.1, 0.08, 0.015, 0),
                ("learned", 0.1, 0.08, 0.015, 0),
                ("off", 0.1, 0.08, 0.015, 0),
                ("physical", 0.1, 0.08, 0.015, 4),
                ("physical", 0.1, 0.08, 0.015, 1),
            ]:
                yield n, s, d, z, f, seed, k, h, cb, gb, vb, fixed


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument(
        "--until",
        required=True,
        type=lambda x: datetime.fromisoformat(x).timestamp(),
    )
    p.add_argument("--seed-limit", type=int)
    args = p.parse_args()
    args.folder.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1)
    for ix, j in enumerate(jobs()):
        if time.time() >= args.until or (
            args.seed_limit and j[5] >= 261001 + args.seed_limit
        ):
            break
        name = "clock-" + "-".join(map(str, j)) + ".json"
        path = args.folder / name
        if path.exists():
            continue
        start = datetime.now(timezone.utc).isoformat()
        try:
            row = advance(*j[:-1], fixed=j[-1])
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
                collision_budget=j[8],
                growth_budget=j[9],
                vmax=j[10],
                steps=j[11],
                status="failed",
                why=str(error),
            )
        row.update(
            panel="adaptive-exp",
            start=start,
            end=datetime.now(timezone.utc).isoformat(),
        )
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(row) + "\n")
        tmp.replace(path)
        if ix % 16 == 0:
            print(ix, name, row["status"], flush=True)
