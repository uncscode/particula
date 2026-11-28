"""Advance selected forecast intervals with native Warp physics."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from itertools import count, product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stage3_midpoint import Midpoint


def native_interval(box, dt, seed, family, micro):
    """Advance native condensation and coagulation from the live forecast."""
    from particula.gpu.kernels._coagulation_config import (
        CoagulationMechanismConfig,
    )
    from stage2_native_split import NativeSplit
    from stage3_native_scale import box as native_box

    tic = perf_counter()
    n = len(box.mass)
    state = native_box(
        n, box.species, n / box.volume, family, dt, box.charge_mode
    )
    # Native arrays place dry material last. Forecasts store it first.
    current = box.mass.cpu().numpy()
    state.mass = np.column_stack((current[:, 1:], current[:, 0]))
    state.gas = box.gas.cpu().numpy().copy()
    state.charge = box.z.cpu().numpy().astype(float)
    state.volume = box.volume
    state.t = np.array([0.0, dt])
    state.levels = state.gas[None, :]
    mech = (
        None
        if box.charge_mode == "neutral"
        else CoagulationMechanismConfig(
            mechanisms=("charged_hard_sphere",),
            charged_model="gopalakrishnan2012",
        )
    )
    driver = NativeSplit([state], 1, micro, mechanism_config=mech)
    driver.run([state], [seed])
    p = driver.owner.particles
    w = p.population.numpy()[0]
    if not np.isin(w, [0.0, 1.0]).all():
        raise ArithmeticError("Native refresh changed unit multiplicity")
    live = w == 1
    m = p.masses.numpy()[0, live]
    z = p.charge.numpy()[0, live]
    gas = driver.owner.gas.concentration.numpy()[0]
    if np.any(m < 0) or np.any(gas < 0) or not np.array_equal(np.rint(z), z):
        raise ArithmeticError("Native refresh returned invalid phase or charge")
    box.mass = torch.as_tensor(
        np.column_stack((m[:, -1], m[:, :-1])), device=box.device
    )
    box.z = torch.as_tensor(z, dtype=torch.int64, device=box.device)
    box.gas = torch.as_tensor(gas.copy(), device=box.device)
    box.collisions += n - len(m)
    torch.cuda.synchronize()
    return perf_counter() - tic


@torch.inference_mode()
def trial(n, s, z, f, seed, policy, k, h, d, steps, micro):
    """Time intervals including graph setup, state copies and observation."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    b = Midpoint(n, s, d, z, "cuda", f, seed, k)
    history = [b.measure()]
    torch.cuda.synchronize()
    setup = perf_counter() - start
    native_s = 0.0
    ticks = []
    # The mask is registered before the run and never reads reference output.
    use = {
        "free": (),
        "early": (0,),
        "middle": (1,),
        "late": (3,),
        "two": (1, 3),
        "all": (0, 1, 2, 3),
    }[policy]
    for frame in range(4):
        if frame in use:
            correction_seed = int(
                np.random.SeedSequence([seed, frame, 4421]).generate_state(1)[0]
            )
            native_s += native_interval(b, h / 4, correction_seed, f, micro)
            ticks.append(frame)
        else:
            dt = h / (4 * steps)
            for _ in range(steps):
                b.vapor(dt / 2, 0.01)
                b.collide(dt, exponential=True)
                b.vapor(dt / 2, 0.01)
        history.append(b.measure())
    torch.cuda.synchronize()
    return dict(
        n=n,
        species=s,
        density=d,
        charge=z,
        family=f,
        seed=seed,
        kernel=k,
        horizon=h,
        steps=steps,
        vmax=0.01,
        vapor_method="midpoint",
        panel="native-refresh-" + policy + f"-m{micro}",
        policy=policy,
        native_micro=micro,
        native_intervals=ticks,
        native_wall_s=native_s,
        total_s=perf_counter() - start,
        setup_s=setup,
        pairs=b.pairs,
        history=history,
        final_count=len(b.mass),
        sampled_collision_steps=b.colsteps,
        sampled_vapor_steps=b.vsteps,
        torch_peak_bytes=torch.cuda.max_memory_allocated(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--warp-site", type=Path)
    p.add_argument(
        "--until",
        type=lambda s: datetime.fromisoformat(s).timestamp(),
        required=True,
    )
    p.add_argument("--sizes", nargs="+", type=int, default=[256, 4096, 16384])
    p.add_argument("--vapors", nargs="+", type=int, default=[2, 32])
    p.add_argument("--charges", nargs="+", default=["neutral", "bipolar"])
    p.add_argument("--micro", nargs="+", type=int, default=[4, 128])
    p.add_argument("--fine-limit", type=int, default=4096)
    p.add_argument("--seeds", type=int)
    p.add_argument("--placement", action="store_true")
    a = p.parse_args()
    if a.warp_site:
        sys.path.append(str(a.warp_site.resolve()))
    import warp as wp

    wp.init()
    torch.set_num_threads(1)
    a.folder.mkdir(parents=True, exist_ok=True)
    for rep in count():
        if time.time() >= a.until or (a.seeds and rep >= a.seeds):
            break
        seed = 321001 + rep
        cases = product(
            a.sizes,
            a.vapors,
            a.charges,
            ["free", "middle", "two", "all"],
            a.micro,
        )
        if a.placement:
            cases = [
                (4096, 32, z, policy, 4 if policy == "free" else 128)
                for z in ("neutral", "bipolar")
                for policy in ("free", "early", "middle", "late")
            ]
            np.random.default_rng(seed).shuffle(cases)
        for n, s, z, policy, micro in cases:
            if time.time() >= a.until:
                break
            if micro != a.micro[0] and (policy == "free" or n > a.fine_limit):
                continue
            path = (
                a.folder / f"refresh-{n}-{s}-{z}-{seed}-{policy}-{micro}.json"
            )
            if path.exists():
                continue
            start = datetime.now(timezone.utc).isoformat()
            try:
                row = trial(
                    n, s, z, 307, seed, policy, "learned", 1.0, 1e12, 1, micro
                )
                row["status"] = "completed"
            except (ValueError, ArithmeticError, RuntimeError) as err:
                row = dict(
                    status="failed",
                    n=n,
                    species=s,
                    charge=z,
                    policy=policy,
                    seed=seed,
                    native_micro=micro,
                    why=str(err),
                )
            row.update(start=start, end=datetime.now(timezone.utc).isoformat())
            tmp = path.with_suffix(".part")
            tmp.write_text(json.dumps(row) + "\n")
            tmp.replace(path)
            print(path.name, row["status"], flush=True)
