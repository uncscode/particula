"""Test conservative operators at lower number density and hour-long forcing."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stage2_hybrid import correct, hybrid
from stage2_population import fixture, observable_names
from stage2_reference import reference
from stage2_split import Split


def box(f, m, concentration, slots, plateau):
    """Set concentration and resolve the first transient on each plateau."""
    b = fixture(f, m, slots, 1)
    b.volume = slots / float(concentration)
    available = np.array(
        [
            0.0,
            0.001,
            0.003,
            0.01,
            0.03,
            0.1,
            0.3,
            1.0,
            3.0,
            10.0,
            30.0,
            60.0,
            300.0,
            900.0,
        ]
    )
    local = np.r_[available[available < plateau], float(plateau)]
    b.t = np.concatenate([i * plateau + local for i in range(4)])
    return b, len(local) - 1


def oracle(job):
    """Integrate continuous vapor and collision dynamics at one tolerance."""
    f, m, d, k, proc, tol, method, a = job
    b, frames = box(f, m, d, a.slots, a.plateau)
    name = (
        f"f{f}-m{m}-n{a.slots}-d{d}-dilute-{proc}-zero-r{k}"
        f"-full-t{tol:.0e}-{method}-f{frames}-l4.npz"
    )
    path = a.out / name
    if path.exists():
        with np.load(path) as saved:
            np.testing.assert_array_equal(b.t, saved["time"])
        return
    tic = perf_counter()
    obs, stats = reference(
        b, k, tol, True, "brownian" if proc == "coupled" else None, method
    )
    np.savez_compressed(
        path,
        obs=obs,
        time=b.t,
        names=observable_names(b.spec),
        seconds=perf_counter() - tic,
        **stats,
    )


def reference_grid(a):
    """Generate independent numerical reference paths by regime."""
    jobs = [
        (*v, a)
        for v in product(
            a.families, a.mix, a.densities, a.seeds, a.process, a.tol, a.method
        )
    ]
    with ProcessPoolExecutor(a.workers) as pool:
        for i, _ in enumerate(pool.map(oracle, jobs, chunksize=1), 1):
            if i % 16 == 0:
                print(i, len(jobs), flush=True)


def roll(sim, proc, policy, scheme, steps, splits, compiled=False):
    """Use the normal process driver or exact short-step replacement."""
    if policy == "none":
        obs, drift, _ = sim.rollout(scheme, steps, splits, "joint", True)
        return obs, drift, 0, 0
    replacement = correct
    if compiled:
        from functools import partial

        from stage2_compiled_rates import Factory

        if proc != "coupled":
            raise ValueError(
                "Compiled corrections require neutral coupled physics"
            )
        replacement = partial(correct, factory=Factory(None))
    result = hybrid(
        sim, proc, policy, scheme, steps, splits, 0.02, replacement=replacement
    )
    return (
        result["obs"],
        result["max_drift"],
        result["physical_advances"].sum(),
        result["physical_rhs_calls"],
    )


@torch.inference_mode()
def forecasts(a):
    """Retain all forcing jumps, concentration fields and event-stream IDs."""
    keys = np.asarray(list(product(a.families, a.mix, a.densities, a.seeds)))
    for proc, transfer, scheme, steps, splits, policy in product(
        a.process, a.transfer, a.scheme, a.vapor_steps, a.split_steps, a.policy
    ):
        kernel = a.kernel if proc == "coupled" else "off"
        name = (
            f"{proc}-{Path(transfer).stem}-{Path(kernel).stem}"
            f"-{scheme}-m{splits}-v{steps}-{policy}.npz"
        )
        path = a.out / name
        if path.exists():
            with np.load(path) as z:
                np.testing.assert_array_equal(keys, z["keys"])
                initial, _ = box(*keys[0, :3], a.slots, a.plateau)
                np.testing.assert_array_equal(initial.t, z["t"])
            continue
        panels, seconds = [], []
        drift = calls = physical = 0
        for begin in range(0, len(keys), a.batch):
            part = keys[begin : begin + a.batch]
            b, frames = box(*part[0, :3], a.slots, a.plateau)
            boxes = [box(*v[:3], a.slots, a.plateau)[0] for v in part]
            tic = perf_counter()
            sim = Split(boxes, part[:, 3], transfer, kernel)
            obs, error, adv, work = roll(
                sim, proc, policy, scheme, steps, splits, a.compiled_correction
            )
            panels.append(obs)
            drift = max(drift, error)
            calls += work
            physical += adv
            seconds.append(perf_counter() - tic)
        meta = dict(
            process="dilute-" + proc,
            charge="zero",
            slots=a.slots,
            frames=frames,
            legs=4,
            representation="full",
            scheme=scheme,
            vapor_steps=steps,
            split_steps=splits,
            species_order="joint",
            transfer=Path(transfer).stem,
            kernel=Path(kernel).stem,
            device="cpu",
            batch=a.batch,
            policy=policy,
            plateau_seconds=a.plateau,
            compiled_correction=a.compiled_correction,
            physical_advances=int(physical),
            physical_rhs=int(calls),
        )
        np.savez_compressed(
            path,
            obs=np.concatenate(panels),
            keys=keys,
            t=b.t,
            names=observable_names(b.spec),
            seconds=seconds,
            max_drift=drift,
            meta=json.dumps(meta),
        )
        print(name, len(keys), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["reference", "forecast"])
    p.add_argument("out", type=Path)
    for k, v in (
        ("families", [101, 21001]),
        ("mix", [0, 1]),
        ("densities", [10**9, 10**10, 10**11]),
        ("seeds", [301]),
        ("vapor-steps", [4]),
        ("split-steps", [1]),
    ):
        p.add_argument("--" + k, nargs="+", type=int, default=v)
    for k, v in (
        ("slots", 16),
        ("plateau", 900),
        ("batch", 128),
        ("workers", 3),
    ):
        p.add_argument("--" + k, type=int, default=v)
    p.add_argument("--kernel", default="brownian")
    p.add_argument("--compiled-correction", action="store_true")
    for k, v in (
        ("process", ["coupled"]),
        ("scheme", ["strang"]),
        ("transfer", ["radau-3-4"]),
        ("policy", ["none"]),
        ("method", ["LSODA"]),
    ):
        p.add_argument("--" + k, nargs="+", default=v)
    p.add_argument("--tol", nargs="+", type=float, default=[1e-7, 1e-9])
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1)
    (reference_grid if a.mode == "reference" else forecasts)(a)
