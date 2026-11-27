"""Time a literal local restart, device integration and full-particle output."""

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stage3_one_box import Box


def save(folder, b):
    """Write individual particle arrays and stochastic restart state."""
    for name in ["mass", "z", "gas"]:
        np.save(
            folder / (name + ".npy"),
            getattr(b, name).cpu().numpy(),
            allow_pickle=False,
        )
    np.save(
        folder / "rng.npy",
        b.generator.get_state().cpu().numpy(),
        allow_pickle=False,
    )


def load(folder, b):
    """Transfer the saved arrays and event generator to the device."""
    for name in ["mass", "z", "gas"]:
        v = np.load(folder / (name + ".npy"), allow_pickle=False)
        setattr(b, name, torch.from_numpy(v).to(b.device))
    b.generator.set_state(
        torch.from_numpy(np.load(folder / "rng.npy", allow_pickle=False))
    )


def size(folder):
    """Measure allocated file lengths in bytes."""
    return sum(p.stat().st_size for p in folder.glob("*.npy"))


@torch.inference_mode()
def run(n, ns, density, charge, kernel, seed, workspace, steps, horizon):
    """Time a warm-file restart through a complete particle output."""
    with tempfile.TemporaryDirectory(dir=workspace) as td:
        a = Path(td) / "input"
        o = Path(td) / "output"
        a.mkdir()
        o.mkdir()
        b = Box(n, ns, density, charge, "cuda", 101, seed, kernel)
        save(a, b)
        inp = size(a)
        # Files are cached by the OS. Include a fresh read from each file and
        # host-device transfer; no cold-cache or durable fsync claim is made.
        torch.cuda.synchronize()
        tic = perf_counter()
        load(a, b)
        torch.cuda.synchronize()
        loaded = perf_counter()
        dt = horizon / (4 * steps)
        curve = [b.measure()]
        for _ in range(4):
            for _ in range(steps):
                b.vapor(dt / 2, 0.02)
                b.collide(dt)
                b.vapor(dt / 2, 0.02)
            curve.append(b.measure())
        torch.cuda.synchronize()
        advanced = perf_counter()
        save(o, b)
        torch.cuda.synchronize()
        finished = perf_counter()
        reload_mass = np.load(o / "mass.npy", mmap_mode="r", allow_pickle=False)
        if reload_mass.shape != (b.n0 - b.collisions, ns + 1):
            raise ArithmeticError("Restart shape mismatch")
        return dict(
            n=n,
            species=ns,
            density=density,
            charge=charge,
            kernel=kernel,
            seed=seed,
            horizon_s=horizon,
            steps=steps,
            input_bytes=inp,
            output_bytes=size(o),
            load_s=loaded - tic,
            advance_s=advanced - loaded,
            output_s=finished - advanced,
            warm_file_end_to_end_s=finished - tic,
            final_n=len(reload_mass),
            worst_ledger=max(r["ledger"] for r in curve),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--n", nargs="+", type=int, default=[1024, 65536, 1000000])
    p.add_argument("--s", nargs="+", type=int, default=[2, 32])
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    from itertools import product

    for n, s, z, k, rep in product(
        a.n,
        a.s,
        ["neutral", "segregated"],
        ["physical", "learned", "off"],
        range(a.reps),
    ):
        f = a.folder / f"restart-{n}-{s}-{z}-{k}-{rep}.json"
        if f.exists():
            continue
        result = run(n, s, 1e12, z, k, 251001 + rep, a.folder, 1, 1.0)
        f.write_text(json.dumps(result) + "\n")
