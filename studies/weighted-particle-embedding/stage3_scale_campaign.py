"""Run the one-box ladder and independent accuracy ensembles."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from itertools import count, product
from pathlib import Path

import torch
from stage3_one_box import simulate


def jobs():
    """Yield the size ladder before adding independent event seeds."""
    sizes = [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1000000]
    # First traverse the entire requested size range with physical collisions.
    for n in sizes:
        for species in [2, 32]:
            yield (
                "capacity",
                n,
                species,
                1e12,
                "neutral",
                101,
                212001,
                "physical",
                1.0,
                1,
                0.02,
            )
    # Interleave methods on identical one-box inputs.
    for rep in range(3):
        for n, species in product(sizes, [1, 2, 8, 16, 32]):
            for kernel in ["physical", "learned", "off"]:
                yield (
                    "scaling",
                    n,
                    species,
                    1e12,
                    "neutral",
                    101,
                    212101 + rep,
                    kernel,
                    1.0,
                    1,
                    0.02,
                )
    for rep in range(3):
        for n, species in product([1024, 65536, 1000000], [2, 32]):
            for z, k in product(
                ["like", "bipolar", "segregated"],
                ["physical", "learned", "off"],
            ):
                yield (
                    "charged-scaling",
                    n,
                    species,
                    1e12,
                    z,
                    101,
                    213001 + rep,
                    k,
                    1.0,
                    1,
                    0.02,
                )
    # Higher densities span cases where omission changes the result. Varying the
    # refinement changes the random matching, so aggregate by independent seed.
    yield from _crossover()


def _crossover():
    for rep in count():
        for n, s, z, density in product(
            [4096, 65536],
            [2, 32],
            ["neutral", "like", "bipolar", "segregated"],
            [1e11, 1e12, 1e13, 1e14],
        ):
            family = [102, 205, 308, 409][rep % 4]
            for k, steps, vmax in [
                ("physical", 4, 0.003),
                ("physical", 2, 0.012),
                ("physical", 1, 0.04),
                ("learned", 2, 0.012),
                ("off", 2, 0.012),
            ]:
                yield (
                    "crossover",
                    n,
                    s,
                    density,
                    z,
                    family,
                    214001 + rep,
                    k,
                    1.0,
                    steps,
                    vmax,
                )
        if rep % 4 == 0:
            for s, z, steps, k in product(
                [2, 32],
                ["neutral", "bipolar"],
                [1, 4],
                ["physical", "learned", "off"],
            ):
                yield (
                    "largest-refinement",
                    1000000,
                    s,
                    1e13,
                    z,
                    308,
                    215001 + rep,
                    k,
                    1.0,
                    steps,
                    0.008,
                )


def run(folder, until, limit):
    """Run the one-box benchmark and retain incomplete configurations."""
    folder.mkdir(exist_ok=True, parents=True)
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    torch.set_num_threads(1)
    torch.empty(1, device=device).sum().item()
    env = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        utc_start=datetime.now(timezone.utc).isoformat(),
        deadline=until,
        gpu_free_total=list(torch.cuda.mem_get_info()),
        pid=os.getpid(),
    )
    (folder / "device.json").write_text(json.dumps(env, indent=2) + "\n")
    for ix, job in enumerate(jobs()):
        if time.time() >= until or (limit and ix >= limit):
            break
        panel, n, s, d, z, f, seed, k, h, steps, vmax = job
        out = (
            folder
            / f"{panel}-{n}-{s}-{d:g}-{z}-{f}-{seed}-{k}-{steps}-{vmax:g}.json"
        )
        if out.exists():
            continue
        start = datetime.now(timezone.utc).isoformat()
        try:
            row = simulate(n, s, d, z, device, f, seed, k, h, steps, vmax)
            row["status"] = "completed"
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            row = dict(
                n=n,
                species=s,
                density=d,
                charge=z,
                family=f,
                seed=seed,
                kernel=k,
                steps=steps,
                vmax=vmax,
                horizon=h,
                status="failed",
                why=str(exc),
            )
            torch.cuda.empty_cache()
        row.update(
            panel=panel, start=start, end=datetime.now(timezone.utc).isoformat()
        )
        tmp = out.with_suffix(".part")
        tmp.write_text(json.dumps(row) + "\n")
        tmp.replace(out)
        if ix % 25 == 0 or row["status"] == "failed":
            print(out.name, row["status"], flush=True)
    (folder / "closed.json").write_text(
        json.dumps(dict(end=datetime.now(timezone.utc).isoformat())) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument(
        "--until",
        required=True,
        type=lambda s: datetime.fromisoformat(s).timestamp(),
    )
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    run(a.folder, a.until, a.limit)
