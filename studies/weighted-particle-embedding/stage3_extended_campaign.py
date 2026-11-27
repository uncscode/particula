"""Test longer GPU forecasts with refined collision clocks."""

import argparse
import json
import time
from datetime import datetime, timezone
from itertools import count, product
from pathlib import Path

import torch
from stage3_one_box import simulate


def config():
    # Only vary single-box particle count via its volume, and time via horizon.
    # Include both small and million-particle boxes in the first pass.
    """Yield held-out configurations in the registered sampling order."""
    for rep in count():
        family = (102, 205, 308, 409)[rep % 4]
        seed = 241001 + rep
        for n, ns, d, z in product(
            [65536, 1000000],
            [2, 32],
            [1e12, 1e13],
            ["neutral", "bipolar", "segregated"],
        ):
            for k, steps, vmax in [
                ("physical", 16, 0.002),
                ("physical", 4, 0.008),
                ("learned", 4, 0.008),
                ("off", 4, 0.008),
            ]:
                yield (
                    "wide-confirm",
                    n,
                    ns,
                    d,
                    z,
                    family,
                    seed,
                    k,
                    1.0,
                    steps,
                    vmax,
                )
        for n, ns, d, z in product(
            [4096, 65536],
            [2, 32],
            [3e12, 7e12, 1e13, 3e13, 1e14],
            ["neutral", "like", "bipolar", "segregated"],
        ):
            kref = 32 if z in ("bipolar", "segregated") and d >= 3e13 else 16
            for k, steps, vmax in [
                ("physical", kref, 0.002),
                ("physical", kref // 2, 0.006),
                ("learned", kref // 2, 0.006),
                ("off", kref // 2, 0.006),
            ]:
                yield "boundary", n, ns, d, z, family, seed, k, 1.0, steps, vmax
        for z, h in product(
            ["neutral", "bipolar", "segregated"], [0.1, 1.0, 10.0, 60.0]
        ):
            d = 1e12
            ref = max(16, int(h * 4)) if z != "neutral" else max(8, int(h))
            for k, ss in [
                ("physical", ref),
                ("physical", ref // 2),
                ("learned", ref // 2),
                ("off", ref // 2),
            ]:
                yield (
                    "long-dose",
                    16384,
                    8,
                    d,
                    z,
                    family,
                    seed,
                    k,
                    h,
                    ss,
                    0.004 if ss == ref else 0.012,
                )
        for n, s, d, z in product(
            [16, 64], [2, 32], [1e11, 1e12, 1e13, 1e14], ["neutral", "bipolar"]
        ):
            ref = 32 if d >= 1e14 and z != "neutral" else 8
            for k, ss in [
                ("physical", ref),
                ("learned", ref // 2),
                ("off", ref // 2),
            ]:
                yield (
                    "small-direct",
                    n,
                    s,
                    d,
                    z,
                    family,
                    seed,
                    k,
                    1.0,
                    ss,
                    0.004,
                )


def run(folder, until, seed_limit):
    """Measure each pending configuration until the experiment cutoff."""
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.empty(1, device="cuda").sum().item()
    for ix, job in enumerate(config()):
        panel, n, ns, d, z, f, seed, k, h, steps, vmax = job
        if time.time() >= until or (seed_limit and seed >= 241001 + seed_limit):
            break
        path = folder / (
            f"{panel}-{n}-{ns}-{d:g}-{z}-{f}-{seed}-{k}-"
            f"{h:g}-{steps}-{vmax:g}.json"
        )
        if path.exists():
            continue
        utc = datetime.now(timezone.utc).isoformat()
        try:
            row = simulate(n, ns, d, z, "cuda", f, seed, k, h, steps, vmax)
            row["status"] = "completed"
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            row = dict(
                n=n,
                species=ns,
                density=d,
                charge=z,
                family=f,
                seed=seed,
                kernel=k,
                horizon=h,
                steps=steps,
                vmax=vmax,
                status="failed",
                why=str(exc),
            )
            torch.cuda.empty_cache()
        row.update(
            panel=panel, start=utc, end=datetime.now(timezone.utc).isoformat()
        )
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(row) + "\n")
        tmp.replace(path)
        if ix % 64 == 0:
            print(ix, path.name, row["status"], flush=True)
    (folder / "closed.json").write_text(
        json.dumps(dict(end=datetime.now(timezone.utc).isoformat())) + "\n"
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
    run(a.folder, a.until, a.seed_limit)
