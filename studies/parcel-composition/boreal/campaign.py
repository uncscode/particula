"""Reference trajectories and replayable state boundaries for the challenge."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from model import REGIMES, SEEDS, Box, init, step


def inventory(boxes):
    """Sum condensed and gas inventory across both equal-volume boxes."""
    return sum(b.dry.sum(0) + b.gas for b in boxes)


def audit(boxes, start, n, source):
    """Verify species, particle number, new lineages and finite state."""
    np.testing.assert_allclose(inventory(boxes), start + source, rtol=2e-10)
    np.testing.assert_equal(
        sum(len(b.dry) for b in boxes),
        n * 2 + sum(b.born - b.mergers for b in boxes),
    )
    np.testing.assert_equal(
        sum(b.born for b in boxes),
        sum(b.reached + b.captured + int(b.pending.sum()) for b in boxes),
    )
    for b in boxes:
        if (
            not np.isfinite(b.dry).all()
            or min(b.dry.min(), b.gas.min(), b.water.min()) < 0
        ):
            raise ValueError("Invalid state")


def write(path, boxes, rng, t):
    """Atomically persist both parcels and the generator state."""
    arrays = {}
    for i, b in enumerate(boxes):
        arrays.update({f"p{i}_{k}": v for k, v in vars(b).items()})
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, t=t, rng=json.dumps(rng.bit_generator.state), **arrays)
    os.replace(tmp, path)


def read(path):
    """Restore a full native-resolution checkpoint."""
    with np.load(path) as s:
        boxes = []
        for i in range(2):
            fields = {k[3:]: s[k] for k in s.files if k.startswith(f"p{i}_")}
            boxes.append(
                Box(
                    **{
                        k: v.item() if not v.ndim else v
                        for k, v in fields.items()
                    }
                )
            )
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(s["rng"].item())
        return boxes, rng, int(s["t"])


def reference(job):
    """Advance one six-hour realization and save its event history."""
    root, seed, regime, n, dt, extra = job
    folder = root / str(seed) / regime
    folder.mkdir(parents=True, exist_ok=True)
    boxes, rng, params = init(seed, n)
    if extra:
        rng = np.random.default_rng(np.random.SeedSequence([seed, extra]))
    start = inventory(boxes)
    source = np.zeros(8)
    best_l = best_j = -1
    write(folder / "00000.npz", boxes, rng, 0)
    rows = []
    for t in range(0, 21600, dt):
        source += step(boxes, rng, params, regime, t, dt)
        end = t + dt
        if end % 3600 == 0 or end in (1800, 5400, 9000, 14400):
            audit(boxes, start, n, source)
            write(folder / f"{end:05d}.npz", boxes, rng, end)
        lw = max(b.liquid for b in boxes)
        j = sum(b.jrate for b in boxes)
        if REGIMES[regime][0] and lw > best_l and 9000 <= end <= 14400:
            best_l = lw
            write(folder / "cloud-peak.npz", boxes, rng, end)
        if j > 0 and j > best_j:
            best_j = j
            write(folder / "burst-peak.npz", boxes, rng, end)
        rows.append(
            [
                end,
                *[
                    v
                    for b in boxes
                    for v in (
                        len(b.dry),
                        b.born,
                        b.reached,
                        b.captured,
                        b.droplets,
                        b.liquid,
                        b.jrate,
                    )
                ],
            ]
        )
    np.savetxt(
        folder / "trajectory.csv",
        rows,
        delimiter=",",
        header="seconds,"
        + ",".join(
            f"{v}_{i}"
            for i in "AB"
            for v in (
                "N",
                "births",
                "reached",
                "lost",
                "drops_cm3",
                "water_g_m3",
                "J",
            )
        ),
        comments="",
    )
    (folder / "done").write_text(f"{n} {dt} {extra}\n")
    print(seed, regime, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--regimes", nargs="+", default=list(REGIMES))
    p.add_argument("--dt", type=int, default=300)
    p.add_argument("--particles", type=int, default=100000)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--workers", type=int, default=5)
    args = p.parse_args()
    if args.dt <= 0 or 1800 % args.dt or args.particles <= 0:
        p.error("Use positive counts and a step dividing 1800 seconds")
    signature = f"{args.particles} {args.dt} {args.repeat}"
    for seed in args.seeds:
        for reg in args.regimes:
            marker = args.directory / str(seed) / reg / "done"
            if marker.exists() and marker.read_text().strip() != signature:
                p.error(f"Existing trajectory has different settings: {marker}")
    jobs = [
        (args.directory, s, r, args.particles, args.dt, args.repeat)
        for s in args.seeds
        for r in args.regimes
        if not (args.directory / str(s) / r / "done").exists()
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(reference, jobs))
