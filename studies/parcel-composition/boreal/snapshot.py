"""Stream the cap, grouping, and bin-width sweep one checkpoint at a time."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from campaign import read
from metrics import EDGES, budget, make_tree, prepare, profile, reference
from model import REGIMES, SEEDS

from optics import Optics


def snapshots(job):
    """Score a history once per distinct saved boundary."""
    folder, out, optpath, grids, rules, limit = job
    out.mkdir(exist_ok=True, parents=True)
    opt = Optics(optpath)
    rows = []
    checkpoint = []
    shape = []
    visited = set()
    target = out / "snapshot.npz"
    if target.exists():
        with np.load(target) as z:
            if (
                list(z["grids"]) != list(grids)
                or list(z["rules"]) != list(rules)
                or z["errors"].shape[2] != limit
            ):
                raise ValueError(
                    "Output directory already contains a different sweep"
                )
            rows.extend(z["errors"])
            checkpoint.extend(z["checkpoints"])
            shape.extend(z["profiles"])
            visited.update(int(r[0]) for r in checkpoint)
        paths = folder.glob("cloud-*.npz")
    else:
        paths = folder.glob("*.npz")
    for path in sorted(paths):
        if ".tmp." in path.name:
            continue
        boxes, _, t = read(path)
        if t in visited:
            continue
        visited.add(t)
        for i, b in enumerate(boxes):
            ref = reference(b, opt)
            checkpoint.append([t, i, *ref[1].sum(0), len(b.dry)])
            case = []
            for grid in grids:
                prep = prepare(b, grid)
                chi, count, absorption = profile(b, prep, ref[1])
                shape += [
                    [t, i, grid, j, n, x, y]
                    for j, (n, x, y) in enumerate(
                        zip(count, chi, absorption, strict=True)
                    )
                ]
                for rule in rules:
                    means, meta, _ = make_tree(prep, limit, rule)
                    case.append(budget(ref, means, meta, opt, limit))
            rows.append(case)
    tmp = out / "snapshot.tmp.npz"
    np.savez_compressed(
        tmp,
        errors=np.array(rows, dtype=np.float32),
        checkpoints=checkpoint,
        profiles=shape,
        grids=grids,
        rules=rules,
    )
    os.replace(tmp, target)
    print(folder.parent.name, folder.name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    p.add_argument("--grids", nargs="+", type=int, default=list(EDGES))
    p.add_argument("--rules", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--regimes", nargs="+", default=list(REGIMES))
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--update-events", action="store_true")
    a = p.parse_args()
    jobs = [
        (
            a.base / str(s) / r,
            a.out / str(s) / r,
            a.optics,
            a.grids,
            a.rules,
            a.limit,
        )
        for s in a.seeds
        for r in a.regimes
        if a.update_events or not (a.out / str(s) / r / "snapshot.npz").exists()
    ]
    if any(not (j[0] / "done").is_file() for j in jobs):
        raise ValueError("Reference trajectories incomplete")
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(snapshots, jobs))
