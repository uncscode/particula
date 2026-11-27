"""Interleaved CUDA solvers on the exact populations timed by native Warp."""

import argparse
import json
import time
from datetime import datetime, timezone
from itertools import chain, count, product
from pathlib import Path

import numpy as np
import torch
from stage3_adaptive_box import advance
from stage3_midpoint import Midpoint
from stage3_one_box import Box


def jobs(million_only=False):
    """Cover the native overlap and refinement options before adding seeds."""
    for rep in count():
        regular = product(
            [256, 4096, 16384, 65536], [2, 32], ["neutral", "bipolar"]
        )
        cases = chain(
            regular, [(n, 32, "neutral") for n in (131072, 262144, 1000000)]
        )
        if million_only:
            cases = [(1000000, 32, "neutral")]
        trials = [
            ("midpoint", "physical", 4),
            ("midpoint", "physical", 1),
            ("first-order", "physical", 1),
            ("midpoint", "learned", 1),
            ("midpoint", "off", 1),
            ("midpoint", "physical", 0),
        ]
        if million_only:
            np.random.default_rng(311001 + rep).shuffle(trials)
        for n, s, z in cases:
            for label, kernel, fixed in trials:
                yield n, s, z, label, kernel, fixed, 311001 + rep


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument(
        "--until",
        type=lambda s: datetime.fromisoformat(s).timestamp(),
        required=True,
    )
    p.add_argument("--million-only", action="store_true")
    a = p.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for i, (n, s, z, method, k, fixed, seed) in enumerate(jobs(a.million_only)):
        if time.time() >= a.until:
            break
        dest = a.folder / f"match-{n}-{s}-{z}-{method}-{k}-{fixed}-{seed}.json"
        if dest.exists():
            continue
        start = datetime.now(timezone.utc).isoformat()
        try:
            row = advance(
                n,
                s,
                1e12,
                z,
                307,
                seed,
                k,
                1.0,
                0.075,
                0.05,
                0.01,
                fixed=fixed,
                box_class=Midpoint if method == "midpoint" else Box,
            )
            row["status"] = "completed"
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            row = dict(
                n=n,
                species=s,
                density=1e12,
                charge=z,
                family=307,
                seed=seed,
                kernel=k,
                horizon=1.0,
                steps=fixed,
                vmax=0.01,
                collision_budget=0.075,
                growth_budget=0.05,
                status="failed",
                why=str(exc),
            )
        row.update(
            start=start,
            end=datetime.now(timezone.utc).isoformat(),
            vapor_method=method,
            panel="native-overlap",
        )
        tmp = dest.with_suffix(".part")
        tmp.write_text(json.dumps(row) + "\n")
        tmp.replace(dest)
        if i % 96 == 0:
            print(i, dest.name, row["status"], flush=True)
