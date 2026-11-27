"""Generate independently seeded families of paired native condensation runs."""

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from pulse import Family, simulate


def draw(seed, challenge):
    """Generate one family from its permanent split-specific seed."""
    rng = np.random.default_rng(seed)
    return Family(
        radius_um=rng.uniform(0.42, 0.94),
        number_m3=10 ** rng.uniform(10.4, 11.6),
        soluble=rng.uniform(0.22, 0.78),
        contrast=rng.uniform(0.94, 1.0)
        if challenge
        else rng.uniform(0.35, 0.93),
        wet_rh=rng.uniform(0.95, 0.98)
        if challenge
        else rng.uniform(0.80, 0.945),
        dry_rh=rng.uniform(0.19, 0.28)
        if challenge
        else rng.uniform(0.29, 0.50),
        initial_water_ratio=rng.uniform(0.14, 0.34),
        log_width=rng.uniform(0.15, 0.38),
    )


def generate(root, split, families, batch, bins, steps, samples, first=0):
    """Write independent native trajectories and a checksummed manifest."""
    out = root / split
    out.mkdir(parents=True, exist_ok=True)
    offset = {
        "train": 100_000,
        "validation": 200_000,
        "test": 300_000,
        "challenge": 400_000,
    }[split]
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required to record the source revision")
    records = []
    for start in range(first, first + families, batch):
        ids = list(
            range(offset + start, offset + min(start + batch, first + families))
        )
        group = [draw(i, split == "challenge") for i in ids]
        path = (
            out / f"families-{ids[0]}-{ids[-1]}-b{bins}-s{steps}-t{samples}.npz"
        )
        if not path.exists():
            data = simulate(group, bins, steps, samples, "cuda")
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **data, ids=ids)
            tmp.replace(path)
        records.append(
            dict(
                file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                ids=ids,
                families=[asdict(f) for f in group],
            )
        )
    (
        out / f"manifest-{first}-{families}-b{bins}-s{steps}-t{samples}.json"
    ).write_text(
        json.dumps(
            dict(
                split=split,
                revision=subprocess.check_output(  # noqa: S603 -- fixed arguments
                    [git, "rev-parse", "HEAD"], text=True
                ).strip(),
                bins=bins,
                steps=steps,
                samples_per_leg=samples,
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument(
        "split", choices=["train", "validation", "test", "challenge"]
    )
    for key, default in (
        ("families", 32),
        ("batch", 16),
        ("bins", 8),
        ("steps", 16),
        ("samples", 16),
        ("first", 0),
    ):
        p.add_argument("--" + key, type=int, default=default)
    generate(**vars(p.parse_args()))
