"""Distinguish event-seed variability from collision-clock effects in shear."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from kernel_transfer import build_solver, check, fixture, phase_fields
from stage3_evidence import write


def run(folder):
    """Compare two independent native draws and a finer collision clock."""
    import warp as wp

    if not wp.is_cuda_available() or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    f = fixture(4096, "Shear")
    rows = []
    for steps, micro in [(1, 4), (2, 2)]:
        solvers = {
            m: build_solver(f, "Shear", m, {}, steps, micro)
            for m in ("native", "sampled physical")
        }
        for seed in range(78100, 78116):
            reference = solvers["native"].run([f], [seed])[0]
            check(reference, f)
            for label, solver, event_seed in [
                ("independent native", solvers["native"], seed + 5000),
                ("sampled physical", solvers["sampled physical"], seed),
            ]:
                obs = solver.run([f], [event_seed])[0]
                _, counts = check(obs, f)
                ref = phase_fields(reference, f.spec.species)
                diff = phase_fields(obs, f.spec.species) - ref
                rows.append(
                    dict(
                        collision_steps=4 * steps,
                        vapor_substeps_per_half=micro,
                        reference_seed=seed,
                        method=label,
                        seed=event_seed,
                        mergers=4096 - counts[-1],
                        ref_mergers=4096 - round(reference[-1, 0] * f.volume),
                        count_error_pct=100
                        * np.max(np.abs(obs[:, 0] / reference[:, 0] - 1)),
                        phase_error_pct=100
                        * np.max(np.abs(diff) / np.maximum(ref, 1e-11)),
                    )
                )
    write(folder / "shear-clock.csv", rows)


if __name__ == "__main__":
    if extra_site := os.getenv("EMBEDDING_WARP_SITE"):
        sys.path.append(extra_site)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "results/kernel-transfer",
    )
    folder = parser.parse_args().folder
    folder.mkdir(parents=True, exist_ok=True)
    run(folder)
