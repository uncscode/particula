"""Run independent aging histories and retain per-snapshot comparisons."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from compression import evaluate
from particula.optics import get_mie_efficiencies
from physics import evolve, make_optics, responses

# Shared by forked workers after the interpolation check.
OPTICS = None
SEEDS = (181, 302, 463, 594, 715, 826, 947, 1068, 1199, 1320, 1441, 1562)


def check_lookup(optics):
    """Check interpolated cross sections against direct Mie."""
    rng = np.random.default_rng(84)
    diameter = np.exp(rng.uniform(np.log(18), np.log(900), 1200))
    bc = rng.uniform(0.0001, 0.98, len(diameter))
    vols = diameter[:, None] ** 3 * np.column_stack(
        [1 - bc, 0 * bc, 0 * bc, bc]
    )
    _, area = responses(vols, optics)
    direct = (
        np.pi
        / 4
        * diameter**2
        * get_mie_efficiencies(
            diameter * 5e-10, 550e-9, 1.5 + (0.45 + 0.79j) * bc
        ).absorption
    )
    relative = abs(area / direct - 1)
    if relative.max() >= 0.02:
        raise ValueError("Mie interpolation exceeded 2% relative error")
    return float(relative.max())


def history(job):
    """Resolve both parcels of one independent source and aging history."""
    seed, n, substeps, limit, protect_bc, out = job
    rows = []
    for checkpoint in evolve(seed, n, substeps):
        _, state, rate, exchange, stock = checkpoint
        for parcel in state:
            rows.append(evaluate(parcel, OPTICS, limit, protect_bc))
    keys = [
        "scores",
        "chi",
        "counts",
        "ccn",
        "babs",
        "rep_chi",
        "rep_ccn",
        "bases",
        "bulk_entropy",
    ]
    arrays = {
        name: np.stack(values)
        for name, values in zip(keys, zip(*rows, strict=True), strict=True)
    }
    np.savez_compressed(
        out / f"{seed}.npz",
        **arrays,
        rate=rate,
        exchange=exchange,
        stock=stock,
        particles=n,
        substeps=substeps,
        protect_bc=protect_bc,
    )
    print(f"Finished seed {seed}, {n} particles/parcel", flush=True)


def main():
    """Save resolved and compressed responses for the declared histories."""
    import multiprocessing as mp

    global OPTICS
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("output", type=Path)
    cmd.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    cmd.add_argument("--particles", type=int, default=100000)
    cmd.add_argument("--substeps", type=int, default=2)
    cmd.add_argument("--limit", type=int, default=64)
    cmd.add_argument(
        "--generic", action="store_true", help="omit the BC carrier split"
    )
    cmd.add_argument("--workers", type=int, default=3)
    args = cmd.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    OPTICS = make_optics()
    print(
        "Maximum Mie lookup relative error:", check_lookup(OPTICS), flush=True
    )
    jobs = [
        (
            seed,
            args.particles,
            args.substeps,
            args.limit,
            not args.generic,
            args.output,
        )
        for seed in args.seeds
    ]
    with ProcessPoolExecutor(
        args.workers, mp_context=mp.get_context("fork")
    ) as pool:
        list(pool.map(history, jobs))


if __name__ == "__main__":
    main()
