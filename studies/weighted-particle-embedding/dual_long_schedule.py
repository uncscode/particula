"""Forecast long continuations of the twelve-plateau vapor schedules."""

import argparse
from pathlib import Path

import numpy as np
import torch
from dual_experiment import continuous, load_files
from dual_hybrid import guarded_rollout, heads
from dual_schedule_score import append_scores, decoded, references
from dual_stages import DualStages
from node_transfer import write_table


def prefix_end(time, plateaus):
    """End just before the next prescribed reservoir reset."""
    starts = np.r_[0, np.flatnonzero(np.diff(time) == 0) + 1]
    if not 1 <= plateaus <= len(starts):
        raise ValueError("Requested plateau count outside trajectory")
    return starts[plateaus] if plateaus < len(starts) else len(time)


def confirm(root, split, geom, families, blocks):
    """Require the old integration and forcing to agree on the whole prefix."""
    folder = root / split / geom
    for n in (64, 128):
        full = load_files(folder / f"b{blocks}-s{n}-k16", families)
        old = load_files(folder / f"b3-s{n}-k16", families)
        ix = prefix_end(full["time"], 12)
        for k in ("ids", "n", "m"):
            np.testing.assert_array_equal(full[k], old[k])
        np.testing.assert_array_equal(full["time"][:ix], old["time"])
        for k in ("levels", "reset_mask"):
            np.testing.assert_array_equal(full[k][:, :12], old[k])
        for k in ("bulk", "gas"):
            np.testing.assert_array_equal(full[k][:, :ix], old[k])


@torch.inference_mode()
def run(
    root, out, paths, split, families, geometries, blocks, guards, bdf=False
):
    """Score each species, reset stratum and forecast prefix."""
    rows = []
    physical = DualStages(3, 3)
    for geom in geometries:
        confirm(root, split, geom, families, blocks)
        data, refs = references(root, split, geom, blocks, families, False)
        if bdf:
            source = load_files(
                root / split / geom / f"b{blocks}-s64-k16", families
            )
            target = root / "long-bdf" / split / geom / f"b{blocks}-f{families}"
            for stem, _ in source["sources"]:
                with np.load(target / f"{stem}-r1e-09-coupled.npz") as a:
                    np.testing.assert_array_equal(a["time"], data["time"])
            fine, coarse = [
                continuous(target, source["sources"], tol, "coupled")
                for tol in (1e-9, 1e-7)
            ]
            if fine.shape != data["bulk"].shape or coarse.shape != fine.shape:
                raise ValueError("Mismatched long-run continuous reference")
            shift = abs(fine - coarse)
            refs.append(("BDF", fine, shift[::2] + shift[1::2]))
        folder = root / "long-records" / split / geom
        folder.mkdir(parents=True, exist_ok=True)
        rec = folder / f"b{blocks}-f{families}-n5.npz"
        mass, num, supply, mask = decoded(rec, data, 5, True)
        for name, net, size, seed, sha in heads(paths):
            for guard in guards:
                if guard == 20 and sha:
                    continue
                wet, gas, expensive, cheap = guarded_rollout(
                    net,
                    physical,
                    mass,
                    num,
                    supply,
                    mask,
                    data["time"],
                    guard,
                    4,
                    False,
                )
                for block in sorted(set([3, 6, 12, blocks])):
                    if block > blocks:
                        continue
                    p = 4 * block
                    stop = prefix_end(data["time"], p)
                    selected = dict(
                        data,
                        time=data["time"][:stop],
                        levels=data["levels"][:, :p],
                        reset_mask=data["reset_mask"][:, :p],
                    )
                    panels = [
                        (label, target[:, :stop], bound[:, :stop])
                        for label, target, bound in refs
                    ]
                    factors = dict(
                        stratified=True,
                        plateaus=p,
                        guard_frames=guard,
                        nodes=5,
                        checkpoint_sha256=sha,
                        full_run_implicit_frames=expensive,
                        full_run_learned_frames=cheap,
                        prefix_verified=True,
                    )
                    append_scores(
                        rows,
                        selected,
                        panels,
                        wet[:, :stop].numpy(),
                        gas[:, :stop].numpy(),
                        (name, size, seed, geom, 1),
                        rec,
                        factors,
                    )
            write_table(rows, out)
            print(geom, name, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("paths", type=Path, nargs="*")
    p.add_argument("--bdf", action="store_true")
    p.add_argument("--split", default="validation")
    p.add_argument("--families", type=int, default=64)
    p.add_argument(
        "--geometries", nargs="+", default=["conditioned", "proportional"]
    )
    p.add_argument("--blocks", type=int, default=24)
    p.add_argument("--guards", nargs="+", type=int, default=[0, 2, 4, 20])
    torch.set_num_threads(1)
    run(**vars(p.parse_args()))
