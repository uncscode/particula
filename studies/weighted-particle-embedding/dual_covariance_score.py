"""Audit matched volatile marginals and qualify joint water-organic records."""

import argparse
from pathlib import Path

import numpy as np
import torch
from dual_experiment import continuous, load_files, reporting
from dual_joint_record import reconstruct, record
from dual_schedule_score import models
from node_transfer import write_table


def panels(root, split, mode, families, bdf):
    """Use memberwise refinement bounds for native and continuous references."""
    base = root / split / mode
    fine, coarse = [
        load_files(base / f"s{s}-k12-t16", families) for s in (128, 64)
    ]
    for k in ("ids", "m", "n", "time", "levels", "reset_mask"):
        np.testing.assert_array_equal(fine[k], coarse[k])
    diff = abs(fine["bulk"] - coarse["bulk"])
    targets = [("native", fine["bulk"], diff[::2] + diff[1::2])]
    if bdf:
        native = load_files(base / "s32-k12-t16", families)
        for k in ("ids", "m", "n", "time", "levels"):
            np.testing.assert_array_equal(fine[k], native[k])
        high, low = [
            continuous(root / "bdf" / mode, native["sources"], tol, "coupled")[
                : len(fine["m"])
            ]
            for tol in (1e-9, 1e-7)
        ]
        diff = abs(high - low)
        targets.append(("BDF", high, diff[::2] + diff[1::2]))
    return fine, targets


def archive(file, d, k, detail, construction):
    """Verify mathematical equalities for the targeted missing joint moments."""
    if k:
        payload = record(d["m"], d["n"], k, detail)
        for key in ("shape", "totals", "organic", "spread"):
            a = payload[key]
            np.testing.assert_allclose(
                a[::2], a[1::2], rtol=1e-7, atol=1e-13 if key == "shape" else 0
            )
        cross = payload["cross"]
        if (
            construction != "uniform"
            and detail == "global"
            or construction == "composition"
            and detail == "size"
        ):
            np.testing.assert_allclose(cross, 0, rtol=0, atol=1e-10)
    else:
        payload = dict(m=d["m"], n=d["n"])
    pending = file.with_suffix(".tmp.npz")
    np.savez_compressed(
        pending,
        **payload,
        ids=d["ids"],
        time=d["time"],
        levels=d["levels"][::2],
        volume=d["volume"],
    )
    pending.replace(file)
    with np.load(file) as a:
        m, n, clip = reconstruct(a) if k else (a["m"], a["n"], 0.0)
        np.testing.assert_array_equal(a["ids"], d["ids"])
        return (
            m,
            n / float(a["volume"]),
            a["levels"].repeat(2, 0),
            a["time"],
            clip,
        )


@torch.inference_mode()
def run(
    root,
    out,
    split,
    constructions,
    families,
    nodes,
    details,
    substeps,
    checkpoints,
    stages,
    bdf,
):
    """Compare degrees of joint information in free forecasts."""
    rows = []
    for mode in constructions:
        d, refs = panels(root, split, mode, families, bdf)
        folder = root / "archives" / split / mode
        folder.mkdir(parents=True, exist_ok=True)
        for k in nodes:
            for detail in details if k else ["full"]:
                file = folder / f"f{families}-n{k}-{detail}.npz"
                mass, num, levels, t, clip = archive(file, d, k, detail, mode)
                m, n, forcing = map(torch.from_numpy, (mass, num, levels))
                for name, net, size, seed, sha, _ in models(
                    checkpoints, False, stages
                ):
                    for s in substeps:
                        w, g = net.rollout(m, n, forcing, t, s)
                        wet = (w * n[:, None, :, None]).sum(2).numpy()
                        for ref, target, bound in refs:
                            before = len(rows)
                            reporting(
                                name,
                                size,
                                wet,
                                g.numpy(),
                                d,
                                target,
                                bound,
                                ref,
                                file,
                                mode,
                                s,
                                seed,
                                rows,
                            )
                            for r in rows[before:]:
                                r.update(
                                    nodes=k,
                                    joint_record=detail,
                                    maximum_correlation_clipping=clip,
                                    checkpoint_sha256=sha,
                                )
            write_table(rows, out)
            print(mode, k, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--split",
        choices=["validation", "test", "challenge"],
        default="validation",
    )
    p.add_argument(
        "--constructions", nargs="+", default=["uniform", "size", "composition"]
    )
    p.add_argument("--families", type=int, default=64)
    p.add_argument("--nodes", nargs="+", type=int, default=[0, 3, 5, 8])
    p.add_argument(
        "--details", nargs="+", default=["marginal", "global", "size", "joint"]
    )
    p.add_argument("--substeps", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--checkpoints", type=Path, nargs="*", default=[])
    p.add_argument("--stages", nargs="*", default=["3x2"])
    p.add_argument("--bdf", action="store_true")
    torch.set_num_threads(1)
    run(**vars(p.parse_args()))
