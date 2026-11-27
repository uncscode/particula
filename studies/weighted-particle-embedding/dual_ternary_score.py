"""Qualify alternative mixed-moment orders for ternary volatile loadings."""

import argparse
from pathlib import Path

import numpy as np
import torch
from dual_experiment import continuous, load_files, reporting
from dual_stages import DualStages
from dual_ternary_record import decode_joint, encode_joint
from node_transfer import write_table


def inputs(root, split, label, families, high, bdf):
    """Build independent native and stiff reference panels for the pairs."""
    folder = root / split / label
    d, c = [
        load_files(folder / f"s{s}-k8", families) for s in (high, high // 2)
    ]
    for k in ("ids", "m", "n", "time", "levels"):
        np.testing.assert_array_equal(d[k], c[k])
    diff = abs(d["bulk"] - c["bulk"])
    panels = [(f"native{high}", d["bulk"], diff[::2] + diff[1::2])]
    if bdf:
        src = load_files(folder / "s32-k8", families)
        for k in ("ids", "m", "n", "time", "levels"):
            np.testing.assert_array_equal(d[k], src[k])
        fine, coarse = [
            continuous(root / "bdf" / label, src["sources"], v, "coupled")[
                : len(d["m"])
            ]
            for v in (1e-9, 1e-7)
        ]
        diff = abs(fine - coarse)
        panels.append(("BDF", fine, diff[::2] + diff[1::2]))
    return d, panels


def restore(path, d, k, detail, mode, label):
    """Verify the moment equalities before opening the serialized projection."""
    if k:
        payload = encode_joint(d["m"], d["n"], k, detail, mode)
        moments = payload["mixed"]
        if (
            label.startswith("quartic")
            and detail != "fourth"
            or label.startswith("cubic")
            and detail == "covariance"
        ):
            np.testing.assert_allclose(moments, 0, atol=2e-12)
        if label.endswith("-composition") and mode != "joint":
            np.testing.assert_allclose(moments, 0, atol=2e-12)
    else:
        payload = dict(m=d["m"], n=d["n"])
    pending = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        pending,
        **payload,
        ids=d["ids"],
        time=d["time"],
        levels=d["levels"][::2],
        volume=d["volume"],
    )
    pending.replace(path)
    with np.load(path) as raw:
        np.testing.assert_array_equal(raw["ids"], d["ids"])
        m, n, shrink = decode_joint(raw) if k else (raw["m"], raw["n"], 1.0)
        return (
            tuple(
                map(
                    torch.from_numpy,
                    (m, n / float(raw["volume"]), raw["levels"].repeat(2, 0)),
                )
            ),
            raw["time"],
            shrink,
        )


@torch.inference_mode()
def run(
    root,
    out,
    split,
    labels,
    families,
    nodes,
    details,
    modes,
    high,
    bdf,
    substeps,
):
    """Measure paired responses against both native and continuous controls."""
    rows = []
    solver = DualStages(3, 2)
    for label in labels:
        d, panels = inputs(root, split, label, families, high, bdf)
        folder = root / "archives" / split / label
        folder.mkdir(parents=True, exist_ok=True)
        for k in nodes:
            grid = (
                [
                    (detail, mode)
                    for detail in details
                    for mode in modes
                    if detail != "marginal" or mode == modes[0]
                ]
                if k
                else [("full", "full")]
            )
            for detail, mode in grid:
                file = folder / f"f{families}-n{k}-{detail}-{mode}.npz"
                (m, n, levels), t, shrink = restore(
                    file, d, k, detail, mode, label
                )
                w, g = solver.rollout(m, n, levels, t, substeps)
                bulk = (w * n[:, None, :, None]).sum(2).numpy()
                for ref, target, bound in panels:
                    before = len(rows)
                    reporting(
                        "rk3i2",
                        0,
                        bulk,
                        g.numpy(),
                        d,
                        target,
                        bound,
                        ref,
                        file,
                        label,
                        substeps,
                        0,
                        rows,
                    )
                    for r in rows[before:]:
                        r.update(
                            nodes=k,
                            terms=detail,
                            conditioning=mode,
                            minimum_positive_rescale=shrink,
                        )
            write_table(rows, out)
            print(label, k, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--split", default="validation")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--families", type=int, default=48)
    p.add_argument("--nodes", nargs="+", type=int, default=[0, 3, 5, 8])
    p.add_argument(
        "--details",
        nargs="+",
        default=["marginal", "covariance", "third", "fourth"],
    )
    p.add_argument("--modes", nargs="+", default=["global", "size", "joint"])
    p.add_argument("--high", type=int, default=256)
    p.add_argument("--substeps", type=int, default=2)
    p.add_argument("--bdf", action="store_true")
    torch.set_num_threads(1)
    run(**vars(p.parse_args()))
