"""Benchmark learned cross-material features on native pulses."""

import argparse
from pathlib import Path

import numpy as np
import torch
from dual_experiment import reporting
from dual_stages import DualStages
from dual_ternary_record import decode_joint
from dual_ternary_score import inputs
from learned_joint import JointFeatures, compress, project
from node_transfer import write_table
from projection_study import digest


@torch.inference_mode()
def evaluate(
    root, out, checkpoints, split, labels, families, nodes, substeps, high, bdf
):
    """Score serialized per-box codes after reloading shared learned weights."""
    rows = []
    solver = DualStages(3, 2)
    for label in labels:
        d, targets = inputs(root, split, label, families, high, bdf)
        dst = root / "learned-records" / split / label
        dst.mkdir(parents=True, exist_ok=True)
        for path in checkpoints:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model = JointFeatures(saved["width"])
            model.load_state_dict(saved["model"])
            v = compress(d["m"], d["n"], nodes, "joint", model)
            file = dst / f"{path.stem}-n{nodes}-f{families}.npz"
            pending = file.with_suffix(".tmp.npz")
            np.savez_compressed(
                pending,
                **v,
                ids=d["ids"],
                time=d["time"],
                levels=d["levels"][::2],
                volume=d["volume"],
                shared_sha256=digest(path),
            )
            pending.replace(file)
            with np.load(file) as raw:
                if raw["shared_sha256"] != digest(path):
                    raise ValueError("Different shared decoder")
                m, n, shrink = decode_joint(project(raw, model))
                t = raw["time"]
                mass, number, levels = map(
                    torch.from_numpy,
                    (m, n / float(raw["volume"]), raw["levels"].repeat(2, 0)),
                )
            w, g = solver.rollout(mass, number, levels, t, substeps)
            bulk = (w * number[:, None, :, None]).sum(2).numpy()
            for ref, target, bound in targets:
                before = len(rows)
                reporting(
                    path.stem,
                    path.stat().st_size,
                    bulk,
                    g.numpy(),
                    d,
                    target,
                    bound,
                    ref,
                    file,
                    label,
                    substeps,
                    saved["seed"],
                    rows,
                )
                for row in rows[before:]:
                    row.update(
                        nodes=nodes,
                        learned_width=saved["width"],
                        flux_fraction=saved["flux_fraction"],
                        minimum_positive_rescale=shrink,
                        shared_sha256=digest(path),
                    )
        write_table(rows, out)
        print(label, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument(
        "--split",
        choices=["validation", "test", "challenge"],
        default="validation",
    )
    p.add_argument("--labels", nargs="+", required=True)
    for k, v in (
        ("families", 48),
        ("nodes", 5),
        ("substeps", 2),
        ("high", 256),
    ):
        p.add_argument("--" + k, type=int, default=v)
    p.add_argument("--bdf", action="store_true")
    torch.set_num_threads(1)
    evaluate(**vars(p.parse_args()))
