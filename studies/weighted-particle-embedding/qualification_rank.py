"""Rank validation checkpoints by resolved horizons across forcing regimes."""

import json

import numpy as np
import torch
from multicycle_score import frames_by_time


def reference(root, key, folder, data):
    """Align refined water contrasts at shared observation times."""
    ids = data["ids"][::2].cpu().numpy()
    if key == "multi":
        with np.load(root / "multicycle-validation-refinement.npz") as d:
            np.testing.assert_array_equal(ids, d["ids"])
            target = d["water"][:, 1] - d["water"][:, 0]
            bound = d["uncertainty"].sum(1)
    else:
        source = (
            root / "extra-validation-refinement.json"
            if key == "extra"
            else root / "validation-refinement.json"
            if key == "pilot"
            else folder.parent / "validation-refinement.json"
        )
        by_id = {r["id"]: r for r in json.loads(source.read_text())["records"]}
        if set(ids) != set(by_id):
            raise ValueError("Incomplete validation refinement")
        target = np.stack([by_id[i]["pair_contrast"] for i in ids])
        bound = np.stack([by_id[i]["reference_error"] for i in ids])
    times, frames = frames_by_time(data["time"][::2].cpu().numpy())
    family = np.arange(len(ids))[:, None]
    target = target[family, frames]
    return dict(
        target=target,
        mask=np.abs(target) > 4 * bound[family, frames],
        frames=frames,
        time=times,
    )


@torch.inference_mode()
def score(chunks, d, ref):
    """Aggregate unique times within each family before applying thresholds."""
    predicted = torch.zeros_like(d["water"])
    for frame, water, _ in chunks:
        predicted.scatter_(1, frame, water)
    contrast = (predicted[1::2] - predicted[::2]).cpu().numpy()
    contrast = np.take_along_axis(contrast, ref["frames"], axis=1)
    target, mask = ref["target"], ref["mask"]
    delta = np.abs(contrast - target)
    relative = (delta * mask).sum(0) / np.maximum(
        (np.abs(target) * mask).sum(0), 1e-30
    )
    signs = ((contrast * target > 0) & mask).sum(0) / np.maximum(mask.sum(0), 1)
    # The initial condition is supplied. Score positive forecast lead times.
    valid = mask.any(0) & (ref["time"] > 0)
    failed = valid & ((signs < 0.9) | (relative > 0.25))
    chain = ~np.maximum.accumulate(failed)
    horizon = max(ref["time"][valid & chain], default=0.0)
    return dict(
        horizon_s=float(horizon),
        fraction=float(horizon / ref["time"][-1]),
        passing_fraction=float((valid & ~failed).sum() / max(valid.sum(), 1)),
        average_relative=float(relative[valid].mean()) if valid.any() else None,
        worst_relative=float(relative[valid].max()) if valid.any() else None,
    )
