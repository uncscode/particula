"""Measure per-observation forecast errors of native quadrature replays."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from multicycle_score import frames_by_time, summarize


def load_reference(root, mode, split):
    """Load verified native member and timestep-refined paired references."""
    multi = mode == "multicycle"
    dataset = (
        root / "data/multicycle" / split if multi else root / "data" / split
    )
    source = sorted(dataset.glob("families-*.npz"))
    if not source:
        raise ValueError(f"No input histories in {dataset}")
    files = [f.name for f in source]
    ids, times, water, digests = [], [], [], []
    for f in source:
        with np.load(f) as data:
            group = data["ids"]
            ids.extend(group.tolist())
            times.extend([data["time"]] * len(group))
            water.append(data["water"].T.reshape(len(group), 2, -1))
            digests.extend(
                [hashlib.sha256(f.read_bytes()).hexdigest()] * len(group)
            )
    reference_water = np.concatenate(water)
    times = np.stack(times)
    if multi:
        with np.load(root / f"multicycle-{split}-refinement.npz") as reference:
            np.testing.assert_array_equal(ids, reference["ids"])
            np.testing.assert_array_equal(digests, reference["source_sha256"])
            np.testing.assert_array_equal(times, reference["time"])
            reference_water = reference["water"]
            uncertainty = reference["uncertainty"]
        pair = reference_water[:, 1] - reference_water[:, 0]
        bound = uncertainty.sum(1)
        reference_level = "finest"
    else:
        ref = json.loads((root / f"{split}-refinement.json").read_text())
        np.testing.assert_array_equal(ids, [r["id"] for r in ref["records"]])
        for f in source:
            if (
                hashlib.sha256(f.read_bytes()).hexdigest()
                != ref["sources"][f.name]
            ):
                raise ValueError(f"Refined reference differs from {f}")
        pair = np.array([r["pair_contrast"] for r in ref["records"]])
        bound = np.array([r["reference_error"] for r in ref["records"]])
        if "member_water" in ref["records"][0]:
            reference_water = np.stack(
                [r["member_water"] for r in ref["records"]]
            )
            reference_level = "finest"
        else:
            reference_level = "archived"
    return files, ids, times, pair, bound, reference_water, reference_level


def score(root, mode, split, out, methods=None):
    """Score a reconstructed population over the complete forcing schedule."""
    files, ids, times, pair, bound, reference_water, reference_level = (
        load_reference(root, mode, split)
    )
    elapsed, frame = frames_by_time(times)
    rows = []
    families = np.arange(len(ids))
    directory = (
        "multicycle-quadrature-replay"
        if mode == "multicycle"
        else "quadrature-replay"
    )
    experiments = (
        sorted((root / directory).glob("*-n*"))
        if methods is None
        else [root / directory / method for method in methods]
    )
    for config in experiments:
        blocks = []
        start = 0
        for name in files:
            with np.load(config / split / ("forecast-" + name)) as result:
                a = len(result["ids"])
                blocks.append(result["water"].T.reshape(a, 2, -1))
                np.testing.assert_array_equal(
                    result["ids"], ids[start : start + a]
                )
                np.testing.assert_array_equal(result["time"], times[start])
                start += a
        forecast = np.concatenate(blocks)
        chain = True
        for j in range(1, len(elapsed)):
            i = frame[:, j]
            pred = forecast[families, :, i]
            native = reference_water[families, :, i]
            # Some older two-leg files preserve only the refined contrast.
            uncertainty = bound[families, i, None]
            result = summarize(
                pred, native, uncertainty, contrast=pair[families, i]
            )
            if result["qualified"] is not None:
                chain = chain and result["qualified"]
            rows.append(
                dict(
                    method=config.name,
                    mode=mode,
                    split=split,
                    elapsed_s=elapsed[j],
                    member_reference=reference_level,
                    **result,
                    contiguous=chain
                    if result["qualified"] is not None
                    else None,
                )
            )
    if not rows:
        raise ValueError("No completed quadrature replays")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(out, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--mode", choices=["pulse", "multicycle"], default="pulse")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--split", default="validation")
    score(**vars(p.parse_args()))
