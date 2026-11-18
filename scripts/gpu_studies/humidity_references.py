"""Refine the independent single-box direct-BAT trajectories for every input."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scripts.gpu_studies.humidity_campaign import run


def errors(candidate, reference):
    """Return per-box percent errors for radius and both transfer legs."""
    radius = np.max(np.abs(candidate[:, :, 0] / reference[:, :, 0] - 1), axis=0)
    metrics = [radius]
    for leg, ref in zip(
        np.split(candidate, 2), np.split(reference, 2), strict=True
    ):
        delta = (leg - leg[0]) - (ref - ref[0])
        peak = np.max(np.abs(ref[:, :, 2] - ref[0, :, 2]), axis=0)
        if np.any(peak <= 0):
            raise ValueError("Zero transfer cannot define a relative scale")
        metrics.append(np.max(np.abs(delta[:, :, 1:]), axis=(0, 2)) / peak)
    return 100 * np.stack(metrics, axis=-1)


def refine(job):
    """Double resolution until all three reference metrics converge."""
    factor, output = job
    previous = None
    for steps in (64, 128, 256, 512, 1024):
        path = output / f"{factor.hex()}-{steps}.npy"
        if not path.exists():
            # Publish only complete arrays if a worker loses disk space.
            with TemporaryDirectory(dir=output) as temp:
                pending = Path(temp) / path.name
                run(1, "cpu", steps, pending, factors=np.array([factor]))
                pending.replace(path)
        current = np.load(path)
        if previous is not None:
            change = errors(previous, current)
            if np.all(change < [0.01, 0.05, 0.05]):
                return factor, steps, change.flatten().tolist()
        previous = current
    raise RuntimeError(f"Reference failed to converge at factor {factor}")


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    factors = {1.0}
    for boxes in (8, 32, 128):
        factors.update(np.geomspace(0.5, 2, boxes))
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(
            pool.map(refine, [(f, args.output) for f in sorted(factors)])
        )
    (args.output / "references.json").write_text(json.dumps(results))
