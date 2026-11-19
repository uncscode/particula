"""Measure full and recomputed reverse trajectories in fresh CUDA processes."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp
from particula.execution.inverse_condensation import terminal_radius
from scripts.inverse_studies.distribution import NUMBER, OPTIONS, TRUTH, VAPOR


def measure(boxes, steps, segment):
    """Measure CUDA pool high water and gradient evaluation time."""
    device = "cuda:0"
    wp.init()
    if not wp.is_mempool_enabled(device):
        raise RuntimeError(
            "CUDA memory pool measurements require an enabled pool"
        )
    repeats = boxes // 2
    x = np.tile(TRUTH, (repeats, 1))
    n = np.tile(NUMBER, (repeats, 1))
    g = np.tile(VAPOR, repeats)
    wp.synchronize_device(device)
    before = wp.get_mempool_used_mem_high(device)
    start = perf_counter()
    radii, grad = terminal_radius(
        x,
        n,
        g,
        steps=steps,
        device=device,
        adjoint=np.ones_like(x),
        segment_steps=segment,
        **OPTIONS,
    )
    wp.synchronize_device(device)
    return dict(
        boxes=boxes,
        steps=steps,
        segment_steps=segment,
        high_water_bytes=wp.get_mempool_used_mem_high(device),
        initial_high_water_bytes=before,
        seconds=perf_counter() - start,
        radius_um=radii[:2].tolist(),
        adjoint=grad[:2].tolist(),
        warp=wp.__version__,
        device=wp.get_device(device).name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("boxes", type=int)
    parser.add_argument("steps", type=int)
    parser.add_argument("segment", type=int, help="zero selects the full tape")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(
            measure(args.boxes, args.steps, args.segment or None), indent=2
        )
        + "\n"
    )
