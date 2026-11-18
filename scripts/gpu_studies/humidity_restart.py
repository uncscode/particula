"""Verify canonical continuation across the humidity reset."""

import argparse
from pathlib import Path

import numpy as np
from scripts.gpu_studies.humidity_campaign import run


def verify(boxes, steps, output):
    """Compare canonical state and histories across the three paths."""
    baseline = None
    for mode, resumed in (
        ("prepared", False),
        ("captured", False),
        ("prepared", True),
        ("captured", True),
    ):
        states = []
        path = output / f"restart-{boxes}-{mode}-{resumed}.npy"
        run(boxes, mode, steps, path, restart=resumed, checkpoints=states)
        if baseline is None:
            baseline = states
            observations = np.load(path)
        np.testing.assert_array_equal(observations, np.load(path))
        for actual, expected in zip(states, baseline, strict=True):
            for left, right, fields in (
                (actual, expected, ("processes", "time_step", "auxiliary")),
                (
                    actual.resident,
                    expected.resident,
                    (
                        "payloads",
                        "rng_continuation",
                        "completed_steps",
                        "simulated_time",
                    ),
                ),
            ):
                for field in fields:
                    if getattr(left, field) != getattr(right, field):
                        raise ValueError(f"Checkpoint mismatch in {field}")
    print(
        f"Exact continuation passed at {boxes} boxes and {steps} steps/sample.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    for boxes in (1, 8, 32, 128):
        verify(boxes, 16, args.output)
