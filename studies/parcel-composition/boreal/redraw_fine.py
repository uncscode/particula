"""Repeat the first held-out course at 25 s steps, keeping physical
leads fixed.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from redraw_design import TEST, WINDOWS, cases, reference
from redraw_run import worker

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("selection", type=Path)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    a = p.parse_args()
    choices = [tuple(json.loads(a.selection.read_text())["recipe"])]
    selected = [replace(c, dt=25) for c in cases(TEST[:1])]
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(reference, [(a.root, c) for c in selected]))
        list(
            pool.map(
                worker,
                [
                    (a.root, a.root / "fine", c, t, choices, a.optics)
                    for c in selected
                    for t in WINDOWS
                ],
            )
        )
