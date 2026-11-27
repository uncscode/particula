"""Draw two parcels from a standalone checkpoint and forecast the next
half hour.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from redraw_checkpoint import draw_box, load
from redraw_design import LEADS
from redraw_run import optics, physical, trajectory
from stage_two_design import Case


def replay(path, draw, table):
    """Restore all forcing and inventories from the archive; use new random
    streams.
    """
    packed, header = load(path.read_bytes())
    case = Case(**header["case"])
    start = header["seconds"]
    key = [case.seed, case.n, start, draw]
    forcing = {
        k: (np.asarray(v) if isinstance(v, list) else v)
        for k, v in header["forcing"].items()
    }
    population = np.random.default_rng(np.random.SeedSequence([*key, 2]))
    events = np.random.default_rng(np.random.SeedSequence([*key, 1]))
    boxes = [draw_box(b, population) for b in packed]
    obs = trajectory(boxes, events, forcing, case, start, optics(table))
    return pd.DataFrame(
        [
            dict(lead=s, **physical(o, forcing["volume"]))
            for s, o in zip(LEADS, obs, strict=True)
        ]
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("redraw-cloud-16.npz"),
        help="Portable checkpoint (default: 16-row cloud sample)",
    )
    p.add_argument("--draw", type=int, default=0)
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    a = p.parse_args()
    print(replay(a.checkpoint, a.draw, a.optics).to_csv(index=False), end="")
