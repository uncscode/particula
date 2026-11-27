"""Sensitivity of the next 75 seconds to using three 25-second steps."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import read
from redraw_checkpoint import draw_box, fit_box
from redraw_run import GATES, errors, percentage
from response_course import CASES, SHORT
from response_pairs import FAMILIES, run, tables
from stage_two_design import initialize


def compare(job):
    """Pair both discretizations at the same saved state and initial stream."""
    root, case, start = job
    opt, scat = tables(root)
    boxes, _, _ = read(root / "response-course" / case.key / f"{start}.npz")
    packed = [fit_box(b) for b in boxes]
    _, _, forcing = initialize(case)
    rows = []
    for trial in range(12):
        seed = [20261129, case.seed, start, trial]
        draw = np.random.default_rng([*seed, 1])
        redrawn = [draw_box(b, draw) for b in packed]
        outcomes = {}
        for name, original in (("full", boxes), ("redraw", redrawn)):
            for dt in (25, 75):
                event = np.random.default_rng([*seed, 0])
                outcomes[name, dt] = run(
                    deepcopy(original),
                    event,
                    forcing,
                    start,
                    75,
                    opt,
                    scat,
                    dt=dt,
                )[-1]
        for arm, pred, ref in (
            ("redraw_25", ("redraw", 25), ("full", 25)),
            ("redraw_75", ("redraw", 75), ("full", 75)),
            ("full_step_change", ("full", 75), ("full", 25)),
        ):
            o, r = outcomes[pred], outcomes[ref]
            e = errors(o, r)
            e["dry_sca_pct"], e["wet_sca_pct"] = percentage(
                o["scattering"], r["scattering"]
            ).max(0)
            e["channel_pct"] = percentage(o["channel"], r["channel"]).max()
            e["drop_pct"], e["lwc_pct"] = np.maximum(
                [e["drop_pct"], e["lwc_pct"]],
                percentage(o["current"], r["current"]),
            )
            row = dict(seed=case.seed, start=start, trial=trial, comparison=arm)
            for family, gates in {
                **FAMILIES,
                "nucleation": GATES["nucleation"],
            }.items():
                row[family] = max(e[k] / limit for k, limit in gates.items())
            rows.append({**row, **e})
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        batches = pool.map(
            compare, [(a.root, c, t) for c in CASES[:4] for t in SHORT]
        )
        data = pd.DataFrame(row for batch in batches for row in batch)
    data.to_csv(a.output, index=False, float_format="%.10g")
