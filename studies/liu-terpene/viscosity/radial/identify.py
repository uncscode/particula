# %% Observe which constant bulk diffusivity each measured time would require.
"""Bound constant-D compatibility with conservative reading/time perturbations."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from radial import inputs, run, write


def scan(packet, out):
    start = time.monotonic()
    f, t, y = inputs(packet)
    ds = np.geomspace(8.0e-22, 1.0e-19, 49)
    targets = np.concatenate([t - 2, t, t + 2])
    rows = []
    intervals = []
    values = []
    for D in ds:
        tt, V, stats = run(f, targets, D, 112)
        pred = np.interp(targets, tt, V).reshape(3, len(t))
        values.append(pred)
        rows.extend(
            dict(
                D_m2_s=D,
                time_min=ti,
                early_model=lo,
                nominal_model=v,
                late_model=hi,
            )
            for ti, lo, v, hi in zip(t, *pred)
        )
    arr = np.array(values)
    if np.any(np.diff(arr, axis=0) > 1.0e-5):
        raise ValueError("Monotonicity failed")
    for i, (ti, obs) in enumerate(zip(t, y)):
        ok = (arr[:, 2, i] <= obs + 0.01) & (arr[:, 0, i] >= obs - 0.01)
        idx = np.flatnonzero(ok)
        if not len(idx) or idx[0] == 0 or idx[-1] == len(ds) - 1:
            raise ValueError("Extend diffusivity grid")
        low, high = ds[idx[0] - 1], ds[idx[-1] + 1]
        sign = np.log(ds)
        mid = np.exp(np.interp(obs, arr[::-1, 1, i], sign[::-1]))
        intervals.append(
            dict(
                time_min=ti,
                vfr=obs,
                D_nominal_m2_s=mid,
                D_outer_lower=low,
                D_outer_upper=high,
                assumed_vfr_reading=0.01,
                assumed_time_min=2,
                valid_grid_nodes=len(idx),
            )
        )
    write(out / "constant-D-grid.csv", rows)
    write(out / "constant-D-bounds.csv", intervals)
    (out / "identify-runtime.json").write_text(
        json.dumps(
            dict(runs=len(ds), wall_seconds=time.monotonic() - start), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("packet", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    scan(a.packet, a.out)
