"""Check the large-size ratio against CPC channel, flow, and service date."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from edges import EPOCH


def audit(path, out):
    a = np.load(path)
    keys, n = a["keys"], a["n"]
    u = a["diagnostic"]
    hours = int(a["hours"]) if "hours" in a else 6
    rows = []
    for block, S in sorted(set(map(tuple, keys[:, :2]))):
        take = (keys[:, 0] == block) & (keys[:, 1] == S)
        if n[take].sum() < max(2, hours * 2 // 3):
            continue
        vals = np.average(u[take], axis=0, weights=n[take])
        day = block * hours // 24
        cpc, ccn, serial, sample, sheath = (vals[1] / vals[0]).tolist()
        rows.append(
            [
                day,
                S / 10,
                (day - EPOCH) % 3 == 0,
                ccn / cpc,
                ccn / serial,
                cpc,
                ccn,
                sample,
                sheath,
                *vals[:, 3],
                *vals[:, 4],
            ]
        )
    f = pd.DataFrame(
        rows,
        columns="day S held pulse_normalized serial_normalized cpc_ratio ccn_ratio sample_flow_ratio sheath_ratio sample_b sample_d sheath_b sheath_d".split(),
    )
    date = np.datetime64("2023-05-24").astype(int)
    results = []
    for label, mask in [
        ("all", f.day.notna()),
        ("before_service", f.day < date),
        ("after_service", f.day > date),
    ]:
        for S, g in f[mask & f.held].groupby("S"):
            z = g.groupby("day").mean()
            for k in f.columns[3:]:
                vals = z[k].dropna().values
                if not len(vals):
                    continue
                rng = np.random.default_rng(771)
                means = vals[rng.integers(len(vals), size=(4096, len(vals)))].mean(
                    axis=1
                )
                results.append(
                    [
                        label,
                        S,
                        k,
                        len(vals),
                        vals.mean(),
                        *np.percentile(means, [2.5, 97.5]),
                    ]
                )
    pd.DataFrame(results, columns="period S metric days mean low high".split()).to_csv(
        out, index=False
    )


if __name__ == "__main__":
    audit(Path(sys.argv[1]), Path(sys.argv[2]))
