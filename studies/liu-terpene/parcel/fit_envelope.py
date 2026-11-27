# %% Resolve the cloud-response range across the nine chamber inversion choices.
"""Refine every fit's critical-size population and quantify its number bracket."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from parcel import PAIRS, Case, load_survival, simulate, write_csv


def envelope(tables, folder, out):
    start = time.monotonic()
    data = list(csv.DictReader((folder / "results/matrix.csv").open()))
    rows = []
    cached = {}
    for pair, (cyc, acy) in PAIRS.items():
        scenarios = [
            (p, rh, model) for p, rh, model in tables if p in (cyc, acy)
        ]
        for p, rh, model in scenarios:
            d, v = tables[p, rh, model]
            for w in (0.2, 1.0):
                k = 0.10 if pair == "monoterpenes" else 0.05
                r = next(
                    x
                    for x in data
                    if x["precursor"] == p
                    and x["rh"] == rh
                    and x["model"] == model
                    and float(x["w"]) == w
                    and float(x["kappa"]) == k
                    and float(x["number"]) == 300
                    and float(x["small"]) == 65
                )
                cut = float(r["original_cutoff_nm"])
                c = Case(
                    precursor=p,
                    rh=rh,
                    w=w,
                    kappa=k,
                    model=model,
                    n=64,
                    probes=64,
                    dz=0.5,
                    rtol=2.0e-6,
                )
                key = (tuple(d), tuple(v), w, k, cut)
                if key not in cached:
                    new = simulate(
                        c,
                        tables,
                        "cuda:0",
                        np.linspace(0.965 * cut, 1.035 * cut, 65),
                    )
                    if float(new["max_threshold_interval_pp"]) > 0.2:
                        raise ValueError(
                            f"Cutoff refinement failed for {key[-3:]}: {new}"
                        )
                    cached[key] = new
                rows.append(
                    dict(
                        pair=pair,
                        rh=rh,
                        w=w,
                        model=model,
                        precursor=p,
                        **cached[key],
                    )
                )
    lookup = {(r["precursor"], r["rh"], r["model"], r["w"]): r for r in rows}
    comps = []
    for r in rows:
        cyc, acy = PAIRS[r["pair"]]
        if r["precursor"] != cyc:
            continue
        a = r
        b = lookup[acy, r["rh"], r["model"], r["w"]]
        comps.append(
            dict(
                pair=r["pair"],
                rh=r["rh"],
                w=r["w"],
                model=r["model"],
                cyclic_pct=a["fraction_pct"],
                acyclic_pct=b["fraction_pct"],
                gain_pp=b["fraction_pct"] - a["fraction_pct"],
                bracket_pp=a["max_threshold_interval_pp"]
                + b["max_threshold_interval_pp"],
            )
        )
    write_csv(out / "envelope.csv", rows)
    write_csv(out / "envelope-comparisons.csv", comps)
    (out / "envelope-runtime.json").write_text(
        json.dumps(
            dict(
                runs=len(cached),
                outputs=len(rows),
                device=wp.get_device("cuda:0").name,
                wall_seconds=time.monotonic() - start,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--survival", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    out = args.out or args.folder / "envelope"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "envelope.csv").exists():
        raise FileExistsError("Use an empty output directory")
    envelope(load_survival(args.survival), args.folder, out)
