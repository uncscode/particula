# %% Count both chamber-constrained particle spectra under *identical* cloud forcing.
"""The carrier controls the water budget. The other spectrum has zero weight."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp
from parcel import PAIRS, Case, initial, load_survival, observe, par, write_csv


def run(pair, names, rh, w, tables, refined):
    k = 0.10 if pair == "monoterpenes" else 0.05
    rows = []
    for carrier, virtual in (names, names[::-1]):
        base = Case(
            precursor=carrier,
            rh=rh,
            w=w,
            kappa=k,
            n=80 if refined else 40,
            probes=240 if refined else 96,
            dz=0.5 if refined else 1.0,
            rtol=1.0e-6 if refined else 2.0e-5,
        )
        states = [
            initial(c, tables) for c in (base, replace(base, precursor=virtual))
        ]
        (a, ax), (b, bx) = states
        extra = replace(
            a,
            dry_radius=np.r_[a.dry_radius, b.dry_radius],
            number=np.r_[a.number, np.zeros_like(b.number)],
        )
        tr = par.run_parcel(
            extra,
            updraft=w,
            steps=round(107 / base.dz),
            timestep=base.dz / w,
            device="cuda:0",
            adaptive=par.ParcelROS3(rtol=base.rtol),
        )
        split = len(a.number)
        for role, name, s, x, wet in [
            ("water-coupled", carrier, a, ax, tr.wet_radius[:, :split]),
            ("zero-weight probe", virtual, b, bx, tr.wet_radius[:, split:]),
        ]:
            result = observe(base, replace(tr, wet_radius=wet), s, x)
            rows.append(
                dict(
                    pair=pair,
                    rh=rh,
                    w=w,
                    refined=refined,
                    carrier=carrier,
                    observed=name,
                    role=role,
                    **result,
                )
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--survival", type=Path, required=True)
    args = parser.parse_args()
    t = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "feedback.csv"
    if path.exists():
        raise FileExistsError("Choose a new output directory")
    tables = load_survival(args.survival)
    rows = [
        r
        for pair, names in PAIRS.items()
        for rh in ("dry", "wet")
        for w in (0.2, 1.0)
        for fine in (False, True)
        for r in run(pair, names, rh, w, tables, fine)
    ]
    write_csv(path, rows)
    report = dict(
        runs=len(rows) // 2,
        diagnosed_population_results=len(rows),
        device=wp.get_device("cuda:0").name,
        wall_seconds=time.monotonic() - t,
    )
    (args.out / "feedback-runtime.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(report)
