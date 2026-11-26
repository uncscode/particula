# %% Matrix and compute
"""Native Particula reference, production, and refinement runs."""

import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from study import HERE, T0, par, simulate


def replicates():
    with (HERE / "film" / "replicates.csv").open() as h:
        data = list(csv.DictReader(h))
    groups = {}
    for trial in ("S4", "S5", "S7"):
        rows = [
            r
            for r in data
            if r["precursor"].lower() == "limonene" and r["trial"] == trial
        ]
        fresh = max(rows, key=lambda r: float(r["mass_fraction"]))
        late = min(rows, key=lambda r: abs(float(r["mass_fraction"]) - 0.30))
        groups[trial] = (
            float(fresh["kappa"]),
            float(late["mass_fraction"]),
            float(late["kappa"]),
        )
    return groups


def thresholds(k, f):
    rd = 50e-9 * np.cbrt(f)
    opt = minimize_scalar(
        lambda logr: -par._equilibrium(rd * np.exp(logr), rd, T0, k),
        bounds=(0, 8),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return 100 * (-opt.fun - 1)


def cases(stage):
    if stage == "screen":
        for trial, (kf, f, ka) in replicates().items():
            for w in (0.2, 1.0, 3.0):
                for name, k, m in [
                    ("fresh", kf, 1.0),
                    ("affinity", ka, 1.0),
                    ("shrinkage", kf, f),
                    ("joint", ka, f),
                ]:
                    yield dict(
                        trial=trial,
                        category="factorial",
                        case=name,
                        k=k,
                        f=m,
                        w=w,
                    )
    elif stage == "refine":
        for n, tol, dz in [(32, 1e-5, 1.0), (96, 1e-6, 1.0), (192, 2e-7, 0.5)]:
            kf, m, ka = replicates()["S4"]
            for name, k, f in [
                ("fresh", kf, 1.0),
                ("affinity", ka, 1.0),
                ("joint", ka, m),
            ]:
                yield dict(
                    trial="S4",
                    category="refine",
                    case=name,
                    k=k,
                    f=f,
                    w=1.0,
                    n=n,
                    tol=tol,
                    cadence_m=dz,
                )
    elif stage == "pilot":
        for name, k, f in [("fresh", 0.047, 1.0), ("joint", 0.110, 0.309)]:
            yield dict(trial="S4", category="pilot", case=name, k=k, f=f, w=1.0)


def run(row, device):
    k, f, w = (row[x] for x in ["k", "f", "w"])
    n = row.get("n", 64)
    kw = {
        x: row[x]
        for x in [
            "tol",
            "cadence_m",
            "median",
            "size_slope",
            "loading",
            "returned_share",
            "path_nm",
            "alpha",
        ]
        if x in row
    }
    start = time.monotonic()
    res = simulate(n, k, f, w, device, **kw)
    # Successive zero-weight probes recover the threshold in the same parcel.
    runs = 1
    for span, count in [(8 / n, 21), (8 / n / 10, 15)]:
        if res["boundary_bin_cm3"] < 0.035:
            break
        probes = np.linspace(
            max(-4, res["cutoff_z"] - span),
            min(4, res["cutoff_z"] + span),
            count,
        )
        res = simulate(n, k, f, w, device, probes=probes, **kw)
        runs += 1
    return dict(
        **row,
        device=device,
        bins=n,
        runs=runs,
        uniform_reference_critical_100nm_pct=thresholds(k, f),
        original_cut_nm=row.get("median", 100.0) * 1.5 ** res["cutoff_z"],
        **res,
        elapsed_s=time.monotonic() - start,
    )


def execute(rows, device, dst):
    dst.parent.mkdir(exist_ok=True, parents=True)
    results = []
    log = dst.with_suffix(".jsonl")
    if log.exists() or dst.exists():
        raise FileExistsError(
            f"Use a new output directory for the replay: {dst}"
        )
    for row in rows:
        item = run(row, device)
        results.append(item)
        with dst.with_suffix(".jsonl").open("a") as h:
            h.write(json.dumps(item) + "\n")
        print(
            row["trial"],
            row["case"],
            round(item["count_cm3"], 2),
            round(item["boundary_bin_cm3"], 2),
            flush=True,
        )
    fields = list(dict.fromkeys(k for row in results for k in row))
    with dst.open("w") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("out", type=Path)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    execute(cases(args.stage), args.device, args.out)
