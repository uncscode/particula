# %% Use observations for treatment and scenario inputs for atmospheric transfer
import argparse
import csv
from pathlib import Path
import numpy as np
from campaign import HERE, execute, replicates


def data(file):
    with (HERE / "film" / file).open() as h:
        return list(csv.DictReader(h))


def design(stage):
    if stage == "history":
        for trial, (kf, f, ka) in replicates().items():
            for row in data("replicates.csv"):
                if row["trial"] == trial and row["precursor"] == "limonene":
                    yield dict(
                        trial=trial,
                        category="history",
                        case="trajectory",
                        f=float(row["mass_fraction"]),
                        k=float(row["kappa"]),
                        w=1.0,
                    )
    elif stage == "structure":
        for trial, (kf, f, ka) in replicates().items():
            for slope in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
                yield dict(
                    trial=trial,
                    category="allocation",
                    case="slope",
                    k=ka,
                    f=f,
                    size_slope=slope,
                    w=1.0,
                )
            yield dict(
                trial=trial,
                category="allocation",
                case="random disappearance",
                k=ka,
                f=1.0,
                loading=300 * f,
                w=1.0,
            )
            for mult in [0.75, 1.25, 1.5, 1.75]:
                yield dict(
                    trial=trial,
                    category="activity",
                    case="relative transfer",
                    k=ka * mult,
                    f=f,
                    mult=mult,
                    w=1.0,
                )
            # Volume-additive mixing of the residue with returned products.
            for ret in [0.25, 0.5, 1.0]:
                for kr in [0.0, kf, ka]:
                    volume = f + ret * (1 - f)
                    k = (ka * f + kr * ret * (1 - f)) / volume
                    yield dict(
                        trial=trial,
                        category="return",
                        case="volume mixing",
                        k=k,
                        f=volume,
                        ret=ret,
                        k_return=kr,
                        w=1.0,
                    )
    elif stage == "regime":
        kf, f, ka = replicates()["S4"]
        for w, median, loading in [
            (5, 150, 100),
            (0.1, 75, 1000),
            (1, 65, 300),
            (1, 150, 300),
            (1, 100, 1000),
            (1, 100, 100),
        ]:
            for name, m, k in [
                ("fresh", 1.0, kf),
                ("affinity", 1.0, ka),
                ("joint", f, ka),
            ]:
                yield dict(
                    trial="S4",
                    category="regime",
                    case=name,
                    k=k,
                    f=m,
                    median=median,
                    loading=loading,
                    w=w,
                )
    elif stage == "humidity":
        rhdata = data("rh_scans.csv")
        for rh in [40, 60, 75, 83]:

            def interp(t, h):
                rr = sorted(
                    (float(r["rh_percent"]), float(r["kappa"]))
                    for r in rhdata
                    if r["precursor"] == "limonene" and r["treatment"] == t
                )
                return np.interp(h, *zip(*rr))

            for trial, (kf, f, ka) in replicates().items():
                for name, m, k, t in [
                    ("fresh", 1.0, kf, "fresh"),
                    ("joint", f, ka, "300 nm humid"),
                ]:
                    yield dict(
                        trial=trial,
                        category="RH shape",
                        case=name,
                        k=k * interp(t, rh) / interp(t, 80),
                        f=m,
                        rh=rh,
                        w=1.0,
                    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    for stage in args.stage:
        execute(design(stage), "cuda:0", args.out / (stage + ".csv"))
