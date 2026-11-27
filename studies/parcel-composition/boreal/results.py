"""Reduce calibration and held-out archives to auditable, per-regime tables."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from model import REGIMES, SEEDS
from selection import TOL, load, select

GATES = {
    "rmse": 2,
    "sup": 5,
    "coated": 5,
    "droplets": 10,
    "liquid": 10,
    "births": 10,
    "survival": 5,
}


def snapshot(root, out, caps):
    """Export cap scores, occupancy and reference absorption."""
    rows = []
    profiles = []
    optics = []
    envelope = {}
    for seed in SEEDS:
        for r in REGIMES:
            errors, checkpoints, choices = load(root, seed, r)
            key = (r, "test" if seed in SEEDS[8:] else "cal")
            worst = errors.max(axis=0)
            envelope[key] = np.maximum(envelope.get(key, worst), worst)
            s = caps[caps.regime == r].set_index(["grid", "rule"])
            for i, choice in enumerate(choices):
                cap = int(s.loc[choice, "cap"])
                for k in sorted({16, cap} - {0}):
                    values = errors[:, i, k - 1]
                    for cp, err in zip(checkpoints, values, strict=True):
                        rows.append(
                            [seed, r, int(cp[0]), int(cp[1]), *choice, k, *err]
                        )
            if seed in SEEDS[8:]:
                cf = 1e-6 / (100000 / 1000)  # nm² to Mm^-1 at cm³ per parcel
                optics.extend(
                    [seed, r, int(t), int(i), a * cf, w * cf, h * cf, int(n)]
                    for t, i, a, w, h, n in checkpoints
                )
                with np.load(root / str(seed) / r / "snapshot.npz") as z:
                    a = z["profiles"]
                profiles.extend(
                    [seed, r, int(t), int(i), int(b), int(n), chi, area * cf]
                    for t, i, g, b, n, chi, area in a[a[:, 2] == 16]
                )
    sweep = (
        [reg, split, *choices[i], k + 1, *v]
        for (reg, split), bounds in envelope.items()
        for i, curve in enumerate(bounds)
        for k, v in enumerate(curve)
    )
    sweep_fields = "regime split grid rule cap rmse sup coated wet hom rows"
    pd.DataFrame(sweep, columns=sweep_fields.split()).to_csv(
        out / "cap-sweep.csv.gz",
        index=False,
        float_format="%.7g",
        compression={"method": "gzip", "mtime": 0},
    )
    df = pd.DataFrame(
        rows,
        columns=(
            "seed regime time parcel grid rule cap rmse sup coated wet hom rows"
        ).split(),
    )
    df.to_csv(
        out / "snapshot-scores.csv.gz",
        index=False,
        float_format="%.7g",
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(
        profiles,
        columns=(
            "seed regime time parcel bin count chi absorption_Mm_inv"
        ).split(),
    ).to_csv(
        out / "resolved-profiles.csv.gz",
        index=False,
        float_format="%.7g",
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(
        optics,
        columns=(
            "seed regime time parcel dry_coated_Mm_inv "
            "wet_coated_Mm_inv dry_homogeneous_Mm_inv N"
        ).split(),
    ).to_csv(
        out / "optics.csv.gz",
        index=False,
        float_format="%.7g",
        compression={"method": "gzip", "mtime": 0},
    )
    return df


def restarts(root, out):
    """Collect forecast scores at common future checkpoints."""
    pieces = []
    for name in [
        "restarts",
        "paired",
        "restart75",
        "restart15",
        "restartdouble",
    ]:
        for f in sorted((root / name).glob("*/*/g*-k*-r*.csv")):
            grid, cap, rule = [int(s[1:]) for s in f.stem.split("-")]
            df = pd.read_csv(f)
            df = df[(df.end % 3600 == 0) | df.end.isin([5400, 9000])].copy()
            df["seed"] = int(f.parent.parent.name)
            df["regime"] = f.parent.name
            df["cap"] = cap
            df["grid"] = grid
            df["rule"] = rule
            df["run"] = name
            pieces.append(df)
    data = pd.concat(pieces, ignore_index=True)
    data.to_csv(
        out / "restart-scores.csv.gz",
        index=False,
        float_format="%.7g",
        compression={"method": "gzip", "mtime": 0},
    )
    return data


def summarize(root, out):
    """Compare frozen budgets with the held-out extrema."""
    caps = select(root / "scores", out / "caps.csv")
    snapshots = snapshot(root / "scores", out, caps)
    rest = restarts(root, out)
    tables = []
    for r in REGIMES:
        cap = int(caps.query("regime==@r and grid==10 and rule==2").cap.iloc[0])
        for split, seeds in [("cal", SEEDS[:8]), ("test", SEEDS[8:])]:
            cases = snapshots[snapshots.seed.isin(seeds)].query(
                "regime==@r and grid==10 and rule==2"
            )
            for k in sorted({cap, 16}):
                d = cases[cases.cap == k]
                if d.empty:
                    continue
                err = d[["rmse", "sup", "coated"]].to_numpy() / TOL
                i, j = np.unravel_index(err.argmax(), err.shape)
                f = d.iloc[i]
                tables.append(
                    [
                        r,
                        split,
                        "snapshot",
                        k,
                        len(d),
                        err[i, j],
                        ["rmse", "sup", "coated"][j],
                        f.seed,
                        f.time,
                        f.parcel,
                        *d[["rmse", "sup", "coated"]].max(),
                        0,
                        0,
                        0,
                        0,
                    ]
                )
            for (k, run), d in rest.query(
                "regime==@r and seed in @seeds"
            ).groupby(["cap", "run"]):
                errors = d[list(GATES)].copy()
                errors.loc[
                    d.end != 21600, ["droplets", "liquid", "births", "survival"]
                ] = np.nan
                err = errors.to_numpy() / list(GATES.values())
                i, j = np.unravel_index(np.nanargmax(err), err.shape)
                f = d.iloc[i]
                tables.append(
                    [
                        r,
                        split,
                        run,
                        k,
                        len(d),
                        err[i, j],
                        list(GATES)[j],
                        f.seed,
                        f.start,
                        f.parcel,
                        *errors.max(),
                    ]
                )
    pd.DataFrame(
        tables,
        columns=(
            "regime split diagnostic cap records normalized limiting "
            "seed boundary_seconds parcel rmse sup coated droplets "
            "liquid births survival"
        ).split(),
    ).to_csv(out / "regime-scores.csv", index=False, float_format="%.8g")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    summarize(a.root, a.out)
