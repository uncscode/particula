"""Freeze a checkpoint recipe on training histories and report redraw
reliability.
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from redraw_design import CANDIDATES, DRAWS, LEADS, TEST, TRAIN, WINDOWS, cases
from redraw_run import FAMILIES, recipe_name

KEY = ["seed", "n", "start", "draw", "recipe"]
OUT = Path(__file__).parent


def collect(root, phase):
    """Require every expected course, checkpoint, draw and lead."""
    seeds = TRAIN if phase == "train" else TEST
    data = []
    for c in cases(seeds):
        for start in WINDOWS:
            q = pd.read_csv(root / phase / c.key / f"{start}.csv.gz")
            nmethods = len(CANDIDATES) + 1 if phase == "train" else 2
            if (
                len(q) != DRAWS * len(LEADS) * nmethods
                or q.duplicated(KEY + ["lead"]).any()
            ):
                raise ValueError(f"Incomplete records for {c.key}/{start}")
            data.append(q)
    return pd.concat(data, ignore_index=True)


def trials(data, paired=False):
    """Separate state reconstruction, next step, terminal and trajectory
    gates.
    """
    cols = list(FAMILIES)
    if paired:
        data = data.drop(columns=cols).rename(
            columns={"paired_" + k: k for k in cols}
        )
    pieces = []
    for label, selected in [
        ("snapshot", data.lead == 0),
        ("next step", data.lead == 75),
        ("half hour", data.lead == 1800),
        ("path", data.lead > 0),
    ]:
        g = data[selected].groupby(KEY)[cols].all().reset_index()
        g["joint"] = g[cols].all(axis=1)
        g["horizon"] = label
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def freeze(data, out):
    """Choose the cheapest 90% recipe or the best worst-cell fallback."""
    g = trials(data)
    paths = g[g.horizon == "path"]
    by = paths.groupby(["recipe", "n", "start"]).joint.mean()
    table = (
        by.groupby("recipe")
        .agg(worst="min", average="mean")
        .join(data.groupby("recipe").stored_bytes.mean().rename("mean_bytes"))
        .reset_index()
    )
    # The fixed-centroid control does not generate new within-row properties.
    candidates = table[
        table.recipe.isin([recipe_name(s) for s in CANDIDATES if s[-1]])
    ]
    qualified = candidates[candidates.worst >= 0.90]
    if len(qualified):
        best = qualified.sort_values(["mean_bytes", "recipe"]).iloc[0]
    else:
        best = candidates.sort_values(
            ["worst", "average", "mean_bytes", "recipe"],
            ascending=[False, False, True, True],
        ).iloc[0]
    table.to_csv(
        out / "redraw-calibration.csv", index=False, float_format="%.8g"
    )
    result = dict(
        recipe=list(map(int, best.recipe.split("-"))),
        qualified_on_training=bool(len(qualified)),
        threshold=0.90,
        training_seeds=TRAIN,
        held_out_seeds=TEST,
    )
    (out / "redraw-selection.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def reliability(g):
    """Describe conditional rates and percentile intervals over whole
    histories.
    """
    rng = np.random.default_rng(83211)
    rows = []
    for fields, frame in g.groupby(["recipe", "n", "horizon"]):
        grouped = frame.groupby("seed").joint.mean().to_numpy()
        sampled = rng.choice(grouped, (4000, len(grouped)), replace=True).mean(
            1
        )
        lo, hi = np.quantile(sampled, [0.025, 0.975])
        rows.append(
            dict(zip(["recipe", "n", "horizon"], fields, strict=True))
            | dict(
                passed=frame.joint.sum(),
                trials=len(frame),
                rate=frame.joint.mean(),
                seed_min=grouped.min(),
                seed_max=grouped.max(),
                block_lo=lo,
                block_hi=hi,
                **frame[list(FAMILIES)].mean().to_dict(),
            )
        )
    return pd.DataFrame(rows)


def render(data, recipe, path):
    """Show the four perturbation endpoints through one held-out trajectory."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 5), layout="constrained")
    for ax, (col, label, start) in zip(
        axes.flat,
        [
            ("abs_a", "Dry absorption, A (Mm⁻¹)", 9000),
            ("drops_cm3", "Droplets, A (cm⁻³)", 9000),
            ("births", "New births (particles)", 1800),
            ("organic_transfer", "Organic uptake (µm³)", 1800),
        ],
        strict=True,
    ):
        g = data[
            (data.seed == TEST[0]) & (data.n == 10000) & (data.start == start)
        ]
        scale = np.pi / 6 * 1e-9 if col == "organic_transfer" else 1.0
        for kind, color in [("full", "#b56e35"), (recipe, "#176785")]:
            b = g[g.recipe == kind]
            mat = scale * b.pivot(index="lead", columns="draw", values=col)
            ax.fill_between(
                mat.index / 60, mat.min(1), mat.max(1), color=color, alpha=0.14
            )
            ax.plot(
                mat.index / 60,
                mat.mean(1),
                color=color,
                label="Full repeat" if kind == "full" else "Redraw",
            )
        ref = scale * g.groupby("lead")["ref_" + col].first()
        ax.plot(
            ref.index / 60, ref, "k--", lw=1.2, label="Original realization"
        )
        ax.set(
            xlabel=f"Minutes after {WINDOWS[start]} checkpoint", ylabel=label
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("phase", choices=["train", "test"])
    p.add_argument("--optics", default=".artifacts/boreal/optics.npz")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument(
        "--figure",
        type=Path,
        default=Path("docs/images/parcel-redraw-course.png"),
    )
    a = p.parse_args()
    d = collect(a.root, a.phase)
    if a.phase == "train":
        print(freeze(d, a.out))
    else:
        recipe = recipe_name(
            json.loads((a.out / "redraw-selection.json").read_text())["recipe"]
        )
        if set(d.recipe) != {"full", recipe}:
            raise ValueError("Held-out recipe differs from the frozen setting")
        all_trials = trials(d)
        summary = reliability(all_trials)
        summary.to_csv(
            a.out / "redraw-reliability.csv", index=False, float_format="%.8g"
        )
        keys = ["recipe", "n", "start", "horizon"]
        event = (
            all_trials.groupby(keys)[[*FAMILIES, "joint"]].mean().reset_index()
        )
        event.to_csv(
            a.out / "redraw-events.csv", index=False, float_format="%.8g"
        )
        reliability(trials(d, True)).to_csv(
            a.out / "redraw-paired.csv", index=False, float_format="%.8g"
        )
        d.to_csv(
            a.out / "redraw-trajectories.csv.gz",
            index=False,
            float_format="%.9g",
            compression={"method": "gzip", "mtime": 0},
        )
        ensemble = pd.concat(
            [
                pd.read_csv(f)
                for f in sorted((a.root / "test").glob("*/*-ensemble.csv"))
            ]
        )
        expected = len(TEST) * 2 * len(WINDOWS) * len(LEADS) * 4
        if len(ensemble) != expected:
            raise ValueError("Missing held-out ensemble results")
        ensemble.to_csv(
            a.out / "redraw-ensembles.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        fine = pd.concat(
            [
                pd.read_csv(f)
                for f in sorted((a.root / "fine").glob("*/*.csv.gz"))
            ]
        )
        if len(fine) != 2 * len(WINDOWS) * DRAWS * 2 * len(LEADS):
            raise ValueError("Missing 25-second sensitivity runs")
        trials(fine).groupby(keys)[[*FAMILIES, "joint"]].mean().to_csv(
            a.out / "redraw-step-check.csv"
        )
        sample_case = next(c for c in cases(TEST[:1]) if c.n == 10000)
        sample = a.root / "test" / sample_case.key / f"{recipe}.npz"
        shutil.copyfile(sample, a.out / "redraw-cloud.npz")
        from redraw_replay import replay

        expected = d.query(
            "seed==@sample_case.seed & n==10000 & start==10800 "
            "& draw==0 & recipe==@recipe"
        ).sort_values("lead")
        actual = replay(sample, 0, a.optics)
        np.testing.assert_allclose(
            actual.to_numpy(),
            expected[actual.columns].to_numpy(),
            rtol=1e-8,
            atol=1e-10,
        )
        render(d, recipe, a.figure)
