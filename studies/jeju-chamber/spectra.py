"""Recreate Jeju spectral cuts and fixed-fit scoring sensitivity from trajectories."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from particula.particles import get_number_per_decade
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "results"
MINUTES = (12, 20, 30)
BANDS = ((2, 5), (5, 10), (10, 20))
SERIES = (
    ("Measured", "#212327", "-", 1.65),
    ("Pooled fit", "#bb650f", "--", 1.4),
    ("Individual fit", "#006a97", "-", 1.4),
)
FILLS = (
    "Ambient, unsalted",
    "Ambient, salt blend",
    "Ambient, pure salt",
    "Sulfate, salt",
    "Sulfate and dust, salt",
    "Colder ambient, salt",
)


def contours(o, pooled, independent, dest):
    ds = o["edges"]
    dw = np.diff(np.log10(ds))
    fig, ax = plt.subplots(
        6, 4, figsize=(13, 13), sharex=True, sharey=True, layout="constrained"
    )
    for r in range(6):
        valid = o["valid"][r]
        rr = o["y"][r]
        vals = [rr, pooled[r], independent[r]]
        vals.append(np.log10((vals[-1] + 0.1 * dw) / (rr + 0.1 * dw)))
        for j, data in enumerate(vals):
            a = ax[r, j]
            img = np.full(data.shape, np.nan)
            img[valid] = (
                data[valid]
                if j == 3
                else get_number_per_decade(data[valid] * 1e6, ds * 1e-6) * 1e-6
            )
            opts = (
                dict(cmap="RdBu_r", vmin=-2, vmax=2)
                if j == 3
                else dict(cmap="viridis", norm=LogNorm(0.1, 1e4))
            )
            show = a.pcolormesh(
                np.arange(len(valid) + 1),
                ds,
                img.T if j == 3 else np.maximum(0.1, img.T),
                **opts,
                rasterized=True,
            )
            if j == 0:
                cm = show
            a.set_yscale("log")
            a.set_yticks(
                [1, 3, 10, 30, 100], labels=["1", "3", "10", "30", "100"]
            )
            if j == 0:
                a.set_ylabel(f"Run {r + 1}\nDiameter (µm)")
            if r == 0:
                a.set_title(
                    [
                        "Measured",
                        "Pooled fit",
                        "Independent fit",
                        "Independent − measured",
                    ][j]
                )
            if r == 5:
                a.set_xlabel("Expansion minute")
    fig.colorbar(
        cm,
        ax=ax[:, :3],
        location="bottom",
        label="Number per decade of optical diameter (cm⁻³)",
    )
    fig.colorbar(
        show, ax=ax[:, 3], location="bottom", label="Signed error (decades)"
    )
    fig.savefig(dest / "six-spectra.png", dpi=135)
    plt.close(fig)


def curves(folder, step):
    with np.load(folder / f"pooled-full-{step}s.npz") as pooled:
        np.testing.assert_array_equal(pooled["runs"], np.arange(6))
        common = pooled["y"]
        expected = [pooled["scores"][:, 0, 0]]
    independent = []
    scores = []
    for run in range(6):
        with np.load(folder / f"R{run + 1}-full-{step}s.npz") as fit:
            np.testing.assert_array_equal(fit["runs"], [run])
            independent.append(fit["y"][0])
            scores.append(fit["scores"][0, 0, 0])
    return (common, np.stack(independent)), np.vstack([*expected, scores])


def loss(y, obs, width, stage, size_group, floor, occupied_only):
    residual = np.abs(np.log10((y + floor * width) / (obs + floor * width)))
    usable = (
        obs >= 0.1 * width if occupied_only else np.ones_like(obs, dtype=bool)
    )
    stages = []
    for time_group in np.unique(stage):
        bins = []
        for diameter_group in np.unique(size_group):
            subset = np.ix_(stage == time_group, size_group == diameter_group)
            mask = usable[subset]
            if mask.any():
                bins.append(residual[subset][mask].mean())
        if bins:
            stages.append(np.mean(bins))
    return np.mean(stages)


def check_and_score(folder, obs):
    rows = []
    width = np.diff(np.log10(obs["edges"]))
    for step in (1, 2):
        fits, archived = curves(folder, step)
        for r, valid in enumerate(obs["valid"]):
            ref = obs["y"][r, valid]
            assert np.isfinite(ref).all() and (ref >= 0).all()
            for floor, occupied in (
                (0.03, False),
                (0.1, False),
                (0.3, False),
                (0.1, True),
            ):
                errors = []
                for i, fit in enumerate(fits):
                    y = fit[r, valid]
                    assert np.isfinite(y).all() and (y >= 0).all()
                    v = loss(
                        y,
                        ref,
                        width,
                        obs["stage"][valid],
                        obs["band"],
                        floor,
                        occupied,
                    )
                    if floor == 0.1 and not occupied:
                        np.testing.assert_allclose(
                            v, archived[i, r], atol=1e-7, rtol=2e-6
                        )
                    errors.append(v)
                rows.append(
                    dict(
                        run=r + 1,
                        timestep_s=step,
                        floor_cm3_per_decade=floor,
                        measured_occupied_only=occupied,
                        pooled_decades=errors[0],
                        individual_decades=errors[1],
                        gain_decades=errors[0] - errors[1],
                    )
                )
    return rows


def draw(axis, edges, spectra):
    width = np.diff(np.log10(edges))
    for spec, (label, color, style, lw) in zip(spectra, SERIES, strict=True):
        density = get_number_per_decade(spec * 1e6, edges * 1e-6) * 1e-6
        np.testing.assert_allclose(width @ density, spec.sum(), rtol=1e-6)
        axis.stairs(
            np.where(density > 0, density, np.nan),
            edges,
            baseline=None,
            label=label,
            color=color,
            linestyle=style,
            linewidth=lw,
        )
    axis.set(
        xscale="log",
        yscale="log",
        xlim=(edges[0], edges[-1]),
        ylim=(0.01, 1.5e4),
    )
    axis.set_xticks([1, 3, 10, 30, 100], labels=["1", "3", "10", "30", "100"])
    axis.set_yticks([0.01, 1, 100, 1e4])
    axis.grid(axis="y", which="major", alpha=0.18)
    axis.spines[["right", "top"]].set_visible(False)


def render(obs, fits, dest):
    mean_fig, mean_axes = plt.subplots(
        2,
        3,
        figsize=(11.6, 6.6),
        layout="constrained",
        sharey=True,
        sharex=True,
    )
    cut_fig, cut_axes = plt.subplots(
        6,
        3,
        figsize=(10.9, 13.7),
        layout="constrained",
        sharey=True,
        sharex=True,
    )
    rows = []
    for r, valid in enumerate(obs["valid"]):
        series = (obs["y"][r], *[fit[r] for fit in fits])
        means = [y[valid].mean(axis=0, dtype=float) for y in series]
        end = np.flatnonzero(valid)[-1]
        ax = mean_axes.flat[r]
        ax.set_title(
            f"Run {r + 1}  ·  {FILLS[r]}\nMinutes 0–{end}", fontsize=10
        )
        draw(ax, obs["edges"], means)
        for j, minute in enumerate(MINUTES):
            assert valid[minute - 1 : minute + 2].all()
            draw(cut_axes[r, j], obs["edges"], [y[minute] for y in series])
            if r == 0:
                cut_axes[r, j].set_title(
                    f"Minute {minute}", fontsize=12, fontweight="semibold"
                )
            if j == 0:
                cut_axes[r, j].set_ylabel(
                    f"Run {r + 1}", fontsize=11, fontweight="semibold"
                )
        for label, arrays in [
            (str(t), [y[t] for y in series])
            for m in MINUTES
            for t in range(m - 1, m + 2)
        ] + [("record_mean", means)]:
            for series_id, spec in enumerate(arrays):
                record = dict(
                    run=r + 1,
                    minute=label,
                    series=SERIES[series_id][0],
                    total_cm3=float(spec.sum(dtype=float)),
                    modal_diameter_um=float(
                        obs["d"][
                            np.argmax(
                                get_number_per_decade(
                                    spec * 1e6, obs["edges"] * 1e-6
                                )
                            )
                        ]
                    ),
                )
                for lo, hi in BANDS:
                    mask = (obs["d"] >= lo) & (obs["d"] < hi)
                    record[f"number_{lo}_{hi}_um_cm3"] = float(
                        spec[mask].sum(dtype=float)
                    )
                rows.append(record)
    for fig, axes, filename in (
        (mean_fig, mean_axes, "mean-distributions.png"),
        (cut_fig, cut_axes, "minute-distributions.png"),
    ):
        fig.supxlabel("Water-equivalent optical diameter (µm)")
        fig.supylabel("Number per decade of diameter (cm⁻³)")
        fig.legend(
            *axes.flat[0].get_legend_handles_labels(),
            loc="outside upper center",
            ncol=3,
            frameon=False,
        )
        fig.savefig(dest / filename, dpi=155)
        plt.close(fig)
    return rows


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.input / "observations.npz") as obs:
        assert obs["y"].shape == (6, 43, 70)
        assert (np.diff(obs["edges"]) > 0).all()
        rows = check_and_score(args.input, obs)
        write_rows(args.out / "score-sensitivity.csv", rows)
        fits, _ = curves(args.input, 1)
        contours(obs, *fits, args.out)
        write_rows(args.out / "window-counts.csv", render(obs, fits, args.out))


if __name__ == "__main__":
    main()
