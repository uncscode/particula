"""Summarize residuals by scene and plot the satellite comparison."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    """Calculate bias and RMSE and display the individual signed residuals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    rows = np.genfromtxt(
        args.comparison, delimiter=",", names=True, dtype=None, encoding="utf8"
    )
    conditions = {"all": np.ones(len(rows), dtype=bool)}
    for surface in ("bare", "ocean"):
        select = rows["surface"] == surface
        conditions[surface] = select
        for key, cut in [
            ("aod550", 0.15),
            ("sza_deg", 40.0),
            ("vza_deg", 35.0),
        ]:
            conditions[f"{surface}_{key}_below_{cut:g}"] = select & (
                rows[key] < cut
            )
            conditions[f"{surface}_{key}_at_least_{cut:g}"] = select & (
                rows[key] >= cut
            )
        for src in np.unique(rows["source"]):
            conditions[surface + "_" + src[-10:]] = select & (
                rows["source"] == src
            )
    for s in np.unique(rows["stratum"]):
        conditions[s] = rows["stratum"] == s
    with args.output_csv.open("w") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(
            [
                "category",
                "observation",
                "n",
                "mean_observed",
                "mean_modeled",
                "bias",
                "rmse",
                "sd_residual",
            ]
        )
        for name, select in conditions.items():
            if not np.any(select):
                continue
            for v in ("sw", "b1", "b2"):
                obs = rows["observed_" + v][select]
                mod = rows["scene_" + v][select]
                d = mod - obs
                writer.writerow(
                    [
                        name,
                        v,
                        len(d),
                        *[
                            f"{x:.6f}"
                            for x in [
                                obs.mean(),
                                mod.mean(),
                                d.mean(),
                                np.sqrt(np.mean(d**2)),
                                d.std(ddof=1),
                            ]
                        ],
                    ]
                )
    fig, axes = plt.subplots(1, 3, figsize=(10.9, 3.4), layout="constrained")
    for ax, v, label in zip(
        axes,
        ["sw", "b1", "b2"],
        [
            "CERES outgoing shortwave",
            "MODIS red · band 1",
            "MODIS near infrared · band 2",
        ],
        strict=True,
    ):
        ax.axhline(0, color="#53636c", linewidth=1)
        for surf, color, marker in [
            ("bare", "#a7521d", "^"),
            ("ocean", "#13717b", "o"),
        ]:
            sel = rows["surface"] == surf
            ax.scatter(
                rows["vza_deg"][sel],
                (rows["scene_" + v] - rows["observed_" + v])[sel],
                color=color,
                marker=marker,
                label="Bare ground" if surf == "bare" else "Ocean",
                alpha=0.83,
                s=29,
            )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("MODIS view zenith (degrees)")
        ax.set_ylabel(
            "Model − observation\n"
            + ("W m⁻²" if v == "sw" else "W m⁻² sr⁻¹ µm⁻¹")
        )
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    axes[0].legend(frameon=False, fontsize=8)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
