"""Plot the dark wall-loss estimate and the temporal transfer of sulfur candidates."""

from pathlib import Path
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def read(name):
    with (HERE / name).open() as f:
        return list(csv.DictReader(f))


def values(rows, k):
    return np.array([float(r[k]) for r in rows])


def main():
    fig, (a, b) = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.4),
        layout="constrained",
        gridspec_kw=dict(width_ratios=[1.1, 1]),
    )
    profile = read("wall_profiles.csv")
    d = values(profile, "d_nm")
    for key, label, c, ls in [
        ("old_nominal_s", "Previous nominal", "#7f8890", "--"),
        ("native_dark_s", "Native law, dark calibration", "#237273", "-"),
    ]:
        a.plot(d, 1e4 * values(profile, key), ls=ls, c=c, lw=1.8, label=label)
    a.plot(
        d,
        values(profile, "old_fitted_s") * 1e4,
        c="#9f592f",
        ls=":",
        label="Previous fit (0.50 × nominal)",
    )
    r = [s for s in read("band_balance.csv") if s["window"] == "primary"]
    lo, hi = values(r, "lo_nm"), values(r, "hi_nm")
    cent = np.sqrt(lo * hi)
    y = (
        values(r, "scale")
        * np.exp(
            np.interp(
                np.log(cent),
                np.log(d),
                np.log(values(profile, "native_nominal_s")),
            )
        )
        * 1e4
    )
    a.errorbar(
        cent,
        y,
        xerr=[cent - lo, hi - cent],
        ls="none",
        fmt="o",
        ms=4,
        color="#1d3036",
        label="Size-band fits (band width)",
    )
    a.set(
        xscale="log",
        yscale="log",
        xlabel="Physical diameter (nm)",
        ylabel="Particle wall-loss rate (10⁻⁴ s⁻¹)",
    )
    a.legend(frameon=False, fontsize=8)
    a.set_title("Run 1833 dark decay", loc="left", weight="bold")
    rows = read("refinement.csv")
    for shift, period, label, marker, col in [
        (-0.1, "peak", "3.8–6 h", "o", "#974e39"),
        (0.1, "late", "6.36–32.97 h", "s", "#376f94"),
    ]:
        yy = []
        for name in ["parent", "gas_oh", "ozone", "peroxide"]:
            r = next(
                v
                for v in rows
                if v["family"] == name
                and v["bins"] == "140"
                and v["precharge_removal"] == "False"
                and v["period"] == period
            )
            yy.append(float(r["rmse"]) / 1000)
        b.scatter(
            np.arange(4) + shift,
            yy,
            marker=marker,
            s=46,
            color=col,
            label=label,
        )
    b.set(
        xticks=range(4),
        xticklabels=[
            "Measured\nacid",
            "OH + SO₂\ngas acid",
            "Fitted\nozone",
            "Fitted\nperoxide",
        ],
        ylabel="Count RMSE above 9.7 nm (10³ cm⁻³)",
        ylim=(0, None),
    )
    b.grid(axis="y", alpha=0.18)
    b.legend(frameon=False, fontsize=8)
    b.set_title("With particle coagulation", loc="left", weight="bold")
    for ax in (a, b):
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(HERE / "wall-and-sulfur.png", dpi=180)


if __name__ == "__main__":
    main()
