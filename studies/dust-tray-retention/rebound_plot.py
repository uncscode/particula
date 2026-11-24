"""Draw the 2 m/s timing and retention comparisons from archived results."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def rows(file, **criteria):
    """Select archived rows with matching case or measurement fields."""
    with (ROOT / file).open() as stream:
        return [
            r
            for r in csv.DictReader(stream)
            if all(r[key] == str(value) for key, value in criteria.items())
        ]


def main():
    """Plot the timing and mass changes at 2 m/s."""
    fig, (time, mass) = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.9),
        gridspec_kw={"width_ratios": [1, 1.15]},
        layout="constrained",
    )
    blue, red, grey = "#275f8a", "#ad502e", "#485257"
    for band, label, color, marker in [
        ("all", "All dust below 63 µm", blue, "o"),
        ("middle", "10–20 µm dust", red, "s"),
    ]:
        data = rows("rebound-timing.csv", wind_ms=2, band=band)
        x = [float(r["time_h"]) for r in data]
        if band == "all":
            time.plot(
                x,
                [100 * float(r["exposure_fraction"]) for r in data],
                "--",
                color=grey,
                label="Airborne exposure",
                zorder=1,
            )
        time.plot(
            x,
            [100 * float(r["observed_fraction_of_increment"]) for r in data],
            marker=marker,
            color=color,
            label=label,
        )
    time.set(
        xlim=(0.42, 3.08),
        ylim=(-4, 104),
        xticks=x,
        xlabel="Hours after low-wind onset",
        ylabel="Fraction of the 0.5–3 h change (%)",
    )
    time.set_title("A   When did the dust accumulate?", loc="left", fontsize=12)
    time.legend(loc="lower right", frameon=True, fontsize=9)

    def step(a, final, charge=0.0):
        return rows(
            "rebound-changing-surface.csv",
            alpha=a,
            early_vc10_ms=0.05,
            late_vc10_ms=final,
            late_charge_scale=charge,
        )[0]

    fitted = rows("rebound-results.csv", case="fitted_fine_removal", hours=3.0)[
        0
    ]
    labels = [
        "Observed",
        "Fixed capture",
        "Late charge",
        "Stronger adhesion",
        "Stronger adhesion,\nhigher impact speeds",
        "Fine removal\n(median fitted)",
    ]
    selected = [
        step(1.0, 0.05),
        step(1.0, 0.05, 10.0),
        step(1.0, 0.2),
        step(2.0, 0.2),
        fitted,
    ]
    obs = [
        rows("rebound-timing.csv", wind_ms=2, band=b) for b in ("all", "middle")
    ]
    gains = [
        [float(rr[-1]["observed_mg"]) - float(rr[0]["observed_mg"])]
        + [float(r[k]) for r in selected]
        for rr, k in zip(
            obs, ("net_gain_mg", "cutoff_middle_gain_mg"), strict=True
        )
    ]
    y = np.arange(len(labels))
    mass.barh(
        y - 0.17, gains[0], 0.31, color=blue, label="All dust below 63 µm"
    )
    mass.barh(y + 0.17, gains[1], 0.31, color=red, label="10–20 µm dust")
    ci = rows("rebound-mass-intervals.csv", cut_um=63.0)[0]
    lo, hi = (float(ci[k]) for k in ("welch_low_mg", "welch_high_mg"))
    mass.errorbar(
        gains[0][0],
        -0.17,
        xerr=[[gains[0][0] - lo], [hi - gains[0][0]]],
        fmt="none",
        ecolor="#182c38",
        capsize=3,
        linewidth=1.4,
    )
    mass.axvline(0, color=grey, lw=0.8)
    mass.set(
        yticks=y,
        yticklabels=labels,
        xlim=(-6, 25),
        xlabel="Net change from 0.5 to 3 h (mg per tray)",
    )
    mass.invert_yaxis()
    mass.set_title("B   How much stayed on the tray?", loc="left", fontsize=12)
    for ax in (time, mass):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y" if ax is time else "x", color="#dedede", lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Dust retention at 2 m/s", fontsize=16, weight="bold")
    fig.savefig(ROOT / "rebound-study.png", dpi=185)
    plt.close(fig)


if __name__ == "__main__":
    main()
