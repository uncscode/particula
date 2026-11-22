"""Regenerate figures from tabular outputs without running the chamber again."""

import csv
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from summarize import read, vals, grouped

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "figures"
DEST.mkdir(exist_ok=True)
INK = "#243745"
BLUE = "#26678C"
ORANGE = "#B7532F"
GREEN = "#187766"
GRAY = "#94A1AA"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.edgecolor": "#85929A",
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.constrained_layout.use": True,
    }
)


def save(fig, name):
    fig.savefig(DEST / (name + ".png"), dpi=180)
    plt.close(fig)


def select(rows, **match):
    return [r for r in rows if all(r[k] == str(v) for k, v in match.items())]


def title(ax, s):
    ax.set_title(s, loc="left", fontsize=12, pad=11)


def coverage():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 4))
    r = read("reference.csv")
    n = [x for x in r if x["primary_RH"] == "True" and x["q_cm3_s"] == "0.0"]
    a.scatter(
        vals(n, "acid_1e6_cm3"),
        vals(n, "ammonia_pptv"),
        s=20,
        color=GRAY,
        alpha=0.75,
    )
    for field, color, marker, text in [
        ("223_low", ORANGE, "s", "223 K holdout"),
        ("248_low", BLUE, "D", "248 K holdout"),
    ]:
        z = select(n, **{field: "True"})
        a.scatter(
            vals(z, "acid_1e6_cm3"),
            vals(z, "ammonia_pptv"),
            c=color,
            marker=marker,
            s=34,
            label=text,
        )
    a.set(
        xscale="log",
        yscale="log",
        xlabel="Sulfuric acid (10⁶ molecules cm⁻³)",
        ylabel="Ammonia (pptv)",
    )
    title(a, "Neutral chamber conditions")
    a.legend()
    with (ROOT / "rate_sources/figure_s3_plotted_pairs.csv").open() as f:
        z = list(csv.DictReader(f))
    obs, pred = [vals(z, k) for k in list(z[0])[1:]]
    lim = [min(min(obs), min(pred)) * 0.7, max(max(obs), max(pred)) * 1.4]
    b.loglog(lim, lim, color=INK, lw=1)
    b.fill_between(
        lim, np.array(lim) / 3, np.array(lim) * 3, color="#e7eef1", zorder=0
    )
    b.scatter(obs, pred, s=10, alpha=0.65, color=BLUE)
    b.set(
        xscale="log",
        yscale="log",
        xlim=lim,
        ylim=lim,
        xlabel="Corrected formation rate (cm⁻³ s⁻¹)",
        ylabel="Published fit (cm⁻³ s⁻¹)",
    )
    title(b, "355 points in Dunne Fig. S3")
    save(fig, "coverage-reference")


def holdouts():
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(9.4, 4), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    rows = read("heldout.csv")
    for ax, split in [(a, "248_low"), (b, "223_low")]:
        sets = [
            sorted(select(
                rows,
                method="fits",
                config=f"{split}_{top}_linear_nominal_neutral_group",
            ), key=lambda row: float(row["acid_cm3"]))
            for top in ("full", "assisted")
        ]
        obs = vals(sets[0], "observed_J")
        x = np.arange(len(obs))
        ax.errorbar(
            x - 0.19,
            obs,
            yerr=[obs - obs / 2.5, 1.5 * obs],
            color=INK,
            fmt="o",
            capsize=3,
            label="CLOUD, factor 2.5",
        )
        for z, shift, style, color, lab in zip(
            sets,
            [0, 0.19],
            ["D", "s"],
            [BLUE, ORANGE],
            ["Binary allowed", "Ammonia required"],
        ):
            ax.scatter(
                x + shift,
                vals(z, "predicted_J"),
                marker=style,
                c=color,
                label=lab,
                zorder=4,
            )
        ax.set(
            yscale="log",
            xticks=x,
            xticklabels=[f'{float(r["acid_cm3"]) / 1e6:g}' for r in sets[0]],
            xlabel="Sulfuric acid (10⁶ molecules cm⁻³)",
            xlim=(-0.55, len(x) - 0.45),
            ylabel="Formation at 1.7 nm (cm⁻³ s⁻¹)",
        )
        title(
            ax,
            "248.3 K  ·  0.38 pptv NH₃"
            if ax == a
            else "223.0 K  ·  0.06 pptv NH₃",
        )
    a.legend(fontsize=8, loc="upper left")
    save(fig, "holdouts")


def survival():
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4))
    rows = select(read("chamber_flux.csv"), line=336, source="archived")
    cases = [
        ("strong_wall", "2 × wall", GRAY),
        ("acid_base", "Reference wall", BLUE),
        ("weak_wall", "½ × wall", GRAY),
        ("acid_pair_hard", "Finite hard sphere", ORANGE),
        ("acid_pair_vdw", "Attractive collision", GREEN),
    ]
    for ax, size in zip(axs, ["3.2", "10.0"]):
        for i, (sc, label, color) in enumerate(cases):
            z = select(rows, threshold_mobility_nm=size, scenario=sc)[0]
            val = float(z["mean_survival"]) * 100
            ax.scatter(val, i, c=color, s=55, zorder=3)
            ax.annotate(
                f"{val:.3g}%",
                (val, i),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        ax.set(
            xscale="log",
            yticks=range(5),
            yticklabels=[v[1] for v in cases],
            xlabel="Particles reaching threshold (%)",
            ylim=(4.5, -0.5),
        )
        ax.set_xlim((0.4, 120) if size == "3.2" else (2.5e-5, 7))
        ax.grid(axis="x", alpha=0.22)
        title(ax, f"{float(size):g} nm mobility diameter")
    fig.suptitle("248.3 K  ·  0.38 pptv NH₃  ·  1.80 × 10⁷ H₂SO₄ molecules cm⁻³")
    save(fig, "survival")


def design():
    rows = read("precision_design.csv")
    fig, ax = plt.subplots(figsize=(8.8, 4))
    settings = [
        ("archived_gas_envelope", "Broad gas ranges"),
        ("ammonia20pct", "NH₃ ±20%, acid −33/+50%"),
        ("both10pct", "Both gases ±10%"),
    ]
    for i, (pr, label) in enumerate(settings):
        y = [
            max(
                float(r["separation"])
                for r in select(
                    rows,
                    precision=pr,
                    T_K=t,
                    both_above_floor=True,
                    both_below_100=True,
                    within_seen_ammonia=True,
                )
            )
            for t in [223.0, 248.3, 278.5]
        ]
        ax.scatter(
            np.arange(3) + (i - 1) * 0.14,
            y,
            marker=["o", "s", "D"][i],
            s=57,
            color=[GRAY, ORANGE, BLUE][i],
            label=label,
            zorder=3,
        )
    for v, lab in [
        (1, "Envelopes touch"),
        (6.25, "Two factor-2.5 bands touch"),
    ]:
        ax.axhline(v, color=INK, lw=0.8, ls="--")
        ax.text(2.29, v, lab, ha="right", va="bottom", fontsize=8)
    ax.set(
        xticks=range(3),
        xticklabels=["223.0 K", "248.3 K", "278.5 K"],
        xlim=(-0.4, 2.35),
        yscale="log",
        ylim=(0.009, 11),
        ylabel="Largest separation factor in screened grid",
    )
    ax.legend(
        loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.24)
    )
    title(ax, "Gas precision and rate discrimination")
    save(fig, "next-measurement")


def warm():
    fig, ax = plt.subplots(figsize=(8.7, 4.3))
    rows = select(
        read("warm_1833.csv"),
        gas_scale=1.15,
        dt_s=30,
        wall_scale=1.0,
        coag=0,
        boundary_nm=2.9,
    )
    cases = [
        ("gas_transfer", BLUE, "Gas transfer"),
        ("finite_hard", GRAY, "Finite hard sphere"),
        ("vdw", ORANGE, "Attractive collisions"),
        ("vdw_wet", GREEN, "Attraction + water uptake"),
    ]
    for name, color, label in cases:
        z = select(rows, model=name)
        ax.plot(
            vals(z, "hour"),
            vals(z, "pred_N10_cm3") / 1000,
            label=label,
            color=color,
        )
    ax.plot(
        vals(z, "hour"),
        vals(z, "obs_N10_cm3") / 1000,
        color=INK,
        lw=2.2,
        label="Measured size distribution",
    )
    ax.set(
        xlabel="Hours since first size spectrum",
        ylabel="Particles from 9.7 nm physical\nto upper bin (10³ cm⁻³)",
        xlim=(0, 6),
        ylim=(0, None),
    )
    ax.legend(fontsize=8)
    title(ax, "CLOUD run 1833 at 297.8 K")
    save(fig, "warm-replay")


def toc():
    fig = plt.figure(figsize=(3, 3), layout=None, facecolor="#fff3ce")
    fig.set_layout_engine("none")
    ax = fig.add_axes([0.11, 0.16, 0.81, 0.51], facecolor="#fff3ce")
    ax.barh(
        [2, 1, 0],
        [0.03, 0.1451, 0.1226],
        color=[INK, BLUE, ORANGE],
        height=0.32,
    )
    for y, label in [
        (2, "CLOUD"),
        (1, "Binary allowed"),
        (0, "Ammonia required"),
    ]:
        ax.text(0.001, y + 0.23, label, fontsize=8)
    ax.set(
        yticks=[],
        xticks=[0, 0.05, 0.10, 0.15],
        xlim=(0, 0.16),
        ylim=(-0.4, 2.6),
        xlabel="Formation rate (cm⁻³ s⁻¹)",
    )
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.spines["left"].set_visible(False)
    fig.text(0.11, 0.85, "CLOUD chamber", weight="bold", size=15, color=INK)
    fig.text(0.11, 0.77, "248.3 K  ·  0.38 pptv ammonia", size=9)
    fig.text(0.11, 0.70, "1.80 × 10⁷ H₂SO₄ molecules cm⁻³", size=8)
    fig.savefig(
        DEST / "graphical-abstract.png", dpi=300, facecolor=fig.get_facecolor()
    )
    plt.close(fig)


if __name__ == "__main__":
    coverage()
    holdouts()
    survival()
    design()
    warm()
    toc()
