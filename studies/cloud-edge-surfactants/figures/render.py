"""Render cloud-edge paper figures from bounded, model-only graph data."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
INPUT = STUDY / "plot_data"
OUTPUT = STUDY.parents[1] / "docs/images/cloud-edge-surfactants"
TEAL, RUST, GRAY = "#197b78", "#bd6234", "#b8c5c8"


def read(path):
    """Load a graph table or a separately obtained observation file."""
    with path.open() as source:
        return list(csv.DictReader(source))


def column(rows, key):
    """Return the numerical coordinates for a named column."""
    return [float(r[key]) for r in rows]


def equilibrium(observations):
    """Plot measured tensions, inventories, ordering and activation curves."""
    if not observations.is_file():
        raise FileNotFoundError(
            f"Fetch and extract the publisher workbook first: {observations}"
        )
    bench = read(observations)
    bounds = read(INPUT / "inventory.csv")
    relative = read(INPUT / "activation-shift.csv")
    colors = {"acid": "#585454", "octyl": "#077696", "dodecyl": "#ab4d28"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    a, b, c, d = axes.ravel()
    for rec, color in colors.items():
        for method, marker in (("trap", "o"), ("pendant", "^")):
            group = [
                r for r in bench if r["recipe"] == rec and r["method"] == method
            ]
            a.errorbar(
                column(group, "rh_percent"),
                column(group, "sigma_mN_m"),
                yerr=[float(r["reported_error_mN_m"] or 0.0) for r in group],
                linestyle="none",
                ms=4,
                marker=marker,
                color=color,
                mfc=color if method == "trap" else "white",
                label=rec if method == "trap" else None,
            )
        if rec != "acid":
            bound = next(row for row in bounds if row["recipe"] == rec)
            b.plot(
                [float(bound["low_um"]), float(bound["high_um"])],
                [rec, rec],
                lw=3,
                color=color,
            )
            b.scatter(float(bound["central_um"]), rec, color=color)
    for rec in ("octyl", "dodecyl"):
        group = [r for r in relative if r["recipe"] == rec]
        x = column(group, "dry_nm")
        d.plot(
            x,
            column(group, "direct_pct"),
            "--",
            color=colors[rec],
            linewidth=1.8,
        )
        d.plot(
            x,
            column(group, "finite_pct"),
            "-",
            color=colors[rec],
            linewidth=1.8,
        )
        d.fill_between(
            x,
            column(group, "low_pct"),
            column(group, "high_pct"),
            color=colors[rec],
            alpha=0.15,
            linewidth=0,
        )
    contrast = read(INPUT / "tension-contrast.csv")
    x = column(contrast, "dry_nm")
    c.fill_between(
        x,
        column(contrast, "low_mN_m"),
        column(contrast, "high_mN_m"),
        color="#b5d7d2",
        label="scenario range",
    )
    c.plot(
        x, column(contrast, "central_mN_m"), color="#185e66", label="central"
    )
    c.axhline(0, color="grey", lw=0.8)
    a.set(
        title="Figure 5 measurements at 295 K",
        xlabel="Relative humidity (%)",
        ylabel="Surface tension (mN m⁻¹)",
    )
    a.legend(frameon=False)
    a.text(0.04, 0.80, "● trap   △ pendant", transform=a.transAxes)
    b.set(
        title="Inventory needed for the 70% RH coverage",
        xlabel="Minimum dry diameter (µm)",
        xscale="log",
        ylim=(-0.5, 1.5),
    )
    b.text(
        0.03,
        0.02,
        "Points: central    Bars: adsorption scenarios",
        transform=b.transAxes,
        fontsize=9,
    )
    c.set(
        title="At 70% RH, the ordering depends on size",
        xscale="log",
        xlabel="Dry diameter (nm)",
        ylabel="Dodecyl minus octyl tension (mN m⁻¹)",
    )
    c.legend(frameon=False)
    d.axhline(0, color="grey", lw=0.8, zorder=-1)
    d.set(
        title="How much does each recipe shift activation?",
        xscale="log",
        xlabel="Dry diameter (nm)",
        ylabel="Critical supersaturation change\nfrom pure citric acid (%)",
    )
    d.set_yscale("symlog", linthresh=0.12, linscale=0.8)
    d.set_yticks(
        [-10, -3, -1, -0.3, -0.1, 0, 0.1, 0.3],
        ["−10", "−3", "−1", "−0.3", "−0.1", "0", "0.1", "0.3"],
    )
    for style, label in (("-", "Finite inventory"), ("--", "Direct transfer")):
        d.plot([], [], style, color="0.25", lw=1.5, label=label)
    d.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes.ravel():
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(OUTPUT / "equilibrium.png", dpi=170)
    plt.close(fig)


def identity():
    """Plot activated octyl, dodecyl and background droplets in each closure."""
    data = read(INPUT / "identity.csv")
    tab = {(r["regime"], r["model"]): r for r in data}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.05), sharey=True)
    fig.subplots_adjust(
        left=0.20, right=0.985, bottom=0.30, top=0.87, wspace=0.14
    )
    for a, reg in zip(axes, ("marine", "urban"), strict=True):
        for j, sc in enumerate(("central", "direct", "fixed70")):
            row = tab[reg, sc]
            n = [
                float(row[k])
                for k in ("octyl_cm3", "dodecyl_cm3", "background_cm3")
            ]
            left = 0
            for val, col, name in zip(
                n,
                [TEAL, RUST, GRAY],
                ["Octyl recipe", "Dodecyl recipe", "Background"],
                strict=True,
            ):
                a.barh(
                    j,
                    val,
                    left=left,
                    color=col,
                    height=0.46,
                    label=name if j == 0 else None,
                )
                left += val
            a.annotate(f"{left:.0f}", (left + 3, j), va="center", fontsize=10)
        a.set_title(reg.title(), loc="left", weight="bold")
        a.set_xlim(0, 402)
        a.set_yticks(
            [0, 1, 2],
            [
                "Dilution and\nfinite adsorption",
                "Measured dilution\ncurve transferred",
                "70%-RH tensions\nheld fixed",
            ],
        )
        a.set_xlabel("Activated droplets per cm³")
        a.spines[["top", "right", "left"]].set_visible(False)
        a.tick_params(axis="y", length=0)
        a.set_ylim(2.55, -0.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.56, 0),
        ncols=3,
        frameon=False,
    )
    fig.savefig(OUTPUT / "identity.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def attribution():
    """Plot paired inlet shifts and the three target-seed counterfactuals."""
    data = read(INPUT / "edge-attribution-paths.csv")
    effects = {
        r["placement"]: r for r in read(INPUT / "edge-attribution-effects.csv")
    }
    teal, rust, muted = "#076e70", "#a24527", "#65737a"
    fig, (a, b) = plt.subplots(
        1,
        2,
        figsize=(9, 3.8),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.25, 1]},
    )
    x = column(data, "inlet_m")
    for control, col, style, label in (
        ("finite", teal, "-", "Acid + surfactants"),
        ("acid", rust, ":", "Matched citric acid"),
    ):
        diff = column(data, control + "_shift_cm3")
        a.plot(
            x,
            diff,
            color=col,
            ls=style,
            lw=2 if control == "finite" else 1.5,
            label=label,
        )
    a.axhline(0, color=muted, lw=0.8)
    a.set(
        xlabel="Parcel inlet position (m)",
        ylabel="Change after dry inflow shifts (cm⁻³)",
        title="Same trajectory, two inflow placements",
    )
    a.legend(loc="lower right", frameon=False, fontsize=8)
    a.title.set_size(10.8)
    b.axvline(0, color=muted, lw=0.8)
    for shift, r, col, mark, label in (
        (-0.10, effects["patterned"], teal, "o", "Original"),
        (0.10, effects["opposed"], rust, "s", "Shifted"),
    ):
        b.scatter(
            [
                float(r[k])
                for k in (
                    "tagged_cm3",
                    "net_seed_addition_cm3",
                    "recipe_minus_acid_cm3",
                )
            ],
            np.arange(3) + shift,
            color=col,
            marker=mark,
            s=35,
            label=label,
        )
    b.set_yticks(
        [0, 1, 2],
        ["On target seeds", "Net added by seeds", "Recipes minus acid"],
    )
    b.set_ylim(2.6, -0.6)
    b.set_xlabel("Flux-weighted droplets per cm³")
    b.set_title("What the target seeds contribute", fontsize=10.8)
    b.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "edge-attribution.png", dpi=200)
    plt.close(fig)


def cloud_edge():
    """Plot six paired heights for each reciprocal inflow experiment."""
    data = read(INPUT / "cloud-edge.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), layout="constrained")
    for ax, resident, title in zip(
        axes,
        ("marine", "urban"),
        ("Marine air + urban inflow", "Urban air + marine inflow"),
        strict=True,
    ):
        for exchange, label, color, marker in [
            (0, "Closed parcel", GRAY, "o"),
            (0.15, "15% exchange", TEAL, "s"),
        ]:
            rows = [
                r
                for r in data
                if r["resident"] == resident
                and float(r["exchange"]) == exchange
            ]
            ax.plot(
                [float(r["height_m"]) for r in rows],
                [float(r["activated_cm3"]) for r in rows],
                label=label,
                color=color,
                marker=marker,
                ms=4,
            )
        ax.axvline(50, color=".4", lw=0.8, ls=":")
        ax.set_title(title, loc="left", weight="bold", size=11)
        ax.set_xlabel("Height above cloud base (m)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Activated droplets (cm⁻³)")
    axes[1].legend(frameon=False, fontsize=9)
    fig.savefig(OUTPUT / "cloud-edge.png", dpi=190)
    plt.close(fig)


def main():
    """Redraw one report figure or all four using local inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=[
            "all",
            "equilibrium",
            "identity",
            "edge-attribution",
            "cloud-edge",
        ],
        default="all",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=STUDY / "figure5.csv",
        help="Locally extracted Figure 5 observations",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.figure in ("all", "equilibrium"):
        equilibrium(args.observations)
    for label, render in (
        ("identity", identity),
        ("edge-attribution", attribution),
        ("cloud-edge", cloud_edge),
    ):
        if args.figure in ("all", label):
            render()


if __name__ == "__main__":
    main()
