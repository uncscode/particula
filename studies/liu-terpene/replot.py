# %%
"""Render overview and proposed test from archived measurements and model outputs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from scripts.render_literature_tocs import dots_svg

ROOT = Path(__file__).resolve().parent
INK = "#243E43"
TEAL = "#135963"
RUST = "#A44D21"
PAPER = "#FBF2E3"


def toc(out, size_data, measurement_bounds):
    """Render overnight ocimene measurements and the prospective size transfer."""
    with measurement_bounds.open() as source:
        measured = sorted(
            (
                r
                for r in csv.DictReader(source)
                if r["precursor"] == "ocimene"
                and r["resolution"] == "half_bin"
                and r["status"] == "paired"
                and int(r["spectrum_min"]) > 900
            ),
            key=lambda r: int(r["spectrum_min"]),
        )
    with size_data.open() as source:
        forecast = [
            r
            for r in csv.DictReader(source)
            if r["precursor"] == "ocimene"
            and r["resolution"] == "half_bin"
            and r["diameter_nm"] == "60"
            and r["time_min"] == "1200"
            and int(r["accepted"]) > 0
        ]
    if len(measured) != 2 or any(r["diameter_nm"] != "220" for r in measured):
        raise ValueError("Expected two overnight observations at 220 nm")
    measured_center = sum(float(r["observed"]) for r in measured) / len(
        measured
    )
    measured_label = f"≈{100 * measured_center:.0f}%"
    forecast_ranges = []
    glyphs = []
    initial_radius = 37
    for kind, label, cx, color in [
        ("evolving_D", "Slower mixing", 78, TEAL),
        ("long_lived_fraction", "Lower volatility", 222, RUST),
    ]:
        rows = [r for r in forecast if r["mechanism"] == kind]
        low = min(float(r["low"]) for r in rows)
        high = max(float(r["high"]) for r in rows)
        midpoint = (low + high) / 2
        radius = initial_radius * midpoint ** (1 / 3)
        forecast_ranges.append(f"{100 * low:.1f}–{100 * high:.1f}%")
        # Illustrate each mechanism inside the residual sphere. Diameter is
        # quantitative; the diffusion path and molecular symbols are schematic.
        if kind == "evolving_D":
            interior = '<path d="M-12 9C-21 -3 -2 -8 -4 1S11 10 14 -7" stroke="white" stroke-width="1.8" fill="none"/><path d="m9 -5 5 -4 2 6" stroke="white" stroke-width="1.8" fill="none"/>'
        else:
            interior = '<path d="M-14 0l7 -8 10 3 8 -6 M3 -5l1 9 10 5" fill="none" stroke="white" stroke-width="1.7"/><g fill="white"><circle cx="-14" r="2.8"/><circle cx="-7" cy="-8" r="3.1"/><circle cx="3" cy="-5" r="3.1"/><circle cx="11" cy="-11" r="2.8"/><circle cx="4" cy="4" r="3.1"/><circle cx="14" cy="9" r="2.8"/></g>'
        glyphs.append(f'''<g transform="translate({cx} 198)">
<circle r="{initial_radius}" fill="none" stroke="#73817e" stroke-width="1.5" stroke-dasharray="4 4"/>
<circle id="residue-{kind}" r="{radius:.3f}" fill="{color}"/>{interior}
<text y="60" font-size="23" font-weight="bold" fill="{color}">{100 * low:.0f}–{100 * high:.0f}%</text>
<text y="78" font-size="13.5">{label}</text></g>''')
    measured_radius = 31 * measured_center ** (1 / 3)
    description = " and ".join(
        f"{100 * float(row['observed']):.1f}% at {int(row['spectrum_min'])} minutes"
        for row in measured
    )
    artwork = f"""<svg xmlns="http://www.w3.org/2000/svg" width="3in" height="3in" viewBox="0 0 300 300" role="img" aria-labelledby="title desc">
<title id="title">Dry volume remaining after β-ocimene aerosol evaporates</title>
<desc id="desc">At 220 nm and 75% relative humidity the two same-fill overnight measurements were {description}. At 60 nm after 20 hours with efficient vapor removal the forecasts are {forecast_ranges[0]} for slower mixing and {forecast_ranges[1]} for lower volatility. Each dashed circle is the initial size for that drawing. Solid particle diameters scale with the cube root of the volume fraction. The model drawings show the midpoint of each range. The winding path and molecular symbols illustrate the hypotheses.</desc>
<rect width="300" height="300" fill="{PAPER}"/>
<g fill="{INK}" font-family="DejaVu Sans, sans-serif">
<text x="16" y="26" font-size="20" font-weight="bold">Dry volume remaining</text>
<text x="17" y="46" font-size="12.5">β-Ocimene · 75% RH · overnight</text>
<g transform="translate(74 95)">
<path d="M-61 -4h13v8h-13m109 -8h13v8H48" fill="#CAD9D2" stroke="{INK}" stroke-width="1.6"/>
<rect x="-48" y="-34" width="96" height="68" rx="18" fill="#E4ECE5" stroke="{INK}" stroke-width="2"/>
<circle r="31" fill="none" stroke="#73817e" stroke-width="1.5" stroke-dasharray="4 4"/>
<circle id="residue-measured" r="{measured_radius:.3f}" fill="{INK}"/></g>
<text x="150" y="86" font-size="15">220 nm · measured</text>
<text x="148" y="115" font-size="29" font-weight="bold">{measured_label}</text>
<g text-anchor="middle">
<path d="M26 148h46m156 0h46" stroke="#9EABA3"/>
<text x="150" y="152" font-size="15" font-weight="bold">60 nm · predicted</text>
{"".join(glyphs)}</g>
<circle cx="19" cy="291" r="5.1" fill="none" stroke="#73817e" stroke-width="1.4" stroke-dasharray="3 2"/>
<text x="30" y="295" font-size="10.7">Initial particle size</text>
</g><g transform="translate(263 279) scale(.85)">{dots_svg()}</g></svg>"""
    (out / "toc.svg").write_text(artwork + "\n")


def figure(out, size_data, measurement_bounds):
    import pandas as pd
    from matplotlib import patheffects
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    runs = pd.read_csv(size_data).query(
        "resolution=='half_bin' and time_min==1200 and accepted>0"
    )
    bounds = runs.groupby(["precursor", "mechanism", "diameter_nm"]).agg(
        low=("low", "min"), high=("high", "max")
    )
    observations = (
        pd.read_csv(measurement_bounds)
        .query("resolution=='half_bin' and time_min>900")
        .sort_values("time_min")
    )
    panel_names = [
        ("ocimene", "β-Ocimene"),
        ("pinene", "α-Pinene"),
        ("caryophyllene", "β-Caryophyllene"),
        ("farnesene", "β-Farnesene"),
    ]
    # At equal values the teal shoulders and outer circle frame the orange trace.
    pool_style = dict(
        color=RUST,
        marker="s",
        ms=4.1,
        mfc="white",
        mew=1.25,
        lw=1.55,
        zorder=4,
        path_effects=[
            patheffects.withStroke(linewidth=2.65, foreground="white")
        ],
    )
    treatments = [
        (
            "evolving_D",
            "Slower internal mixing",
            dict(
                color=TEAL,
                marker="o",
                ms=8.5,
                lw=5,
                mew=0,
                solid_capstyle="round",
                zorder=3,
            ),
        ),
        ("long_lived_fraction", "Added low-volatility material", pool_style),
    ]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 6.5),
        sharex=True,
        sharey="row",
        layout="constrained",
    )
    for ax, (name, title) in zip(axes.flat, panel_names):
        obs = observations[observations.precursor == name]
        native = int(obs.diameter_nm.iloc[0])
        ax.set_title(
            f"{title}   ·   measured at {native} nm",
            loc="left",
            fontsize=11.5,
            pad=6,
        )
        ax.axvline(native, color="#a9adae", linestyle=":", lw=0.9, zorder=0)
        if name == "ocimene":
            ax.axvline(60, color="#657277", lw=0.85, ls="--", zorder=0)
            ax.text(
                0.062,
                0.975,
                "60 nm test",
                transform=ax.transAxes,
                va="top",
                fontsize=8.5,
                color=INK,
            )
        for kind, _, style in treatments:
            band = bounds.loc[(name, kind)]
            x = band.index.to_numpy()
            ax.fill_between(
                x,
                band.low,
                band.high,
                color=style["color"],
                alpha=0.15,
                linewidth=0,
                zorder=1,
            )
            ax.plot(x, (band.low + band.high) / 2, **style)
        offsets = (
            np.linspace(-2.5, 2.5, len(obs)) if len(obs) > 1 else np.zeros(1)
        )
        measurement = ax.errorbar(
            obs.diameter_nm.to_numpy() + offsets,
            obs.observed.to_numpy(),
            yerr=np.vstack(
                (obs.observed - obs.read_low, obs.read_high - obs.observed)
            ),
            color="#171e24",
            fmt="x",
            ms=6,
            mew=1.35,
            elinewidth=1.35,
            capsize=3.8,
            linestyle="none",
            zorder=7,
            label="Overnight measurement and reading range",
        )
        for artist in (*measurement[1], *measurement[2]):
            artist.set_path_effects(
                [
                    patheffects.withStroke(linewidth=2.8, foreground="white"),
                    patheffects.Normal(),
                ]
            )
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([60, 90, 120, 140, 165, 220])
        ax.set_xlim(53, 229)
    handles = [
        Line2D([], [], label=label, **style) for _, label, style in treatments
    ]
    handles.append(measurement)
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    fig.supxlabel(
        "Starting mobility diameter (nm)   ·   model predictions at 20 h and 75% RH",
        fontsize=11,
    )
    fig.supylabel("Apparent dry volume remaining (% of initial)", fontsize=11)
    fig.savefig(out / "size-test.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument(
        "--size-data",
        type=Path,
        default=ROOT / "uncertainty/results/model_ranges.csv",
    )
    parser.add_argument(
        "--measurement-bounds",
        type=Path,
        default=ROOT / "uncertainty/results/read_bounds.csv",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    toc(args.out, args.size_data, args.measurement_bounds)
    figure(args.out, args.size_data, args.measurement_bounds)
