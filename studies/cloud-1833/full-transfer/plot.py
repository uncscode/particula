"""Replot count histories and size-resolved model errors on Donahue's clock."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from compare import (
    FIGURES,
    MODELS,
    PERIODS,
    RESULTS,
    diameter,
    edges,
    missing,
    observed,
    stage,
    tail,
    time,
)
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from particula import plotting

plt.style.use("particula.default")

FIGURES.mkdir(parents=True, exist_ok=True)
elapsed = time + 6.5
styles = {
    "Vapor transfer": (plotting.COLORS["indigo"], "-."),
    "Hard sphere": (plotting.COLORS["rust"], "--"),
    "Attraction": (plotting.COLORS["teal"], "-"),
    "Donahue fit": (plotting.COLORS["berry"], ":"),
}
series = pd.read_csv(RESULTS / "matched-series.csv.gz")
keep = series[(series.cells != 192) & (series.segment != "excimer_led_hold")]
fig, axs = plt.subplots(
    2,
    1,
    figsize=(10.8, 7.8),
    layout="constrained",
    gridspec_kw={"height_ratios": [1.4, 1]},
)
fig.suptitle(
    "Growth laws at the same incoming particle flux",
    fontsize=16,
    weight="bold",
    x=0.095,
    ha="left",
)
for ax in axs:
    ax.set_ylabel("Particles/cm³, 9.7–64.7 nm")
    ax.set_xlabel("Time from run start (hours)")
    for name, group in keep.groupby("model", sort=False):
        for seg, pts in group.groupby("segment"):
            c, ls = styles[name]
            ax.plot(pts.paper_h, pts.model_tail, color=c, ls=ls, lw=2)
    ax.scatter(
        elapsed,
        observed @ tail,
        label="Measured",
        s=5,
        color="#212d32",
        alpha=0.78,
        zorder=4,
    )
ax = axs[0]
for lo, hi in [(6.36, 10.79), (23.26, 26.37)]:
    ax.axvspan(lo, hi, color="#e1e4e2", zorder=-1)
    ax.text(
        (lo + hi) / 2,
        22100,
        "Cleaning",
        ha="center",
        color="#59666b",
        fontsize=11,
    )
ax.text(2.6, 23700, "Calibration", ha="center", color="#475960", fontsize=11)
ax.text(
    16.4, 23700, "Excimer transfer", ha="center", color="#475960", fontsize=11
)
ax.text(29.7, 23700, "Mixed light", ha="center", color="#475960", fontsize=11)
ax.set_xlim(0, 33)
ax.set_ylim(-700, 25500)
ax.set_title(
    "Complete record · gray intervals mark chamber cleaning", loc="left"
)
axs[1].set_xlim(elapsed[stage == 21].min() - 0.04, 33)
axs[1].set_ylim(-2, 95)
axs[1].set_title("After cleaning · enlarged count scale", loc="left")
handles = [
    Line2D([], [], marker=".", ls="none", color="#212d32", label="Measured")
] + [
    Line2D([], [], color=c, ls=ls, lw=2, label=k)
    for k, (c, ls) in styles.items()
]
fig.legend(handles=handles, loc="outside lower center", ncol=5)
fig.savefig(FIGURES / "count-comparison.png")
plt.close(fig)

pred = np.load(RESULTS / "predictions-288.npz")
pub = np.load(RESULTS / "published-spectra.npz")
panel_titles = {
    "Vapor transfer": "Particula · ordinary vapor transfer",
    "Hard sphere": "Particula · finite hard-sphere capture",
    "Attraction": "Particula · finite hard-sphere capture with van der Waals attraction",
    "Donahue fit": "Donahue et al. · published fit",
}
matrix = {}
for name in [*MODELS, "Donahue fit"]:
    arr = np.full_like(observed, np.nan)
    if name == "Donahue fit":
        for panel, (first, last) in [("fig09a", (1, 7)), ("fig10b", (9, 18))]:
            idx = (stage >= first) & (stage <= last)
            arr[idx] = pub[panel][idx]
    else:
        for seg, (first, last) in PERIODS.items():
            arr[(stage >= first) & (stage <= last)] = pred[seg][
                MODELS.index(name)
            ]
    arr[np.isin(stage, [1, 9, 21]) & np.r_[True, stage[1:] != stage[:-1]]] = (
        np.nan
    )
    matrix[name] = arr
for cutoff, filename in [
    (None, "spectrum-comparison-unmasked.png"),
    (50, "spectrum-comparison.png"),
]:
    fig, axs = plt.subplots(
        5, 1, sharex=True, figsize=(11.5, 11.4), layout="constrained"
    )
    fig.suptitle(
        "Where the size distributions agree and diverge",
        x=0.085,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    x = np.r_[
        elapsed[0] - 1 / 60,
        (elapsed[1:] + elapsed[:-1]) / 2,
        elapsed[-1] + 1 / 60,
    ]

    def paint(ax, v, cmap, **kwargs):
        ax.grid(False)
        color = plotting.colormap(cmap)
        color.set_bad("#e3e7e6")
        q = ax.pcolormesh(
            x,
            10 ** edges[4:],
            v[:, 4:].T,
            cmap=color,
            shading="flat",
            rasterized=True,
            **kwargs,
        )
        ax.set_yscale("log")
        ax.set_yticks([4, 10, 30, 60], labels=["4", "10", "30", "60"])
        ax.set_ylabel("Diameter (nm)")
        return q

    measured = observed.copy()
    measured[missing, :7] = np.nan
    q = paint(axs[0], np.log10(1 + measured), "magnitude", vmin=0, vmax=5)
    axs[0].set_title("CLOUD run 1833 · measurements", loc="left")
    axs[0].set_title(
        "Three separate Particula runs · same particle entry at 2.9 nm",
        loc="right",
        fontsize=11,
        color="#3c555c",
    )
    cb = fig.colorbar(q, ax=axs[0], ticks=[0, 2, 4], pad=0.015)
    cb.ax.set_yticklabels(["0", "100", "10,000"])
    cb.set_label("Particles/cm³\nper decade")
    for ax, (name, arr) in zip(axs[1:], matrix.items(), strict=True):
        val = np.log10((1 + arr) / (1 + measured))
        q = paint(
            ax,
            val,
            "difference",
            norm=TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5),
        )
        if cutoff is not None:
            weak = np.isfinite(arr) & (measured <= cutoff)
            flags = np.where(weak, (arr > cutoff).astype(float), np.nan)
            ax.pcolormesh(
                x,
                10 ** edges[4:],
                flags[:, 4:].T,
                cmap=ListedColormap(["#eae4d3", plotting.COLORS["indigo"]]),
                vmin=0,
                vmax=1,
                shading="flat",
                rasterized=True,
            )
        ax.set_title(panel_titles[name], loc="left")
    cb = fig.colorbar(
        q, ax=list(axs[1:]), ticks=[-1, 0, 1], pad=0.015, shrink=0.65
    )
    cb.ax.set_yticklabels(["0.1×", "1×", "10×"])
    cb.set_label("Model / measured\n(with unit offset)")
    axs[-1].set_xlabel("Time from run start (hours)")
    axs[-1].set_xlim(0, 33)
    if cutoff is not None:
        labels = [
            ("#eae4d3", "Both ≤50"),
            (plotting.COLORS["indigo"], "Only model >50"),
            ("#e3e7e6", "Unavailable"),
        ]
        fig.legend(
            handles=[Patch(color=c, label=l) for c, l in labels],
            loc="outside lower center",
            ncol=3,
            frameon=False,
            title="Ratio colors where measured density >50 particles/cm³ per decade",
        )
    fig.savefig(FIGURES / filename)
    plt.close(fig)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
fig.suptitle(
    "How the capture law changes the distribution", fontsize=15, weight="bold"
)
for model, arr in [("Measured", observed), *matrix.items()]:
    curve = arr[stage == 12, 4:].mean(0)
    if model == "Measured":
        ax.scatter(
            diameter[4:], curve, color="#212d32", s=11, label=model, zorder=5
        )
    else:
        c, ls = styles[model]
        ax.plot(diameter[4:], curve, color=c, ls=ls, label=model, lw=2)
ax.set_xscale("log")
ax.set_xlim(3.7, 65)
ax.set_xticks([4, 8, 15, 30, 60], labels=["4", "8", "15", "30", "60"])
ax.set_xlabel("Physical diameter (nm)")
ax.set_ylabel("Particles/cm³ per decade")
ax.set_title("Medium light · stage 12 mean spectrum")
scores = pd.read_csv(RESULTS / "stage-scores.csv")
for model in MODELS:
    g = scores[
        (scores.cells == 288)
        & (scores.model == model)
        & (scores.segment == "excimer_led")
        & scores.stage.between(9, 17)
    ]
    c, ls = styles[model]
    bx.plot(g.stage, g.shape_pct, ls=ls, color=c, marker=".", ms=6)
bx.set_xlabel("Excimer stage")
bx.set_xticks(range(9, 18))
bx.set_ylabel("Size-distribution mismatch (%)")
bx.set_title("Shape at equal particle entry")
fig.legend(handles=handles, loc="outside lower center", ncol=5)
fig.savefig(FIGURES / "capture-comparison.png")
plt.close(fig)
