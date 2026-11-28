"""Plot the SABRE tracer-group uptake fits and small-mode allocation profile."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

INK = "#20313c"
SHARED = "#a74e13"
DUAL = "#006a75"


def render(data, profile, marks, output):
    """Render field-derived marks, fitted probabilities and loss shares."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10, 4.55), gridspec_kw={"width_ratios": [1.16, 1]}
    )
    fig.subplots_adjust(
        left=0.08, right=0.965, top=0.78, bottom=0.24, wspace=0.39
    )
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#98a5a8")
        axis.tick_params(color="#98a5a8", labelcolor=INK, labelsize=9)
        axis.set_axisbelow(True)
    left.set_title(
        "Uptake inferred for each air-mass group",
        loc="left",
        fontsize=12,
        pad=12,
    )
    shared = marks.sort_values("n2o_lo")
    left.hlines(
        shared.gamma_marker,
        shared.n2o_lo + 0.25,
        shared.n2o_lo + 4.75,
        label="Decker et al. poster",
        color=INK,
        linewidth=3.0,
        zorder=4,
    )
    for model, text, color, linestyle in (
        ("shared", "Shared probability", SHARED, "--"),
        ("two-class", "Separate probabilities", DUAL, "-"),
    ):
        rows = data[data.model == model].sort_values("n2o_lo")
        left.plot(
            rows.n2o_lo + 2.5,
            rows.fitted,
            color=color,
            ls=linestyle,
            lw=2.0,
            label=text,
            zorder=3,
        )
    left.set_xlabel("N₂O tracer mixing ratio (ppbv)", fontsize=10)
    left.set_ylabel("Effective N₂O₅ uptake probability", fontsize=10)
    left.set_xlim(279, 321)
    left.set_xticks([280, 290, 300, 310, 320])
    left.set_yscale("log")
    left.set_ylim(0.001, 0.021)
    yticks = [0.001, 0.002, 0.005, 0.01, 0.02]
    left.yaxis.set_major_locator(FixedLocator(yticks))
    left.yaxis.set_major_formatter(
        FixedFormatter([f"{val:.3g}" for val in yticks])
    )
    left.yaxis.set_minor_formatter(NullFormatter())
    left.grid(axis="y", color="#dee4e5")
    left.legend(loc="lower left", frameon=False, fontsize=8, handlelength=1.8)
    right.set_title(
        "Predicted small-mode share of uptake", loc="left", fontsize=12, pad=12
    )
    y = np.arange(len(profile))
    shares = profile.median_mode_rate_share.to_numpy() * 100
    right.barh(y, shares, height=0.45, color=DUAL)
    right.set_ylim(-0.8, 3.7)
    right.set_xlim(0, 102)
    right.invert_yaxis()
    right.set_yticks(y, [f"{v:g}" for v in profile.small_to_other])
    right.set_ylabel("Assumed γ (small) / γ (remaining)", fontsize=10)
    right.set_xticks([0, 20, 40, 60, 80])
    right.spines["bottom"].set_bounds(0, 80)
    right.set_xlabel(
        "Median small-mode loss share (%)", fontsize=10, loc="left"
    )
    right.text(
        94, -0.72, "Total k¹", ha="center", va="center", fontsize=8, color=INK
    )
    for yi, share, rate in zip(y, shares, profile.median_rate_s, strict=True):
        right.text(
            share + 1.6,
            yi,
            f"{share:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
            color=INK,
        )
        right.text(
            94,
            yi,
            f"{rate * 1e7:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color=INK,
        )
    fig.text(
        0.08,
        0.93,
        "Similar bulk uptake permits different small-mode contributions",
        fontsize=16,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.08,
        0.09,
        "Eight preliminary poster marks from March 8, 11 and 15, 2023. "
        "Lines use fitted probabilities with wet-area normalization.",
        fontsize=8.5,
        color=INK,
    )
    fig.text(
        0.08,
        0.04,
        "¹ Median total aerosol loss coefficient (10⁻⁷ s⁻¹) in 180 Arctic "
        "windows. Each assumed probability ratio is refitted.",
        fontsize=8.5,
        color=INK,
    )
    fig.savefig(
        output,
        dpi=170,
        facecolor="white",
        pil_kwargs={"lossless": True, "method": 6, "quality": 100},
    )
    plt.close(fig)
