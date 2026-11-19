"""Experiment design using the time-to-count integral in dynamic.py."""

# The assertions qualify numerical results before publication.
# ruff: noqa: S101
import csv
import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from dynamic import (
    OUT,
    A,
    bisect_vector,
    critical,
    nodes,
    resistance,
    save_csv,
    seq,
    shortcut,
    thresholds,
)
from matplotlib import style as mplstyle
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FormatStrFormatter
from particula import plotting
from scipy.special import ndtr, ndtri

SIZES = np.unique(np.r_[np.geomspace(10, 200, 65), 20, 30, 50, 75, 100, 150])
TIMES = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
KAPPAS = (0.03, 0.10, 0.30, 0.61)
WIDTHS = (1.10, 1.20)
WINDOW = (0.05, 1.4)


def setpoint(dry_nm, k, seconds, order=80):
    """Supersaturation (%) required for the given diameter to reach 1 µm."""
    dry, k = np.broadcast_arrays(np.asarray(dry_nm) * 1e-9, np.asarray(k))
    peak, _ = critical(dry, k)
    initial = bisect_vector(lambda wet: seq(wet, dry, k) - 0.95, dry, peak)
    target = np.full(dry.shape, 1e-6)
    assert np.all(initial < target)
    barrier = np.minimum(peak, target)
    z, w = nodes(order)
    eq, coeff = [], []
    for left, right in ((initial, barrier), (barrier, target)):
        wet = left[..., None] + (right - left)[..., None] * z
        eq.append(seq(wet, dry[..., None], k[..., None]))
        coeff.append(
            w * (right - left)[..., None] * wet * resistance(wet, 1.0) / 4
        )
    eq, coeff = np.concatenate(eq, axis=-1), np.concatenate(coeff, axis=-1)
    lo, hi = seq(barrier, dry, k), np.full(dry.shape, 1.8)

    def transit(forcing):
        return np.sum(coeff / (forcing[..., None] - eq), axis=-1)

    assert np.all(transit(hi) < seconds)
    for _ in range(49):
        mid = (lo + hi) / 2
        slow = transit(mid) > seconds
        lo, hi = np.where(slow, mid, lo), np.where(slow, hi, mid)
    return 100 * ((lo + hi) / 2 - 1)


def invert_full(dry_nm, ss):
    """Invert the full κ-Köhler maximum for each dry diameter."""
    dry = np.broadcast_to(dry_nm * 1e-9, ss.shape)
    lo, hi = np.zeros_like(ss), np.full_like(ss, 5.0)
    valid = (ss < 100 * np.expm1(A / dry)) & (
        ss > 100 * (critical(dry, hi)[1] - 1)
    )
    for _ in range(42):
        mid = (lo + hi) / 2
        lower = critical(dry, mid)[1] > 1 + ss / 100
        lo, hi = np.where(lower, mid, lo), np.where(lower, hi, mid)
    return np.where(valid, (lo + hi) / 2, np.nan)


def generate():  # noqa: C901 -- scan and verification sequence
    """Generate the scan planner, retrieval grid and refinement checks."""
    rows = []
    for k in KAPPAS:
        for t in TIMES:
            mid = setpoint(SIZES, k, t)
            full = invert_full(SIZES, mid)
            short = shortcut(SIZES * 1e-9, 1 + mid / 100)
            valid = (critical(SIZES * 1e-9, k)[0] <= 1e-6) & (mid > 0)
            full, short = (
                np.where(valid, full, np.nan),
                np.where(valid, short, np.nan),
            )
            for width in WIDTHS:
                d10 = SIZES * width ** ndtri(0.9)
                d90 = SIZES * width ** ndtri(0.1)
                low, high = setpoint(d10, k, t), setpoint(d90, k, t)
                assert np.all((low < mid) & (mid < high))
                for i, d in enumerate(SIZES):
                    rows.append(
                        dict(
                            median_nm=d,
                            gsd=width,
                            input_kappa=k,
                            seconds=t,
                            ss10_pct=low[i],
                            ss50_pct=mid[i],
                            ss90_pct=high[i],
                            optical_gate_after_critical=bool(valid[i]),
                            full_kappa=full[i],
                            shortcut_kappa=short[i],
                            full_error_pct=100 * (full[i] / k - 1),
                            shortcut_error_pct=100 * (short[i] / k - 1),
                        )
                    )
    save_csv("lab-setpoints.csv", rows)
    # Tabulate size at which a particle just reaches the counter threshold.
    ss = np.geomspace(*WINDOW, 125)
    fine_d = np.geomspace(3, 550, 1601)
    transition = []
    max_error = 0.0
    quadr_error = 0.0
    for t in TIMES:
        ref = setpoint(fine_d, 0.03, t)
        assert np.all(np.diff(ref) < 0)
        positive = ref > 0
        cut = np.exp(
            np.interp(
                np.log(ss),
                np.log(ref[positive][::-1]),
                np.log(fine_d[positive][::-1]),
            )
        )
        assert ss.min() > ref.min() and ss.max() < ref.max()
        for width in WIDTHS:
            counts = 100 * ndtr(
                np.log(SIZES[None, :] / cut[:, None]) / np.log(width)
            )
            transition.append(counts)
            for d in (30, 100, 200):
                sample = (
                    d * 1e-9 * width ** ndtri((np.arange(1024) + 0.5) / 1024)
                )
                for j in (10, 65, 100, 120):
                    event = thresholds(
                        sample, 0.03, "input", 1 + ss[j] / 100, 1.0, 96
                    )[1]
                    max_error = max(
                        max_error,
                        abs(
                            100 * np.mean(event <= t)
                            - counts[j, np.where(SIZES == d)[0][0]]
                        ),
                    )
        for k in KAPPAS:
            quadr_error = max(
                quadr_error,
                float(
                    np.max(
                        abs(setpoint(SIZES, k, t, 120) - setpoint(SIZES, k, t))
                    )
                ),
            )
    assert max_error < 0.5 and quadr_error < 0.002
    # Dense composition axis for the retrieval-bias reference.
    kk = np.geomspace(0.01, 0.8, 45)
    d, k = np.meshgrid(SIZES, kk)
    bias = []
    forcing = []
    for t in (1.0, 5.0):
        point = setpoint(d, k, t)
        forcing.append(point)
        eligible = (critical(d * 1e-9, k)[0] <= 1e-6) & (point > 0)
        error = 100 * (shortcut(d * 1e-9, 1 + point / 100) / k - 1)
        bias.append(np.where(eligible, error, np.nan))
    np.savez_compressed(
        OUT / "lab-contours.npz",
        dry_nm=SIZES,
        ss_pct=ss,
        seconds=TIMES,
        gsd=WIDTHS,
        count_pct=np.reshape(transition, (len(TIMES), 2, len(ss), len(SIZES))),
        kappa=kk,
        bias_times=[1.0, 5.0],
        shortcut_error_pct=bias,
        ss50_pct=forcing,
    )
    (OUT / "lab-scan-checks.json").write_text(
        json.dumps(
            dict(
                max_count_check_pp=max_error,
                max_ss50_refinement_percentage_points=quadr_error,
            ),
            indent=2,
        )
        + "\n"
    )


@mplstyle.context("particula.default")
def plot_setpoints():
    """Plot 10%, 50% and 90% optical counting setpoints."""
    with (OUT / "lab-setpoints.csv").open() as file:
        rows = list(csv.DictReader(file))
    fig, axes = plotting.subplots(1, 2, sharey=True)
    for ax, width in zip(axes, WIDTHS, strict=True):
        ax.axhspan(0.05, 1.4, color="#eef1f0")
        for k, color in zip(
            KAPPAS, ["#006960", "#1972ab", "#a54b1c", "#6d4989"], strict=True
        ):
            selected = [
                r
                for r in rows
                if float(r["gsd"]) == width
                and float(r["input_kappa"]) == k
                and float(r["seconds"]) == 5
                and r["optical_gate_after_critical"] == "True"
            ]
            d = np.array([float(r["median_nm"]) for r in selected])
            low, mid, high = (
                np.array([float(r[key]) for r in selected])
                for key in ("ss10_pct", "ss50_pct", "ss90_pct")
            )
            # Omit portions of bands whose large-diameter tail can
            # trigger before activation.
            d10 = d * width ** ndtri(0.9)
            valid = (critical(d10 * 1e-9, k)[0] < 1e-6) & (low > 0)
            ax.fill_between(d, low, high, where=valid, color=color, alpha=0.16)
            ax.plot(d, mid, color=color, lw=2, label=f"κ = {k:.2f}")
        ax.axhline(1.4, color="#626e69", ls="--", lw=0.9)
        ax.set(
            xscale="log",
            yscale="log",
            xlim=(10, 200),
            ylim=(0.05, 25),
            title=f"Geometric standard deviation {width:.2f}",
            xlabel="Median dry mobility diameter (nm)",
        )
        ax.set_xticks(
            [10, 20, 40, 70, 100, 150, 200],
            labels=["10", "20", "40", "70", "100", "150", "200"],
        )
        ax.set_yticks(
            [0.05, 0.1, 0.2, 0.5, 1, 1.4, 3, 10, 20],
            labels=["0.05", "0.1", "0.2", "0.5", "1", "1.4", "3", "10", "20"],
        )
        ax.grid(alpha=0.16)
        plotting.legend(ax, loc="upper right")
    axes[0].set_ylabel("Supersaturation setpoint (%)")
    fig.suptitle(
        "Choose a 5 s supersaturation scan"
        "\nLine = 50% counted   ·   Band = 10–90% counted   ·   "
        "Gray = example 0.05–1.4% scan window",
        fontsize=12,
    )
    plotting.fit(fig)
    fig.savefig(OUT / "lab-scan-planner.png", dpi=300)
    plt.close(fig)


@mplstyle.context("particula.default")
def plot():
    """Plot the laboratory setpoints, count contours and κ error."""
    plot_setpoints()
    with np.load(OUT / "lab-contours.npz") as data:
        d, ss = data["dry_nm"], data["ss_pct"]
        fig, axes = plotting.subplots(
            2, 2, sharex=True, sharey=True, size="wide"
        )
        for row, t in enumerate((1.0, 5.0)):
            for col, width in enumerate(WIDTHS):
                ax = axes[row, col]
                counts = data["count_pct"][list(TIMES).index(t), col]
                ax.grid(False)
                field = ax.contourf(
                    d,
                    ss,
                    counts,
                    levels=np.arange(0, 101, 5),
                    cmap=plotting.colormap("magnitude"),
                )
                levels = ax.contour(
                    d,
                    ss,
                    counts,
                    levels=[10, 50, 90],
                    colors="white",
                    linewidths=[0.8, 1.6, 0.8],
                )
                ax.clabel(levels, fmt="%d%%", fontsize=12)
                ax.set(
                    xscale="log",
                    yscale="log",
                    title=f"{t:g} s exposure   ·   GSD {width:.2f}",
                )
                ax.set_yticks(
                    [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.4],
                    labels=["0.05", "0.1", "0.2", "0.4", "0.7", "1.0", "1.4"],
                )
        for ax in axes[:, 0]:
            ax.set_ylabel("Imposed supersaturation (%)")
        for ax in axes[-1]:
            ax.set_xlabel("Median dry mobility diameter (nm)")
        for ax in axes.flat:
            ax.set_xticks(
                [10, 20, 40, 70, 100, 150, 200],
                labels=["10", "20", "40", "70", "100", "150", "200"],
            )
            ax.minorticks_off()
        fig.suptitle(
            "Which fraction reaches 1 µm?  |  Weakly "
            "hygroscopic aerosol, κ = 0.03",
            fontsize=15,
        )
        fig.colorbar(
            field,
            ax=axes,
            label="Particles reaching 1 µm (%)",
            ticks=[0, 20, 40, 60, 80, 100],
        )
        plotting.fit(fig)
        fig.savefig(OUT / "lab-count-reference.png", dpi=300)
        plt.close(fig)
        fig, axes = plotting.subplots(1, 2, sharey=True)
        for i, t in enumerate((1.0, 5.0)):
            ax = axes[i]
            kk = data["kappa"]
            e = data["shortcut_error_pct"][i]
            extent = 150
            ax.set_facecolor("#dedede")
            ax.grid(False)
            field = ax.pcolormesh(
                d,
                kk,
                e,
                norm=SymLogNorm(linthresh=5, vmin=-extent, vmax=extent),
                cmap=plotting.colormap("difference"),
                shading="auto",
                rasterized=True,
            )
            lines = ax.contour(
                d,
                kk,
                e,
                levels=[-25, -5, 0, 5, 25, 100],
                colors="#303030",
                linewidths=0.7,
            )
            ax.clabel(lines, fmt="%g%%", fontsize=12)
            ax.contour(
                d,
                kk,
                data["ss50_pct"][i],
                levels=[1.4],
                colors="#164c30",
                linestyles="--",
                linewidths=2,
            )
            ax.set(
                xscale="log",
                yscale="log",
                title=f"{t:g} s exposure",
                xlabel="Median dry diameter (nm)",
                xlim=(10, 200),
            )
            ax.set_xticks(
                [10, 20, 40, 70, 100, 200],
                labels=["10", "20", "40", "70", "100", "200"],
            )
            ax.set_yticks(
                [0.01, 0.03, 0.1, 0.3, 0.8],
                labels=["0.01", "0.03", "0.1", "0.3", "0.8"],
            )
        axes[0].set_ylabel("Prescribed κ")
        fig.suptitle(
            "Signed error in κ at the 50% optical count point  "
            "·  gray = optical gate before activation\n"
            "Dashed boundary: sizes to its left need more than "
            "1.4% supersaturation",
            fontsize=13,
        )
        fig.colorbar(
            field,
            ax=axes,
            label="Eq. 10 retrieval bias (%)",
            ticks=[-100, -25, -5, 0, 5, 25, 100],
            format=FormatStrFormatter("%g"),
            extend="both",
        )
        plotting.fit(fig)
        fig.savefig(OUT / "lab-kappa-error.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    parser = ArgumentParser()
    parser.add_argument("--replot", action="store_true")
    if not parser.parse_args().replot:
        generate()
    plot()
