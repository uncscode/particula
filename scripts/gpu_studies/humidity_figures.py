"""Plot concentration sensitivity and timing from saved data."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sensitivity(report, destination):
    """Show radius transients and the size of both water transfers."""
    factors = np.geomspace(0.5, 2, 128)
    time = np.r_[np.arange(33), 32 + np.arange(33)] * 2.5
    fig, (radius, transfer) = plt.subplots(
        1, 2, figsize=(9.2, 3.6), layout="constrained"
    )
    for index, color in zip(
        (0, 63, 127), ("#2166ac", "#555555", "#b35806"), strict=True
    ):
        radius.plot(
            time,
            np.asarray(report["radius_um"])[:, (0, 63, 127).index(index)],
            color=color,
            label=f"{factors[index]:.2f} × baseline",
        )
    radius.axvline(80, color="0.6", ls=":")
    radius.set(xlabel="Time (ms)", ylabel="Wet radius (µm)")
    radius.legend(frameon=False)
    for key, label, color in (
        ("wetting_mg_m3", "Wetting uptake", "#2166ac"),
        ("drying_loss_mg_m3", "Drying loss", "#b35806"),
    ):
        transfer.plot(
            factors,
            report[key],
            label=label,
            color=color,
        )
    transfer.set(
        xlabel="Number concentration / 10¹¹ m⁻³",
        ylabel="Particle water transferred (mg m⁻³)",
    )
    transfer.legend(frameon=False)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def timing(cases, destination):
    """Plot the full range of seven timed repetitions at each size."""
    fig, (total, memory) = plt.subplots(
        1, 2, figsize=(9.2, 3.6), layout="constrained"
    )
    for mode, label, color in (
        ("cpu", "Native CPU", "#444444"),
        ("prepared", "Prepared CUDA", "#b35806"),
        ("captured", "Captured CUDA", "#2166ac"),
    ):
        batch = sorted(
            (c for c in cases if c["mode"] == mode and c["steps"] == 16),
            key=lambda c: c["boxes"],
        )
        boxes = [c["boxes"] for c in batch]
        times = np.array(
            [[r["total"] for r in c["repetitions"]] for c in batch]
        )
        median = np.median(times, axis=1)
        total.errorbar(
            boxes,
            median,
            yerr=[median - times.min(axis=1), times.max(axis=1) - median],
            fmt="o-",
            label=label,
            color=color,
            capsize=3,
        )
        memory.plot(
            boxes,
            [
                max(r["peak_rss_kib"] for r in c["repetitions"]) / 1024
                for c in batch
            ],
            "o-",
            color=color,
            label=label,
        )
    for ax in (total, memory):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 32, 128], ["1", "8", "32", "128"])
        ax.set_xlabel("Independent boxes")
    total.set_yscale("log")
    total.set_ylabel("Time to saved history (s)")
    memory.set_ylabel("Peak host RSS (MiB)")
    total.legend(frameon=False)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    folder = Path("docs/images/gpu")
    folder.mkdir(parents=True, exist_ok=True)
    sensitivity(report, folder / "bat-pulse-sensitivity.png")
    timing(report["cases"], folder / "bat-pulse-timing.png")
