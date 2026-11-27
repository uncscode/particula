"""Locate the first cloudy-restart error in dry diameter and model time."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from particula.particles import get_binned_number_spectrum
import pandas as pd
from campaign import read
from checkpoint import decode, encode
from matplotlib.colors import SymLogNorm
from model import KAPPA, step


def draw(base, out, figpath):
    """Replay the hardest held-out boundary and localize droplet error."""
    seed, regime = 1562, "c1b1n0"
    source = base / str(seed) / regime / "10800.npz"
    ckpt = out / "cloudy-restart.npz"
    encode(ckpt, source, seed, regime)
    full, r0, t = read(source)
    rep, r1, h = decode(ckpt)
    edges = 1.7 * 10 ** (np.arange(31) / 10)
    rows = []
    time = []
    history = []
    for s in range(t, 21600, h["dt"]):
        for boxes, rng in ((full, r0), (rep, r1)):
            step(boxes, rng, h["forcing"], regime, s, h["dt"])
        counts = []
        for boxes in [full, rep]:
            b = boxes[0]
            dry = b.dry.sum(1)
            active = (dry + b.water > 1000**3) & (
                b.water > 1.05 * 49 * (b.dry @ KAPPA)
            )
            diameter = np.cbrt(dry[active]) * 1e-9
            weight = np.full(diameter.shape, 1e6 / h["forcing"]["volume"])
            counts.append(get_binned_number_spectrum(
                diameter, weight, edges * 1e-9
            ).bin_number_density * 1e-6)
        end = s + h["dt"]
        time.append(end / 3600)
        history.append(
            [full[0].droplets, rep[0].droplets, full[0].liquid, rep[0].liquid]
        )
        rows.extend(
            [end, edges[i], edges[i + 1], a, b]
            for i, (a, b) in enumerate(zip(*counts, strict=True))
        )
    data = pd.DataFrame(
        rows, columns="time low_nm high_nm full_cm3 reconstructed_cm3".split()
    )
    data.to_csv(out / "cloud-error.csv", index=False, float_format="%.8g")
    a = np.asarray(history)
    history = pd.DataFrame(
        {"hour": time, "full_cm3": a[:, 0], "redraw_cm3": a[:, 1]}
    )
    history.to_csv(out / "cloud-history.csv", index=False)
    plot(data, history, figpath)
    print(
        "checkpoint bytes",
        ckpt.stat().st_size,
        "future peak",
        a[:, :2].max(0),
        "liquid",
        a[:, 2:].max(0),
    )


def plot(data, history, figpath):
    """Draw the cloud history and dry-bin errors from the saved tables."""
    edges = np.r_[data.low_nm.unique(), data.high_nm.max()]
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(8.4, 6), sharex=True, layout="constrained"
    )
    time = history.hour
    a = history[["full_cm3", "redraw_cm3"]].to_numpy()
    ax.plot(time, a[:, 0], label="Full particles", lw=2, color="#16456b")
    ax.plot(
        time, a[:, 1], label="16 per dry bin", lw=2, ls="--", color="#b74418"
    )
    ax.set(
        ylabel="Cloud droplets (cm⁻³)",
        title="Restart inside the cloud · history 1562, parcel A",
    )
    ax.legend(frameon=False)
    delta = (
        (data.reconstructed_cm3 - data.full_cm3)
        .to_numpy()
        .reshape(-1, len(edges) - 1)
    )
    mx = max(abs(delta.min()), delta.max(), 1)
    im = bx.pcolormesh(
        np.r_[3.0, time],
        edges,
        delta.T,
        cmap="RdBu_r",
        norm=SymLogNorm(0.25, vmin=-mx, vmax=mx),
    )
    bx.set(
        yscale="log",
        ylim=(17, 1700),
        xlabel="Hours since parcel initialization",
        ylabel="Dry diameter (nm)",
    )
    fig.colorbar(im, ax=bx, label="Droplet error (cm⁻³ per bin)")
    fig.savefig(figpath, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("figure", type=Path)
    a = p.parse_args()
    draw(a.base, a.out, a.figure)
