"""Assess proposed measured-signal cutoffs on saved size distributions."""

import numpy as np
import pandas as pd
from compare import (
    FIGURES,
    MODELS,
    RESULTS,
    observed,
    stage,
    tail,
    time,
    widths,
)
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

columns = slice(4, None)
w = widths[columns]
preds = np.load(RESULTS / "predictions-288.npz")
author = np.load(RESULTS / "published-spectra.npz")
results = []
for label, seg, start, end in [
    ("Excimer", "excimer_led", 9, 17),
    ("After cleaning", "mixed", 21, 25),
]:
    ix = np.flatnonzero((stage >= start) & (stage <= end))[1:]
    models = dict(zip(MODELS, preds[seg][:, 1 : 1 + len(ix), columns]))
    if label == "Excimer":
        models["Donahue fit"] = author["fig10b"][ix, columns]
    obs = observed[ix, columns]
    masks = {
        "All": np.ones_like(obs, dtype=bool),
        "Density >50": obs > 50,
        "Bin count >50": obs * w > 50,
        "Whole spectrum >50": np.broadcast_to(
            (obs @ w > 50)[:, None], obs.shape
        ),
        "Tail count >50": np.broadcast_to(
            (observed[ix] @ tail > 50)[:, None], obs.shape
        ),
    }
    for criterion, mask in masks.items():
        ww = mask * w
        weight = ww.sum(axis=1)
        use = weight > 0
        for name, model in models.items():
            logerr = abs(np.log10((1 + model) / (1 + obs)))
            factor = (
                10 ** np.mean((logerr * ww).sum(axis=1)[use] / weight[use])
                if use.any()
                else np.nan
            )
            results.append(
                dict(
                    window=label,
                    criterion=criterion,
                    model=name,
                    spectra_retained=use.sum(),
                    spectra=len(obs),
                    cell_pct=100 * mask.mean(),
                    particle_pct=100 * (obs * ww).sum() / (obs * w).sum(),
                    factor=factor,
                )
            )
pd.DataFrame(results).to_csv(RESULTS / "threshold-check.csv", index=False)
# Two views of the same attractive replay. Keeping the context panel lets the reader see false positives.
from compare import PERIODS, edges, missing

meas = observed[:, columns].copy()
meas[missing, :3] = np.nan
p = np.full_like(meas, np.nan)
for seg, (a, b) in PERIODS.items():
    ix = np.flatnonzero((stage >= a) & (stage <= b))
    p[ix[1:]] = preds[seg][MODELS.index("Attraction"), 1:, columns]
valid = np.isfinite(meas) & np.isfinite(p)
err = np.log10((1 + p) / (1 + meas))
fig, axes = plt.subplots(
    2, 1, sharex=True, figsize=(9.2, 4.7), layout="constrained"
)
x = np.r_[time[0] - 1 / 60, (time[:-1] + time[1:]) / 2, time[-1] + 1 / 60] + 6.5
for i, ax in enumerate(axes):
    mask = valid if i == 0 else valid & (meas > 50)
    ax.set_facecolor("#c6cdce")
    if i == 1:
        ax.pcolormesh(
            x,
            10 ** edges[4:],
            np.where(valid & ~mask, 1, np.nan).T,
            cmap=ListedColormap(["#e7dfce"]),
            shading="flat",
        )
    q = ax.pcolormesh(
        x,
        10 ** edges[4:],
        np.where(mask, err, np.nan).T,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5),
    )
    ax.set(yscale="log", ylabel="Diameter (nm)", xlim=(0, 33))
    ax.set_yticks([4, 10, 30, 60], labels=["4", "10", "30", "60"])
    ax.set_title(
        [
            "All available size and time bins",
            "Bins with measured density above 50 particles/cm³ per decade",
        ][i],
        loc="left",
        fontsize=11,
    )
axes[-1].set_xlabel("Hours from run start")
cb = fig.colorbar(q, ax=list(axes), ticks=[-1, 0, 1], shrink=0.85)
cb.ax.set_yticklabels(["0.1×", "1×", "10×"])
cb.set_label("Model / observed\n(with unit offset)")
fig.suptitle(
    "CLOUD run 1833 · attractive capture at equal particle entry", fontsize=14
)
fig.legend(
    handles=[
        Patch(color="#e7dfce", label="Below display cutoff (≤50)"),
        Patch(color="#c6cdce", label="Comparison unavailable"),
    ],
    loc="outside lower center",
    ncol=2,
    frameon=False,
)
FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES / "threshold-comparison.png", dpi=150)
