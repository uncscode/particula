# %% Replot and verify the CLOUD study from the retained results. No GPU needed.
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, SymLogNorm

out = Path(sys.argv[1]).resolve()
z = np.load(out / "spectra.npz")
idx = z["test_indices"]
d = z["diameter"]
e = z["edges"]
widths = np.diff(e)
t = z["time"][idx]
stage = z["stage"][idx]
obs = z["observed"][idx]
p = z["test_selected"]
fine = np.load(out / "mechanism.npz")["density"][[0, 2, 3, 5]]
from matplotlib.ticker import ScalarFormatter

plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
colors = ["#394E75", "#08776B", "#C66A1B", "#9F417A"]
st = np.arange(9, 19)


def annotate(ax):
    for s in st[1:]:
        ax.axvline(
            t[stage == s][0], alpha=0.32, color="#626262", lw=0.65, ls=":"
        )


def time_axis(ax):
    ax.set_xlabel("Archive time (h)")


def mid_edges(x):
    return np.r_[
        x[0] - (x[1] - x[0]) / 2,
        (x[:-1] + x[1:]) / 2,
        x[-1] + (x[-1] - x[-2]) / 2,
    ]


def panel(ax, y, label, kind, low=False):
    a = y.copy()
    if not low:
        a[:, :4] = np.nan
    if kind == "density":
        a = np.log10(1 + a)
        norm = Normalize(0, 5)
        cmap = "viridis"
    else:
        norm = SymLogNorm(linthresh=100, vmin=-45000, vmax=45000)
        cmap = "RdBu_r"
    mesh = ax.pcolormesh(
        mid_edges(t),
        10**e,
        a.T,
        shading="flat",
        norm=norm,
        cmap=cmap,
        rasterized=True,
    )
    ax.set_yscale("log")
    ax.set_yticks([2, 4, 10, 20, 50])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(10 ** e[0], 10 ** e[-1])
    ax.set_title(label, loc="left", fontsize=10.5)
    if not low:
        ax.axhspan(10 ** e[0], 10 ** e[4], color="#eeeae3", zorder=1)
        ax.text(
            t[0] + 0.2,
            1.9,
            "Below scored range",
            color="#5A554E",
            va="center",
            fontsize=8,
        )
    return mesh


fig, axes = plt.subplots(
    3, 2, figsize=(11, 8.6), sharex=True, sharey=True, layout="constrained"
)
panel(
    axes[0, 0],
    obs,
    "Measured spectrum  ·  merged mobility instruments",
    "density",
    True,
)
im = panel(axes[0, 1], fine[0], "Vapor transfer  ·  measured acid", "density")
panel(
    axes[1, 0],
    fine[1],
    "Collision-enhanced growth  ·  measured acid",
    "density",
)
panel(
    axes[1, 1],
    fine[3],
    "Collision-enhanced growth  ·  published chemistry",
    "density",
)
res = panel(
    axes[2, 0],
    fine[1] - obs,
    "Enhanced minus measured spectrum  ·  measured acid",
    "residual",
)
panel(
    axes[2, 1],
    fine[3] - obs,
    "Enhanced minus measured spectrum  ·  published chemistry",
    "residual",
)
for ax in axes[:, 0]:
    ax.set_ylabel("Physical diameter (nm)")
for ax in axes[-1]:
    time_axis(ax)
fig.colorbar(im, ax=axes[:2, :], label="log₁₀(1 + density*)", shrink=0.8)
fig.colorbar(res, ax=axes[-1, :], label="Signed density*", shrink=0.9)
fig.suptitle(
    "How the CLOUD size distribution evolved during excimer stages 09–18",
    fontsize=15,
)
fig.supxlabel(
    "*Particle number per cm³ per decade of physical diameter. Stages 09–17 and the observed portion of stage 18.",
    fontsize=9,
)
fig.savefig(out / "spectrum.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(
    3,
    1,
    figsize=(10, 8.8),
    layout="constrained",
    gridspec_kw={"height_ratios": [1, 1.2, 1]},
)
f = pd.read_csv(out / "lamp_budget.csv").iloc[idx]
ax[0].scatter(t, 1.15 * f.ufra_raw / 1e6, s=5, c="black", label="UFRA × 1.15")
ax[0].plot(
    t, f.author_chem / 1e6, color=colors[1], label="Published chemistry", lw=1.4
)
ax[0].plot(
    t,
    f.lamp_coupled_acid / 1e6,
    color=colors[3],
    ls="--",
    label="Reduced chemistry + simulated sink",
    lw=1.3,
)
ax[0].set_ylabel("Sulfuric acid\n(10⁶ molecules/cm³)")
annotate(ax[0])
ax[0].legend(frameon=False, ncol=3, fontsize=9)
for yv, col, ls, label in [
    (obs, "black", "-", "Measured"),
    (fine[0], colors[0], "--", "Vapor, measured acid"),
    (fine[1], colors[1], "-", "Enhanced, measured acid"),
    (fine[3], colors[2], "-.", "Enhanced, published chemistry"),
]:
    ax[1].plot(t, yv[:, 4:] @ widths[4:], color=col, lw=1.5, ls=ls, label=label)
ax[1].set_ylabel("3.41–64.68 nm\nparticles/cm³")
annotate(ax[1])
ax[1].legend(frameon=False, ncol=2, fontsize=9)
time_axis(ax[1])
time_axis(ax[0])
tab = pd.read_csv(out / "stages.csv")
g = tab.query('run=="fine"')
for c, col, marker, ls, label in [
    (0, colors[0], "s", "--", "Vapor, measured acid"),
    (1, colors[1], "o", "-", "Enhanced, measured acid"),
    (3, colors[2], "D", "-.", "Enhanced, chemistry"),
]:
    sub = g[g.case == c]
    ax[2].plot(
        sub.stage, sub.score, ls=ls, marker=marker, color=col, label=label
    )
ax[2].set(
    xticks=st,
    xlabel="Excimer stage (18 includes early dark decay only)",
    ylabel="Spectral error (dex)",
)
ax[2].legend(frameon=False, fontsize=9, ncol=3)
fig.savefig(out / "stages.png", dpi=160)
plt.close(fig)
# A time-block uncertainty calculation preserving all same-spectrum observations.
valid = stage < 18
valid[0] = False
err = (
    abs(np.log10(1 + p[:, :, 4:]) - np.log10(1 + obs[:, 4:])) @ widths[4:]
) / widths[4:].sum()
expected = (
    pd.read_csv(out / "scores.csv")
    .query("run=='test' and case<4")
    .score.to_numpy()
)
np.testing.assert_allclose(expected, err[:, valid].mean(1), atol=1e-12)
gain = err[0] - err[1]
acid = err[3] - err[1]
group = stage[valid]
nt = np.array([(group == s).sum() for s in st[:-1]])
rng = np.random.default_rng(1833)
draw = rng.integers(0, 9, (20000, 9))


def summarize(v):
    sums = np.array([v[valid][group == s].sum() for s in st[:-1]])
    boot = sums[draw].sum(1) / nt[draw].sum(1)
    return dict(
        mean=float(v[valid].mean()),
        stage_resampling=np.quantile(boot, [0.025, 0.975]).tolist(),
        leave_one_stage=[float(v[valid][group != s].mean()) for s in st[:-1]],
    )


result = {
    "growth_gain_192": summarize(gain),
    "chemical_minus_measured_192": summarize(acid),
}
# Moving size-selected number threshold timing is a different spectral diagnostic.
tail = np.maximum(0, e[1:] - np.maximum(e[:-1], np.log10(9.7)))
err_fine = (
    abs(np.log10(1 + fine[:, :, 4:]) - np.log10(1 + obs[:, 4:])) @ widths[4:]
) / widths[4:].sum()
result["growth_gain_fine"] = summarize(err_fine[0] - err_fine[1])
result["chemical_minus_measured_fine"] = summarize(err_fine[3] - err_fine[1])
for cutoff in [1000, 5000]:
    result["first_tail_" + str(cutoff)] = {}
    for label, yv in zip(
        [
            "observed",
            "vapor_measured",
            "enhanced_measured",
            "vapor_chemistry",
            "enhanced_chemistry",
        ],
        [obs, *fine],
    ):
        values = yv @ tail
        above = np.flatnonzero((values[:-1] < cutoff) & (values[1:] >= cutoff))
        result["first_tail_" + str(cutoff)][label] = (
            None if not len(above) else float(t[above[0] + 1])
        )
(out / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
# Full experimental context. Modeled windows start from different measured spectra.
fig, ax = plt.subplots(
    3,
    1,
    figsize=(10, 8.5),
    layout="constrained",
    gridspec_kw={"height_ratios": [1.4, 1, 1]},
)
all_t = z["time"]
raw = z["observed"].copy()
bad = pd.read_csv(out / "forcing.csv").small_missing.to_numpy()
raw[bad, :7] = np.nan
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("#e1e1df")


def raster(a, times, val, name):
    mesh = a.pcolormesh(
        mid_edges(times),
        10**e,
        np.log10(1 + val.T),
        shading="flat",
        vmin=0,
        vmax=5,
        cmap=cmap,
    )
    a.set_yscale("log")
    a.set_yticks([2, 4, 10, 20, 50])
    a.yaxis.set_major_formatter(ScalarFormatter())
    a.set_ylabel("Diameter (nm)")
    a.set_title(name, loc="left", fontsize=10.5)
    return mesh


im = raster(
    ax[0],
    all_t,
    raw,
    "Full measured run  ·  gray cells mark the known small-instrument outage",
)
for x, s in [(-6.48, "Hg–Xe"), (4.29, "Excimer"), (16.20, "Later stages")]:
    ax[0].text(x + 0.2, 47, s, color="white", fontsize=10)
it = z["train_indices"]
raster(
    ax[1],
    all_t[it],
    raw[it],
    "Mercury–xenon training window  ·  measured spectrum",
)
train = z["train_selected"][1].copy()
train[:, :4] = np.nan
raster(
    ax[2],
    all_t[it],
    train,
    "Mercury–xenon training window  ·  measured acid and collision-enhanced growth",
)
ax[2].set_xlabel("Archive time (h)")
fig.colorbar(
    im, ax=ax, label="log₁₀(1 + particles/cm³ per diameter decade)", shrink=0.8
)
fig.savefig(out / "campaign-spectrum.png", dpi=150)
plt.close(fig)
