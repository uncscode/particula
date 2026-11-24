"""Water uptake, measured count and size-resolved residuals from three retained runs."""

from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import pandas as pd
from particula import plotting

plt.style.use("particula.default")


def plot(directory, spectra, output):
    directory = Path(directory)
    study = Path(__file__).resolve().parents[1]
    with np.load(study / "spectra.npz") as data:
        t, d, e, y, stage = (
            data[k] for k in ["time", "diameter", "edges", "observed", "stage"]
        )
    missing = pd.read_csv(study / "lamp_budget.csv").small_missing.to_numpy()
    ix = np.flatnonzero((stage >= 9) & (stage <= 18))
    o = y[ix]
    with np.load(spectra) as data:
        files = {
            key: data[key]
            for key in data
            if key.startswith("test_288_aiomfac_")
            and key.endswith("_attractive")
        }
    paper_time = t[ix] + 6.5
    tail = np.maximum(0, e[1:] - np.maximum(e[:-1], np.log10(9.7)))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    curve = pd.read_csv(directory / "activity.csv")
    a = ax[0, 0]
    a.plot(
        curve.diameter_nm,
        100 * curve.water_mass_fraction,
        label="AIOMFAC liquid",
        color=plotting.COLORS["teal"],
    )
    a.plot(
        curve.diameter_nm,
        100 * curve.original_water_mass_fraction,
        label="Original hydration",
        color=plotting.COLORS["indigo"],
        ls="--",
    )
    a.set(
        xscale="log",
        xlabel="Wet diameter (nm)",
        ylabel="Water mass (%)",
        title="Water uptake during growth",
    )
    a.legend(frameon=False)
    a = ax[0, 1]
    a.plot(
        paper_time,
        o @ tail,
        color=plt.rcParams["text.color"],
        ls="-",
        label="Measured",
        lw=1.5,
    )
    for r, c, ls in [
        (1.0, plotting.COLORS["teal"], "-"),
        (0.5, plotting.COLORS["rust"], "--"),
        (0.0, plotting.COLORS["berry"], ":"),
    ]:
        key = f"test_288_aiomfac_{r}_attractive"
        a.plot(
            paper_time,
            files[key] @ tail,
            color=c,
            ls=ls,
            lw=1.5,
            label=f"{r:.0%} retained",
        )
    a.set(
        xlabel="Time from run start (hours)",
        ylabel="Particles ≥9.7 nm (cm⁻³)",
        title="Size-selected concentration",
    )
    a.legend(loc="upper left")
    for a, r in zip(ax[1], [1.0, 0.5]):
        pred = files[f"test_288_aiomfac_{r}_attractive"]
        residual = (pred - o).copy()
        residual[np.ix_(missing[ix], np.arange(7))] = np.nan
        a.grid(False)
        img = a.pcolormesh(
            paper_time,
            d[4:],
            residual[:, 4:].T,
            cmap=plotting.colormap("difference").with_extremes(bad="#e3e7e6"),
            shading="nearest",
            norm=SymLogNorm(100, vmin=-40000, vmax=40000),
        )
        a.set(
            yscale="log",
            xlabel="Time from run start (hours)",
            ylabel="Measured diameter (nm)",
            title=f"Density residual · {r:.0%} retained",
        )
    fig.colorbar(img, ax=ax[1], label="Particles cm⁻³ per decade")
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    plot(directory, directory / "plot-spectra.npz", sys.argv[1])
