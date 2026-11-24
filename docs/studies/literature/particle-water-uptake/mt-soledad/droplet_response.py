# %% [markdown]
# # Compare count loss and droplet growth at Mt. Soledad
#
# A charcoal denuder at Mt. Soledad removed trace gases from one sampling
# path before particles entered a cloud condensation nuclei counter. A
# comparison path bypassed the denuder. Would a common growth delay, fitted
# to the decrease in counts at 0.4% supersaturation, also explain which
# droplet sizes reached the detector at 0.8%?
#
# Read the saved optical channels for particles selected at 120–250 nm and
# the date-weighted count and size estimates. Both model parameters were
# fitted on 12 dates at 0.4% and held fixed on seven other dates at 0.8%.
# This CPU example reads those results. The [field study](index.md)
# describes the particle simulations and instrument timing.
#
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from particula import plotting

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    table_dir = Path.cwd()
    if not (table_dir / "optical-spectrum.csv").is_file():
        table_dir = Path(__file__).resolve().parent
    if not (table_dir / "optical-spectrum.csv").is_file():
        table_dir = (
            Path(__file__).resolve().parents[5] / "studies/mt-soledad/figures"
        )
    report = {
        "spectrum": pd.read_csv(table_dir / "optical-spectrum.csv").to_dict(
            "records"
        ),
        "summary": pd.read_csv(table_dir / "optical-summary.csv").to_dict(
            "records"
        ),
    }

# %% [markdown]
# ## Check the total response before inspecting the channels
#
# First divide each branch's optical counts by its concurrent condensation
# particle counter signal. Express both results relative to the bypass total.
# A ratio less than one indicates that fewer particles were counted on the
# denuded path per reference particle. Compare this ratio for the two
# settings, then check the change in mean wet diameter among particles that
# remained visible at 0.8%.
#
# %%
summary = pd.DataFrame(report["summary"])
names = {
    "measured": "Measured",
    "scalar": "Size-independent count loss",
    "delay": "Common growth delay",
}
for series, label in names.items():
    measures = summary[summary.series == series].set_index(["split", "metric"])
    calibration = measures.at[("train", "count_ratio"), "mean"]
    held = measures.at[("test", "count_ratio"), "mean"]
    shift = measures.at[("test", "wet_shift_um"), "mean"]
    print(
        f"{label}: count ratio {calibration:.3f} at 0.4%, {held:.3f} at 0.8%; "
        f"held diameter change {shift:+z.3f} µm"
    )

# %% [markdown]
# ## Locate the missing counts in the held outlet spectrum
#
# Plot the complete measured optical bins at 0.8%. The height is the fraction
# of the total bypass response in each bin, so the sum of bin heights gives
# the response ratio. Draw the two predictions against the same measured
# bypass spectrum. A delay can move particles into smaller optical bins or
# below the lower threshold. Size-independent count loss scales every bin by
# the same factor.
#
# %%
spectrum = pd.DataFrame(report["spectrum"])
held_spectrum = spectrum[spectrum.split == "test"].sort_values("channel")
edges = np.r_[held_spectrum.lower_um.iloc[0], held_spectrum.upper_um]
plt.style.use("particula.default")
fig, ax = plotting.subplots(size="wide")
for column, label, color, stroke in [
    ("bypass", "Measured bypass", "#7c8985", "-"),
    ("denuded", "Measured denuded path", "#18322d", "-"),
    ("scalar", names["scalar"], plotting.COLORS["teal"], "--"),
    ("delay", names["delay"], plotting.COLORS["rust"], "-."),
]:
    ax.stairs(
        held_spectrum[column], edges, color=color, linestyle=stroke, label=label
    )
ax.set(
    xlim=(0.5, 10),
    ylim=(0, 0.60),
    xlabel="Wet outlet diameter (µm)",
    ylabel="Counts per bin / bypass total",
)
plotting.legend(ax)
plotting.fit(fig)
plt.show()

# %% [markdown]
# Both models reproduce the calibration count ratio of 0.796. At the higher
# setting, the common delay predicts almost complete detection and shifts
# the mean survivor diameter by −1.54 µm. The measured shift was −0.013 µm
# and the count ratio was 0.776. A size-independent reduction preserves the
# measured spectral shape more closely. Particle transmission, counting
# efficiency and a subset that grows too slowly to reach the detector can
# each produce this pattern. Varying the growth time while independently
# counting the particles emerging from both inlets would distinguish them.
