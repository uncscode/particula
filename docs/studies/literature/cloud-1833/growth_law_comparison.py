# %% [markdown]
# # Compare particle capture laws at the same incoming flux
#
# In CLOUD run 1833 each model receives the same number of incoming particles
# at 2.9 nm. Which model places them in the observed size bins? Use the
# saved CUDA stage summaries to compare the three capture laws. This example
# runs on CPU. The [main study](index.md) describes the chamber simulations.
#
# Run the downloaded example beside its stage-scores.csv, or run the script
# from the repository root to use the study's copy.
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from particula import plotting

table_path = Path("stage-scores.csv")
if not table_path.is_file():
    table_path = Path("studies/cloud-1833/full-transfer/stage-scores.csv")

scores = pd.read_csv(table_path)
excimer = scores.loc[
    (scores["segment"] == "excimer_led")
    & (scores["cells"] == 288)
    & scores["stage"].between(9, 17)
]

# %% [markdown]
# Each two-minute spectrum is normalized to its own total. The mismatch is
# the fraction that would need to move between size bins to match the
# measurements. Recover the pooled value by weighting the stage means by
# the number of comparable spectra in each stage.
# %%
by_stage = excimer.pivot(index="stage", columns="model", values="shape_pct")
laws = ["Vapor transfer", "Hard sphere", "Attraction"]
weights = excimer.pivot(index="stage", columns="model", values="shape_n")
pooled = (by_stage * weights).sum() / weights.sum()
for law in laws:
    print(f"{law}: {pooled[law]:.1f}% mismatch")

# %% [markdown]
# Compare the nine stages individually to see whether one stage dominates
# the pooled result. Markers help distinguish the models at low light,
# where the shape differences between collision treatments are smallest.
# %%
plt.style.use("particula.default")
fig, ax = plotting.subplots(size="wide")
for law, marker, color, stroke in zip(
    laws, ("o", "s", "^"), ("indigo", "rust", "teal"), ("-.", "--", "-")
):
    ax.plot(
        by_stage.index,
        by_stage[law],
        marker=marker,
        color=plotting.COLORS[color],
        linestyle=stroke,
        label=law,
    )
ax.set(
    xlabel="Excimer stage",
    ylabel="Size-distribution mismatch (%)",
    xticks=by_stage.index,
)
plotting.legend(ax)
plotting.fit(fig)
plt.show()

# %% [markdown]
# The model including attractive capture gives the closest shape in each
# stage. A separate diagnostic asks whether those particles reach larger
# sizes at the right time. The [particle-entry comparison](particle-entry.md)
# tests how the timing changes when newly formed particles arrive in bursts.
