# %% [markdown]
# # Compare measured water uptake and ion peaks with the two fitted pathways
#
# Two pathways predict similar water uptake after sulfur dioxide exposure.
# This example pairs each prediction with its own measurement at three hours.
# Growth was measured at 200 nm and 80% readout relative humidity. The
# chromatograms were measured at 100 nm under the same reaction conditions.

# %%
import csv
from pathlib import Path


def data_directory():
    """Locate the figure data in a checkout or the downloaded example."""
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for directory in (start, *start.parents):
            candidate = directory / "docs/data/xiong-sea-spray"
            if (candidate / "growth-time.csv").is_file():
                return candidate
    raise FileNotFoundError("Extract the full example download with its data.")


def at_three_hours(filename):
    with (data_directory() / filename).open() as source:
        return next(
            row for row in csv.DictReader(source) if float(row["time_h"]) == 3
        )


# %% [markdown]
# A diameter growth factor is the wet diameter divided by the dry diameter.
# Compare the residual with the spread the authors reported for that point.

# %%
growth = at_three_hours("growth-time.csv")
measured = float(growth["gf"])
spread = float(growth["spread"])
models = {
    "Direct with retained chloride": "retention_neutral",
    "Parallel": "parallel_neutral",
}
print(f"Measured growth factor {measured:.2f} ± {spread:.2f}")
for label, key in models.items():
    prediction = float(growth[key])
    print(f"{label}: {prediction:.3f}, difference {prediction - measured:+.3f}")

# %% [markdown]
# Read the ratio of sulfate to chloride peak areas at the same exposure time.
# The bounds come from integrating the edges of the printed traces.
# They measure uncertainty in reading the graph.

# %%
ion = at_three_hours("ion-time.csv")
print(
    f"Measured area ratio {float(ion['late_to_early_peak_area_ratio']):.2f}, "
    f"reading range {float(ion['raster_edge_min']):.2f}–{float(ion['raster_edge_max']):.2f}"
)
for label, key in models.items():
    print(f"{label}: {float(ion[key]):.2f}")

# %% [markdown]
# Both growth predictions fall inside the reported spread. Both ion predictions
# exceed the upper graphical bound. The late peaks motivated the retained-
# chloride model, so this comparison is retrospective. Early measurements of
# particle sulfur(IV) could distinguish the two reaction histories directly.
