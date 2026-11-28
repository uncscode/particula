# %% [markdown]
# # Recover the half-second large-droplet shortfall
#
# The REACh hologram samples a small region of a cloud chamber. To compare
# like with like, take the same recording interval, wet diameters and
# concentration units from the observed and simulated spectra.

# %%
import csv
import io
from pathlib import Path
from zipfile import ZipFile


def figure_archive():
    """Find the public tables in a checkout or an extracted example download."""
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for parent in (start, *start.parents):
            for relative in (
                "figure-data.zip",
                "website/public/study-data/reach-droplets/figure-data.zip",
            ):
                candidate = parent / relative
                if candidate.is_file():
                    return candidate
    raise FileNotFoundError(
        "Extract the full example download with its figure tables."
    )


# %% [markdown]
# Each row is the number density in a one-micrometre bin. Integrate the
# 50–65 µm band and average the two quarter seconds centered at 5.125 and
# 5.375 s. The forward calculation used the early hologram to initialize
# the droplets and the measured gas conditions to advance them.

# %%
from collections import defaultdict

quarter_totals = defaultdict(float)
with (
    ZipFile(figure_archive()) as archive,
    archive.open("spectra.csv") as stream,
):
    for row in csv.DictReader(io.TextIOWrapper(stream)):
        center = float(row["t_mid_s"])
        diameter = float(row["diameter_um"])
        if (
            row["run"] == "unforced"
            and 5 <= center < 5.5
            and 50 <= diameter < 65
        ):
            quarter_totals[row["series"], center] += (
                float(row["density_cm-3_um-1"]) * 1.0
            )

means = {}
for series in ("Hologram", "Forward physics", "Spatial update"):
    means[series] = (
        quarter_totals[series, 5.125] + quarter_totals[series, 5.375]
    ) / 2
    print(f"{series}: {means[series]:.3f} droplets/cm³")
print(f"Measured / forward: {means['Hologram'] / means['Forward physics']:.2f}")

# %% [markdown]
# The forward model misses about three quarters of the large droplets in
# this interval. The spatial update uses spectra on both sides of the burst
# to fit its controls. Even with earlier camera-guided redistribution it
# recovers less than half the observed concentration.
