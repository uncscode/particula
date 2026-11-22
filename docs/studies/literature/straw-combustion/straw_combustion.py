# %% [markdown]
# ## Replay the fitted reactor at Zeuthen's six feed compositions
#
# The [study](index.md) asks how particle counts respond when SO₂, oxygen
# or potassium chloride changes in a cooling synthetic flue gas. Use the
# example ZIP for the study-specific reactor equations, observed outlet
# values, reconstructed temperatures, and fitted parameters. The original
# calibration used CUDA. This CPU replay fixes the coefficients and
# integrates the same forward model on two radial paths for each feed.
#
# Compare two fits of the model with all cluster pathways available.
# One fit used all 85 measurements. The other used 56, reserving the
# high-oxygen and high-KCl aerosol feeds among its 29 withheld readings.
# Each fit used one parameter set across all reactor conditions.

# %%
import matplotlib.pyplot as plt
from scripts.straw_combustion.replay import run_replay

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    rows, _ = run_replay(device="cpu", fits=("all_combined", "split_combined"))
    report = {}
    for fit in ("all_combined", "split_combined"):
        fitted = [row for row in rows if row["fit"] == fit]
        withheld = [r for r in fitted if not r["in_original_train"]]
        report[fit] = {
            "all_in_band": sum(abs(r["residual_bands"]) <= 1 for r in fitted),
            "reserved_in_band": sum(
                abs(r["residual_bands"]) <= 1 for r in withheld
            ),
            "counts": {
                r["case"]: [r["measured"], r["tolerance"], r["predicted"]]
                for r in fitted
                if r["dataset"] == "aerosol" and r["target"] == "count"
            },
        }

# %% [markdown]
# ## Compare outlet particle counts on a common scale
#
# Each marker is one reactor outlet measurement or calculation. The
# vertical whiskers show the reported experimental scatter. The first
# three feeds form the sulfur series. Comparing the two fitted markers
# for the high-oxygen and high-KCl feeds shows what changed when those
# measurements were included in calibration.

# %%
conditions = (
    "No SO2",
    "Reference",
    "High SO2",
    "Low O2",
    "High O2",
    "High KCl",
)
labels = ("No SO₂", "Reference", "High SO₂", "Low O₂", "High O₂", "High KCl")
all_counts = report["all_combined"]["counts"]
split_counts = report["split_combined"]["counts"]
fig, ax = plt.subplots(figsize=(7.6, 3.5), layout="constrained")
ax.errorbar(
    [i - 0.16 for i in range(len(conditions))],
    [all_counts[c][0] for c in conditions],
    yerr=[all_counts[c][1] for c in conditions],
    fmt="o",
    color="#243E43",
    capsize=3,
    label="Zeuthen ± scatter",
)
ax.scatter(
    range(len(conditions)),
    [all_counts[c][2] for c in conditions],
    marker="s",
    color="#135963",
    label="85-reading fit",
)
ax.scatter(
    [i + 0.16 for i in range(len(conditions))],
    [split_counts[c][2] for c in conditions],
    marker="x",
    color="#A44D21",
    label="56-reading fit",
)
ax.set_xticks(range(len(conditions)), labels)
ax.set(ylabel="Outlet count (million particles per cm³)", ylim=(0, None))
ax.legend(frameon=False)
ax.spines[["right", "top"]].set_visible(False)
plt.show()

# %% [markdown]
# ## Check agreement beyond the six counts
#
# Count a reading inside its study working band when the absolute residual
# divided by its assigned tolerance is at most one. The full calibration
# matches both high-feed counts. The held-out comparison also tests dry-gas
# concentrations, particle sizes and masses across different conditions.

# %%
print(
    f"All-data calibration: {report['all_combined']['all_in_band']}/85 in band"
)
print(
    f"56-reading calibration: {report['split_combined']['reserved_in_band']}/29 withheld in band"
)
