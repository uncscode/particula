# %% [markdown]
# Saffman and Turner calculated how initially identical drops merge in uniform turbulence. Can a constant collision rate match both the mean mass and the rare large drops? The [study report](index.md) compares their equations and Figure 2 with Particula.
#
# ## Run the shear-only cloud
#
# Start with 600 drops/cm³ and 1.5 g/m³ of water. The turbulent dissipation stays fixed while each merger changes the radii available for the next encounter. Seven CUDA ensembles vary the box volume, seed, time step or observation time. The CPU reference integrates the authors’ discrete mass balance and evaluates the constant-rate solution independently.
#
# Use the downloadable ZIP for a standalone run with Particula, Warp, NumPy, SciPy, pandas and Matplotlib. The documentation renderer can also replot its saved numerical results on CPU. Inspect the exact eightfold class at the last time in Figure 2 before comparing it with the paper’s separate cumulative thresholds.

# %%
import matplotlib.pyplot as plt
from scripts.shear_drop_spectrum.plots import spectrum_figure

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.shear_drop_spectrum.study import simulate

    report = simulate()

endpoint = next(
    row
    for row in report["summary"]
    if row["case"] == "large_fine_pooled" and row["K"] == 0.09
)
print(f"Device: {report['device']}")
print(
    f"Exact eightfold drops per million initial drops: {1e6 * endpoint['n8']:.0f}"
)
print(
    f"Constant rate fitted to mean mass: {1e6 * endpoint['matched_constant']:.0f}"
)
for row in report["physical"]:
    print(
        f"At {row['minutes']} min, at least {row['mass_at_least']} initial masses: "
        f"{row['cuda_per_L']:.0f} drops/L"
    )

# %% [markdown]
# ## Compare the spectrum and the chance of seeing a rare drop
#
# The left panel overlays the independent equations, particle-resolved means with 95% Monte Carlo intervals, and digitized points from the authors’ theoretical curve. The right panel counts independent boxes containing an exact eightfold drop. Wilson intervals show the sampling uncertainty in that fraction.

# %%
spectrum_figure(report)
plt.show()

# %% [markdown]
# Particula tracks the size-dependent shear solution through the eightfold mass class. Matching the final mean mass with a constant rate still gives only about one sixth as many drops in that class. Even at the last time, more than half the larger boxes contained zero exact eightfold drops. To test this prediction in a chamber, measure the initial width and sampling volume, then count the upper tail independently of the mean.
