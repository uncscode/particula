# %% [markdown]
# Can a simulation match a cloud droplet gain while starting with too few
# droplets? Replay the localized dry-inlet case from the [study](index.md).
# Use the ZIP download with Particula, NumPy, Matplotlib, Warp, a CUDA GPU
# and the CUDA compiler (`nvcc` on PATH, or set `NVCC`).
#
# ## Compare each treatment with its own continuously seeded background
#
# Both seed loadings evolve for six hours from empty, saturated air. A control
# and a dry-air branch each continue six hours from that same background.
# The two-layer model uses a fixed four-second NaCl water-transfer table.
# The local upper counter records the final hour. Each stage checks number
# and water conservation. Published measurements are read only for comparison.

# %%
import matplotlib.pyplot as plt

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.pi_chamber_lidar.replay import simulate

    report = simulate()

print(f"CUDA device: {report['device']}")
for row in report["states"]:
    print(
        f"{row['case']}: {row['model_cloud']:.1f} cloud droplets/cm³, "
        f"{row['model_water']:.1f} mg/m³ visible liquid water"
    )

# %% [markdown]
# ## Read the change alongside the absolute population
#
# Both panels show cloud droplets from 2 to 40.679 µm in polluted air.
# The first counts the background. The second subtracts that background
# from the dry-air treatment. A common zero within each panel preserves
# the ordering, and the numerical labels give the concentrations.

# %%
states = {row["case"]: row for row in report["states"]}
background = states["polluted-B"]
treated = states["polluted-E"]
quantities = ("measured_cloud", "model_cloud")
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), layout="constrained")
for axis, title, values in zip(
    axes,
    ("Background", "Gain after dry air enters"),
    (
        [background[k] for k in quantities],
        [treated[k] - background[k] for k in quantities],
    ),
    strict=True,
):
    bars = axis.bar(
        ["Measured", "Particula"], values, color=["#243E43", "#135963"]
    )
    axis.bar_label(bars, fmt="%.0f", padding=4)
    axis.set(
        title=title, ylabel="Cloud droplets (cm⁻³)", ylim=(0, max(values) * 1.2)
    )
    axis.spines[["top", "right"]].set_visible(False)
plt.show()

# %% [markdown]
# The simulated polluted gain is close to the measured gain even though
# the simulation retains fewer than one-third of the measured background
# cloud droplets. The report tests this interpretation against complete
# counter spectra, lidar profiles and matched side-inlet seed histories.
