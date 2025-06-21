# %%
import matplotlib.pyplot as plt

# %%
import numpy as np
from particula.util.materials.surface_tension import get_surface_tension
from particula.util.materials.vapor_pressure import get_vapor_pressure


# %% silica mineral
T_silica = 2500  # K
# value_silica = get_surface_tension("Si", T_silica).item()
# print(f"Surface tension of silica at {T_silica} K: {value_silica:.5f} N/m")


temp_array = np.linspace(50, 5000, 500)
# Major mineral components found in sand and atmospheric dust
components = {
    "Silicon (Si)": "Si",
    "Silicon dioxide (SiO₂)": "SiO2",
    "Aluminium oxide (Al₂O₃)": "Al2O3",
    "Iron(III) oxide (Fe₂O₃)": "Fe2O3",
    "Calcium carbonate (CaCO₃)": "CaCO3",
}
colors = plt.cm.tab10(np.linspace(0, 1, len(components)))

# %%
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
for (label, formula), color in zip(components.items(), colors):
    try:
        vp = get_vapor_pressure(formula, temp_array)
        st = get_surface_tension(formula, temp_array)
    except Exception as err:
        print(f"Skipping {label}: {err}")
        continue
    ax1.plot(
        temp_array,
        vp / 1e6,
        color=color,
        linestyle="-",
        label=f"{label} $P_{{sat}}$",
    )
    ax2.plot(temp_array, st, color=color, linestyle="--", label=f"{label} σ")
ax1.set_yscale("log")
ax1.set_xlabel("Temperature (K)")
ax1.set_ylabel("Vapor Pressure (MPa)", color="r")
ax2.set_ylabel("Surface Tension (N/m)", color="b")
ax1.tick_params(axis="y", labelcolor="r")
ax2.tick_params(axis="y", labelcolor="b")
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper center",
    fontsize="x-small",
    ncol=2,
)
ax1.set_title("Vapor Pressure and Surface Tension of Sand/Dust Components")
ax1.grid()
fig.tight_layout()
fig.savefig("vapor_press_surftens_sand_dust.png", dpi=300)
