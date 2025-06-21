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
vap_pressures = get_vapor_pressure("silicon", temp_array)
sigma = get_surface_tension("silicon", temp_array)

# %%
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
ax1.plot(temp_array, vap_pressures / 1e6, "r-", label="Vapor Pressure (MPa)")
ax2.plot(temp_array, sigma, "b-", label="Surface Tension (N/m)")
ax1.set_yscale("log")
ax1.set_xlabel("Temperature (K)")
ax1.set_ylabel("Vapor Pressure (MPa)", color="r")
ax2.set_ylabel("Surface Tension (N/m)", color="b")
ax1.tick_params(axis="y", labelcolor="r")
ax2.tick_params(axis="y", labelcolor="b")
ax1.set_title("Vapor Pressure and Surface Tension of Silicon")
ax1.grid()
fig.tight_layout()
fig.savefig("vapor_pressure_surface_tension_silicon.png", dpi=300)
