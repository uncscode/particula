# %%
import matplotlib.pyplot as plt

# %%
from typing import Union, Optional
import numpy as np
from numpy.typing import NDArray
from particula.util.materials.surface_tension import get_surface_tension
from particula.util.materials.vapor_pressure import get_vapor_pressure

# Aluminium example
T = 1000  # K
value = get_surface_tension("Al", T).item()  # in N/m

print(f"Surface tension of liquid aluminum at {T} K: {value:.5f} N/m")
# print("Temperature range:", gamma.Tmin, "to", gamma.Tmax)

# %% silica mineral
T_silica = 2500  # K
# value_silica = get_surface_tension("Si", T_silica).item()
# print(f"Surface tension of silica at {T_silica} K: {value_silica:.5f} N/m")


def evaluate_surface_tension(
    temperature: Union[float, NDArray[np.float64]],
    Tc: float,
    sigma: NDArray[np.float64],
    n: NDArray[np.float64],
    Tmin: Optional[float] = None,
    Tmax: Optional[float] = None,
    transition_width: float = 100.0,
) -> Union[float, NDArray[np.float64]]:
    """
    Evaluate surface tension using extended DIPPR-106 correlation.

    Applies:
      - Clipping below Tmin (value held constant).
      - Smooth sigmoid decay to zero above Tmax.

    Arguments:
        temperature: Scalar or array of temperatures [K].
        Tc: Critical temperature [K].
        sigma: Array of sigma coefficients.
        n: Array of exponent coefficients.
        Tmin: Lower validity limit [K].
        Tmax: Upper validity limit (e.g., boiling point) [K].
        transition_width: Temperature range over which surface tension
            decays to zero [K], default is 100 K.

    Returns:
        Surface tension [N/m].
    """
    T = np.asarray(temperature, dtype=np.float64)
    scalar_input = np.isscalar(temperature)

    # Evaluate surface tension at adjusted temperature (clip below Tmin)
    T_eval = T.copy()
    if Tmin is not None:
        T_eval = np.maximum(T_eval, Tmin)

    Tr = 1 - T_eval / Tc
    Tr = Tr[..., np.newaxis]  # for broadcasting

    sigma = np.asarray(sigma, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    sigma_val = np.sum(sigma * Tr**n, axis=-1)

    # Hold value constant below Tmin
    if Tmin is not None:
        sigma_Tmin = evaluate_surface_tension(
            Tmin, Tc, sigma, n, Tmin=None, Tmax=None
        )
        sigma_val = np.where(T < Tmin, sigma_Tmin, sigma_val)

    # Apply sigmoid decay to 0 above Tmax
    if Tmax is not None:
        decay = 1.0 / (1.0 + np.exp((T - Tmax) / transition_width))
        sigma_val *= decay

    return sigma_val.item() if scalar_input else sigma_val


coeffs = {
    "Tc": 5159.0,
    "sigma": np.array(
        [0.8155472195894874, 0.5565923686778642, -0.10177569593245787]
    ),
    "n": np.array([1.058540215655986, 1.0583393302743886, 1.632607479918296]),
    "Tmin": 1273.0,
    "Tmax": 1923.0,
    "transition_width": 10.0,
}

T_test = np.linspace(1000, 3500, 300)
sigma_vals = evaluate_surface_tension(T_test, **coeffs)


# Plotting the results

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(T_test, sigma_vals, label="Surface Tension", color="blue")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Surface Tension (N/m)")
ax.set_title("Surface Tension vs Temperature")
ax.grid()
ax.legend()
plt.tight_layout()
plt.show()
# %%

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
plt.show()

# %%
