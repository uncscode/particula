# %%
from thermo import SurfaceTension
from chemicals import CAS_from_any


# %%
# Get CAS number for aluminum (liquid form)
cas = CAS_from_any("Al")

# Create surface tension object
gamma = SurfaceTension(CASRN=cas)

# gamma.stablepoly_fit_coeffs
# Compute surface tension at a given temperature, e.g., 1000 K
T = 1000  # K
value = gamma.T_dependent_property(T)  # in N/m

print(f"Surface tension of liquid aluminum at {T} K: {value:.5f} N/m")
print("Temperature range:", gamma.Tmin, "to", gamma.Tmax)

# %% silica mineral
cas_silica = CAS_from_any("Si")
gamma_silica = SurfaceTension(CASRN=cas_silica)
# Compute surface tension at a given temperature, e.g., 1000 K
print("Temperature range:", gamma_silica.Tmin, "to", gamma_silica.Tmax)
T_silica = 1500  # K
value_silica = gamma_silica.T_dependent_property(T_silica)  # in N/m
print(f"Surface tension of silica at {T_silica} K: {value_silica:.5f} N/m")

# %%
