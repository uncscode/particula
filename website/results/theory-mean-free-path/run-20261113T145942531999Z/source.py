# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Mean Free Path
#
# Minimal reproducible cell to verify execution.

# %%
import particula as par

mean_free_path = par.gas.get_molecule_mean_free_path(
    molar_mass=28.97e-3,
    temperature=298.15,
    pressure=101325,
    dynamic_viscosity=par.gas.get_dynamic_viscosity(temperature=298.15),
)
print(mean_free_path)

