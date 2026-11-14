# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: particula_dev312
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dynamic Viscosity
#
# The dynamic viscosity is a property of the fluid, defining the resistance of the fluid to its own movement. The dynamic viscosity is calculated using the Sutherland formula ([reference](https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula)). The function can be found and is documented in `util.dynamic_viscosity.py`. It takes inputs of `temperature`, `reference_viscosity`, `reference_temperature`, and `sutherland_constant`. It returns a value for the dynamic viscosity at those variables. At default conditions (298.15 K and 101325 Pa), the dynamic viscosity is approximately 1.84e-5 kg/m/s. The Sutherland formula is
#
# $$
# \mu = \frac{\mu_{0}\, (T/T_{0})^{3/2}\, (T_{0} + C)}{C + T}
# $$
#
# where $\mu$ is the dynamic viscosity, $\mu_{0}$ is the reference dynamic viscosity, $T$ is temperature, $T_{0}$ is the reference temperature, and $C$ is the Sutherland constant.

# %%
# In Colab uncomment the following command to install particula:
# #!pip install particula[extra] --quiet
import particula as par


air_dynamic_viscosity = par.gas.get_dynamic_viscosity(
    temperature=298.15
)  # will produce approx 1.84e-5 kg/m/s

print(
    f"Air dynamic viscosity at 298.15 K is {air_dynamic_viscosity:.2e} kg/m/s"
)
