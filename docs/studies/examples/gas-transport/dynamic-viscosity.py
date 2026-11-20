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
# # Dynamic viscosity: the gas matters too
#
# How does the surrounding air enter an aerosol transport calculation? Dynamic
# viscosity describes its resistance to shear and appears in drag and particle
# mobility. This small CPU calculation establishes the gas-property input before
# using it in a larger model.
#
# Particula uses Sutherland's temperature relation for air:
#
# $$
# \mu(T)=\mu_0\left(\frac{T}{T_0}\right)^{3/2}
# \frac{T_0+C}{T+C}.
# $$
#
# Here $\mu$ and $\mu_0$ have units Pa s (kg m⁻¹ s⁻¹), and $T$, $T_0$, and $C$
# are in kelvin. At $T=T_0$ the expression returns $\mu_0$. The public function
# accepts temperature, reference viscosity, and reference temperature; the
# Sutherland constant is supplied by the implementation, not a keyword argument.
#
# This is a temperature-dependent air correlation. Pressure is not an input,
# and a successful positive-temperature call does not establish accuracy for
# arbitrary gases or extreme conditions. See [NASA's viscosity discussion](https://www.grc.nasa.gov/www/k-12/airplane/viscosity.html)
# for the gas-transport context and [Sutherland's formula](https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula)
# for the relation.
#
# The following call evaluates air at 298.15 K with the default reference values.

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

# %% [markdown]
# ## Use the number in a physical model
#
# The printed value is approximately 1.84 × 10⁻⁵ Pa s. This is dynamic viscosity;
# kinematic viscosity would also require gas density. The result is a property
# calculation, not a validation against measurements.
#
# At fixed particle size, increasing viscosity reduces continuum Stokes mobility.
# A Brownian diffusion calculation also depends on temperature and, for small
# particles, a slip correction. Consequently, viscosity alone cannot determine
# the full temperature dependence of an aerosol process. Follow this connection
# to [mean free path](mean-free-path.py) and [Brownian coagulation](../../tutorials/coagulation-populations/coagulation_1_pmf_pattern.py).
#
# For a temperature-dependent experiment, a useful extension is to compare the
# correlation over the experimental range with an appropriate reference dataset,
# then propagate the property change into the chosen transport calculation.
# Keep other changing gas properties explicit and establish the correlation's
# validity for the gas and conditions; that comparison is not performed here.

# %% [markdown]
# [Return to the equation and its assumptions](../../../theory/properties/air-transport.md#air-viscosity).
