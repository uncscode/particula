# %% Load the coulomb enhancement module

import numpy as np
import matplotlib.pyplot as plt
from particula.particles.properties import coulomb_enhancement
from particula.util.machine_limit import safe_exp



coulomb_potential_ratio = np.linspace(700, 10_000, 1000)

# Calculate the kinetic and continuum enhancements
kinetic_enhance = coulomb_enhancement.kinetic(coulomb_potential_ratio)
continuum_enhance = coulomb_enhancement.continuum(coulomb_potential_ratio)

# manual

denominator = 1 - safe_exp(-1 * coulomb_potential_ratio)
continuum2 = np.divide(
    coulomb_potential_ratio,
    denominator,
    out=np.ones_like(denominator),
    where=denominator != 0,
)

# Plot the results
fig, ax = plt.subplots()
ax.plot(coulomb_potential_ratio, kinetic_enhance, label="Kinetic enhancement")
ax.plot(coulomb_potential_ratio, continuum_enhance, label="Continuum enhancement")
ax.plot(coulomb_potential_ratio, continuum2, label="Continuum enhancement2",
        linestyle='--')
ax.set_yscale("log")
ax.set_xlabel("Coulomb potential ratio")
ax.set_ylabel("Enhancement factor")
ax.set_title("Coulomb enhancement factors")
ax.legend()
plt.show()
# %%