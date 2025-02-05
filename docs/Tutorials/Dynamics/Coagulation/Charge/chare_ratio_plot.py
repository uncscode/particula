# %% Load the coulomb enhancement module

import numpy as np
import matplotlib.pyplot as plt
from particula.particles.properties import coulomb_enhancement




coulomb_potential_ratio = np.linspace(-2000, 2000, 1000)

# Calculate the kinetic and continuum enhancements
kinetic_enhance = coulomb_enhancement.kinetic(coulomb_potential_ratio)
continuum_enhance = coulomb_enhancement.continuum(coulomb_potential_ratio)

# Plot the results
fig, ax = plt.subplots()
ax.plot(coulomb_potential_ratio, kinetic_enhance, label="Kinetic enhancement")
ax.plot(coulomb_potential_ratio, continuum_enhance, label="Continuum enhancement")
ax.set_xlabel("Coulomb potential ratio")
ax.set_ylabel("Enhancement factor")
ax.set_title("Coulomb enhancement factors")
ax.legend()
plt.show()
# %%