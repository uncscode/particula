---
template: home.html
title: Particula
description: Predict Experiments, Expand Your Insights.
hide:
  - toc
---

# Particula

## What is Particula?

Particula is an open-source, Python-based aerosol simulator that bridges experimental data with computational models. It captures gas-particle interactions, transformations, and dynamics to power **predictive aerosol science**—so you can uncover deeper insights and accelerate progress.

---

## Why Use Particula?

Aerosols influence atmospheric science, air quality, and human health in powerful ways. Gaining insight into how they behave is essential for effective pollution control, accurate cloud formation modeling, and safer indoor environments. Particula provides a robust, flexible framework to simulate, analyze, and visualize aerosol processes with precision—empowering you to make breakthroughs and drive impactful science.

---

## How Does Particula Help You?

Whether you’re a researcher, educator, or industry expert, Particula is designed to **empower your aerosol work** by:

- **Harnessing ChatGPT integration** for real-time guidance, troubleshooting, and Q&A, [**here**](https://chatgpt.com/g/g-67b9dffbaa988191a4c7adfd4f96af65-particula-assistant).
- **Providing a Python-based API** for reproducible and modular simulations.
- **Interrogating your experimental data** to validate and expand your impact.
- **Fostering open-source collaboration** to share ideas and build on each other’s work.

---

## Get Started with Particula

[Setup Particula](Examples/Setup_Particula/index.md){ .md-button .md-button--primary }
[API Reference](API/README.md){ .md-button }
[Examples](Examples/index.md){ .md-button }
[Theory](Theory/index.md){ .md-button }

---

### :simple-pypi: PyPI Installation
If your Python environment is already set up, install Particula directly from PyPI:
```shell
pip install particula
```

### :simple-condaforge: Conda Installation

Alternatively, you can install Particula using conda:
```shell
conda install -c conda-forge particula
```

If you are new to Python or plan on going through the Examples, head to [Setup Particula](Examples/Setup_Particula/index.md) for more comprehensive installation instructions.

### Quick Start Example

This “Quick Start Example” demonstrates a concise workflow for building an aerosol system in Particula and performing a single condensation step.

```python
import numpy as np
import particula as par

# 1. Build the GasSpecies for an organic vapor:
organic = (
    par.gas.GasSpeciesBuilder()
    .set_name("organic")
    .set_molar_mass(180e-3, "kg/mol")
    .set_vapor_pressure_strategy(
        par.gas.ConstantVaporPressureStrategy(1e2)  # Pa
    )
    .set_partitioning(True)
    .set_concentration(np.array([1e2]), "kg/m^3")
    .build()
)

# 2. Use AtmosphereBuilder to configure temperature, pressure, and species:
atmosphere = (
    par.gas.AtmosphereBuilder()
    .set_temperature(298.15, "K")
    .set_pressure(101325, "Pa")
    .set_more_partitioning_species(organic)
    .build()
)

# 3. Build the particle distribution:
#    Using PresetParticleRadiusBuilder, we set mode radius, GSD, etc.
particle = (
    par.particles.PresetParticleRadiusBuilder()
    .set_mode(np.array([100e-9]), "m")
    .set_geometric_standard_deviation(np.array([1.2]))
    .set_number_concentration(np.array([1e8]), "1/m^3")
    .set_density(1e3, "kg/m^3")
    .build()
)

# 4. Create the Aerosol combining the atmosphere and particle distribution:
aerosol = (
    par.AerosolBuilder()
    .set_atmosphere(atmosphere)
    .set_particles(particle)
    .build()
)

# 5. Define the isothermal condensation strategy:
condensation_strategy = par.dynamics.CondensationIsothermal(
    molar_mass=180e-3,  # kg/mol
    diffusion_coefficient=2e-5,  # m^2/s
    accommodation_coefficient=1.0,
)

# 6. Build the MassCondensation process:
process = par.dynamics.MassCondensation(condensation_strategy)

# 7. Execute the condensation process over 10 seconds:
result = process.execute(aerosol, time_step=10.0)

#   The result is an Aerosol instance with updated particle properties.
print(result)
```

---

## Join the Community

We welcome contributions from scientists, developers, and students—and anyone curious about aerosol science! Whether you’re looking to ask questions, get help, or contribute fresh ideas, you’ve come to the right place.

Get more by posting on [GitHub Discussions](https://github.com/uncscode/particula/discussions) and tag any of the [contributors](https://github.com/uncscode/particula/graphs/contributors) using `@github-handle`.

- 💬 [**Ask questions** and **get help**](https://github.com/uncscode/particula/discussions/new?category=q-a).
- 🚀 [*Share your research*](https://github.com/uncscode/particula/discussions/new?category=show-and-tell) with the community to inspire others.
- 📣 [*Give us feedback.*](https://github.com/uncscode/particula/discussions/new?category=feedback)
- 🌟 **Contribute** to Particula by [*submitting pull requests*](https://github.com/uncscode/particula/pulls) or [*reporting issues*](https://github.com/uncscode/particula/issues) on GitHub.
- 🔗 Read our [**Contributing Guide**](contribute/CONTRIBUTING.md) to learn how you can make an impact.

We’re excited to collaborate with you! ✨

---

## Cite Particula in Your Research

Particula [Computer software]. [DOI: 10.5281/zenodo.6634653](https://doi.org/10.5281/zenodo.6634653)
