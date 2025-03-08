

---
# Activity_Functions.md

# Activity Tutorial

This Jupyter notebook is designed to deepen your understanding of mixing behaviors in solutions, focusing on both theoretical models and practical applications. We will explore ideal and non-ideal mixing rules, differentiate between mass-based and molar-based approaches, and introduce the kappa value parameterization for predicting water activity.

## Introduction to Mixing Rules

Mixing rules are essential for predicting the properties of solutions, including their thermodynamic behavior and phase equilibria. In this notebook, we will:

- **Define and compare different mixing rules:** Understand how various rules apply to different types of solutions.
- **Mass-Based vs. Molar-Based Mixing:** Discuss the implications of choosing mass-based or molar-based calculations for different applications.
- **Kappa Value based Activity:** Learn about the kappa value parameterization and its role in modeling water activity in non-ideal solutions.

## Structure of the Notebook

1. **Mass-Based vs. Molar-Based vs. Volueme-Based**
   - Definitions and when to use each method
   - Examples and comparative analysis

2. **Kappa Value Parameterization**
   - Theory behind kappa values
   - Practical exercises on calculating water activity


```python
import numpy as np
import matplotlib.pyplot as plt

import particula as par
```

## Mass Arrays

First we'll need to create some mass concentration arrays to use in our examples. We will use a 2-component and 3-component system for demonstration purposes.

In each the first component is water and the second component is a solute. The mass fractions of the solute will be varied to demonstrate the different mixing rules.


```python
masses_1 = np.linspace(0, 100, 1000)
masses_2 = np.full_like(masses_1, 30)
masses_3 = np.linspace(150, 10, 1000)

density_1 = 1000  # kg/m^3
density_2 = 2000  # kg/m^3
density_3 = 1500  # kg/m^3

molar_mass_1 = 18.01528e-3  # g/mol
molar_mass_2 = 200e-3  # g/mol
molar_mass_3 = 150e-3  # g/mol

mass_2component = np.column_stack((masses_1, masses_2))
mass_3component = np.column_stack((masses_1, masses_2, masses_3))

density_2component = np.array([density_1, density_2])
density_3component = np.array([density_1, density_2, density_3])

# get mole fractions
mass_fractions_2component = par.util.get_mole_fraction_from_mass(
    mass_2component, molar_masses=np.array([molar_mass_1, molar_mass_2])
)
mass_fractions_3component = par.util.get_mole_fraction_from_mass(
    mass_3component,
    molar_masses=np.array([molar_mass_1, molar_mass_2, molar_mass_3]),
)
```

## Molar-Based Mixing

The ideal in this context refers to all the activity coefficients being equal to 1. This is the simplest case and is often used as a reference point for more complex models. In this case, then we are just mixing based on molar fractions.



```python
activity_2component_molar = par.particles.get_ideal_activity_molar(
    mass_concentration=mass_2component,
    molar_mass=np.array([molar_mass_1, molar_mass_2]),
)


activity_3component_molar = par.particles.get_ideal_activity_molar(
    mass_concentration=mass_3component,
    molar_mass=np.array([molar_mass_1, molar_mass_2, molar_mass_3]),
)


# Create the figure and axis objects
fig, ax = plt.subplots(2, 1, figsize=(5, 8))

# Plot each component in the 2-component system with separate colors
ax[0].plot(
    mass_fractions_2component[:, 0],
    activity_2component_molar[:, 0],
    color="blue",
    label="Water (1)",
)
ax[0].plot(
    mass_fractions_2component[:, 0],
    activity_2component_molar[:, 1],
    color="green",
    label="Component 2",
)
ax[0].set_xlabel("Water Mole Fraction")
ax[0].set_ylabel("Activity coefficient")
ax[0].set_title("Activity coefficient vs mass fraction for 2 components")
ax[0].legend()

# Plot the 3-component system without setting specific colors
ax[1].plot(
    mass_fractions_3component[:, 0],
    activity_3component_molar[:, 0],
    label="Water (1)",
    color="blue",
)
ax[1].plot(
    mass_fractions_3component[:, 0],
    activity_3component_molar[:, 1],
    label="Component 2",
    color="green",
)
ax[1].plot(
    mass_fractions_3component[:, 0],
    activity_3component_molar[:, 2],
    label="Component 3",
    color="red",
)
ax[1].set_xlabel("Water Mole Fraction")
ax[1].set_ylabel("Activity coefficient")
ax[1].set_title("Activity coefficient vs mass fraction for 3 components")
ax[1].legend()

# Display the plot
plt.tight_layout()
plt.show()
```


    
![png](output_5_0.png)
    


## Volume-Based Mixing

In this next example, we will use volume-based mixing. This is common for use with liquid mixtures, where the volume of the solution is the sum of the volumes of the components. This is a simple way to mix solutions, but it is not always accurate.


```python
# Volume activity coefficient
activity_2component_volume = par.particles.get_ideal_activity_volume(
    mass_concentration=mass_2component,
    density=density_2component,
)

activity_3component_volume = par.particles.get_ideal_activity_volume(
    mass_concentration=mass_3component,
    density=density_3component,
)

# Create the figure and axis objects
fig, ax = plt.subplots(2, 1, figsize=(5, 8))

# Plot each component in the 2-component system with separate colors
ax[0].plot(
    mass_fractions_2component[:, 0],
    activity_2component_volume[:, 0],
    color="blue",
    label="Water (1)",
)
ax[0].plot(
    mass_fractions_2component[:, 0],
    activity_2component_volume[:, 1],
    color="green",
    label="Component 2",
)
ax[0].set_xlabel("Water Mole Fraction")
ax[0].set_ylabel("Activity coefficient")
ax[0].set_title("Activity coefficient vs mass fraction for 2 components")
ax[0].legend()

# Plot the 3-component system without setting specific colors
ax[1].plot(
    mass_fractions_3component[:, 0],
    activity_3component_volume[:, 0],
    label="Water (1)",
    color="blue",
)
ax[1].plot(
    mass_fractions_3component[:, 0],
    activity_3component_volume[:, 1],
    label="Component 2",
    color="green",
)
ax[1].plot(
    mass_fractions_3component[:, 0],
    activity_3component_volume[:, 2],
    label="Component 3",
    color="red",
)
ax[1].set_xlabel("Water Mole Fraction")
ax[1].set_ylabel("Activity coefficient")
ax[1].set_title("Activity coefficient vs mass fraction for 3 components")
ax[1].legend()

# Display the plot
plt.tight_layout()
plt.show()
```


    
![png](output_7_0.png)
    


## Mass-Based Mixing

In this example, we will use mass-based mixing. This is the simplest, as our mass fractions are directly proportional to the mass of the components.


```python
# Mass activity coefficient
activity_2component_mass = par.particles.get_ideal_activity_mass(
    mass_concentration=mass_2component,
)
activity_3component_mass = par.particles.get_ideal_activity_mass(
    mass_concentration=mass_3component,
)

# Create the figure and axis objects
fig, ax = plt.subplots(2, 1, figsize=(5, 8))

# Plot each component in the 2-component system with separate colors
ax[0].plot(
    mass_fractions_2component[:, 0],
    activity_2component_mass[:, 0],
    color="blue",
    label="Water (1)",
    linewidth=4,
    linestyle="--",
)
ax[0].plot(
    mass_fractions_2component[:, 0],
    activity_2component_mass[:, 1],
    color="green",
    label="Component 2",
)
ax[0].set_xlabel("Water Mole Fraction")
ax[0].set_ylabel("Activity coefficient")
ax[0].set_title("Activity coefficient vs mass fraction for 2 components")
ax[0].legend()

# Plot the 3-component system without setting specific colors
ax[1].plot(
    mass_fractions_3component[:, 0],
    activity_3component_mass[:, 0],
    label="Water (1)",
    color="blue",
    linewidth=4,
    linestyle="--",
)
ax[1].plot(
    mass_fractions_3component[:, 0],
    activity_3component_mass[:, 1],
    label="Component 2",
    color="green",
    linewidth=3,
)
ax[1].plot(
    mass_fractions_3component[:, 0],
    activity_3component_mass[:, 2],
    label="Component 3",
    color="red",
)
ax[1].set_xlabel("Water Mole Fraction")
ax[1].set_ylabel("Activity coefficient")
ax[1].set_title("Activity coefficient vs mass fraction for 3 components")
ax[1].legend()

# Display the plot
plt.tight_layout()
plt.show()
```


    
![png](output_9_0.png)
    


## Kappa-Activity

The kappa value parameterization is a simple way to model non-ideal water interactions in solutions.


```python
# kappa activity coefficient

kappa_1 = 0.0  # kappa of water
kappa_2 = 0.6  # kappa of component 2
kappa_3 = 1.2  # kappa of component 3

water_index = 0

activity_2component_kappa = par.particles.get_kappa_activity(
    mass_concentration=mass_2component,
    kappa=np.array([kappa_1, kappa_2]),
    density=density_2component,
    molar_mass=np.array([molar_mass_1, molar_mass_2]),
    water_index=water_index,
)
activity_3component_kappa = par.particles.get_kappa_activity(
    mass_concentration=mass_3component,
    kappa=np.array([kappa_1, kappa_2, kappa_3]),
    density=density_3component,
    molar_mass=np.array([molar_mass_1, molar_mass_2, molar_mass_3]),
    water_index=water_index,
)

# Create the figure and axis objects
fig, ax = plt.subplots(2, 1, figsize=(5, 8))

# Plot each component in the 2-component system with separate colors
ax[0].plot(
    mass_fractions_2component[:, 0],
    activity_2component_kappa[:, 0],
    color="blue",
    label="Water (1)",
)
ax[0].plot(
    mass_fractions_2component[:, 0],
    activity_2component_kappa[:, 1],
    color="green",
    label="Component 2",
)
ax[0].set_xlabel("Water Mole Fraction")
ax[0].set_ylabel("Activity coefficient")
ax[0].set_title("Kappa Activity coefficient vs mass fraction for 2 components")
ax[0].legend()

# Plot the 3-component system without setting specific colors
ax[1].plot(
    mass_fractions_3component[:, 0],
    activity_3component_kappa[:, 0],
    label="Water (1)",
    color="blue",
)
ax[1].plot(
    mass_fractions_3component[:, 0],
    activity_3component_kappa[:, 1],
    label="Component 2",
    color="green",
)
ax[1].plot(
    mass_fractions_3component[:, 0],
    activity_3component_kappa[:, 2],
    label="Component 3",
    color="red",
)
ax[1].set_xlabel("Water Mole Fraction")
ax[1].set_ylabel("Activity coefficient")
ax[1].set_title("Kappa Activity coefficient vs mass fraction for 3 components")
ax[1].legend()

# Display the plot
plt.tight_layout()
plt.show()
```


    
![png](output_11_0.png)
    


## Summary

By the end of this notebook, you should have a better understanding of mixing rules, mass-based vs. molar-based calculations, and the kappa value parameterization for predicting water activity in non-ideal solutions. You will also have learned how to apply these concepts to practical examples and visualize the results using plots.

You saw how different mixing rules can be used to predict the properties of solutions and how they can affect the accuracy of the predictions. You also learned about the kappa value parameterization and how it can be used to model water activity in non-ideal solutions. These concepts are essential for condensation and phase equilibrium calculations when aerosol particles are present in the atmosphere.


---
# Activity_Tutorial.md

# Activity Tutorial

This Jupyter notebook is designed to deepen your understanding of mixing behaviors in solutions, focusing on both theoretical models and practical applications. We will explore ideal and non-ideal mixing rules, differentiate between mass-based and molar-based approaches, and introduce the kappa value parameterization for predicting water activity.

## Introduction to Mixing Rules

Mixing rules are essential for predicting the properties of solutions, including their thermodynamic behavior and phase equilibria. In this notebook, we will:

- **Define and compare different mixing rules:** Understand how various rules apply to different types of solutions.
- **Mass-Based vs. Molar-Based Mixing:** Discuss the implications of choosing mass-based or molar-based calculations for different applications.
- **Kappa Value based Activity:** Learn about the kappa value parameterization and its role in modeling water activity in non-ideal solutions.

## Structure of the Notebook

1. **Mass-Based vs. Molar-Based Mixing**
   - Definitions and when to use each method
   - Examples and comparative analysis

2. **Kappa Value Parameterization**
   - Theory behind kappa values
   - Practical exercises on calculating water activity

## Strategies, Builders, and Factories

We'll show examples for getting the strategy directly, form a builder, and from a factory.


```python
import numpy as np
import matplotlib.pyplot as plt

import particula as par
```

## Strategy: Ideal Activity Mass-Based Mixing

The ideal in this context refers to all the activity coefficients being equal to 1. This is the simplest case and is often used as a reference point for more complex models. In this case, then we are just mixing based on mass fractions in the solution. Let's start witha mixture of water and sucrose.

With an Ideal mass based mixing rule, the activity and partial pressure reduction is just the mass fraction of the component in the mixture.



```python
activity_mass = par.particles.ActivityIdealMass()  # no parameters needed

# mixture
mass_mixture = np.array([0.2, 0.8])  # water, sucrose

activities = activity_mass.activity(mass_concentration=mass_mixture)
print(f"Activities: {activities}")

# partial pressures
pure_pressure = np.array([100, 10])  # water, sucrose
partial_pressures = activity_mass.partial_pressure(
    pure_vapor_pressure=pure_pressure, mass_concentration=mass_mixture
)

print(f"Partial pressures: {partial_pressures}")
```

    Activities: [0.2 0.8]
    Partial pressures: [20.  8.]


## Builder: Molar-Based Mixing

All strategies have a builder method that can be used to create a new strategy with different parameters. In this case, we will create a molar-based mixing rule using the builder method.

Using the same mixture of water and sucrose, we will now calculate the activity and partial pressure reduction based on molar fractions in the solution. We should see a large effect due to the difference in molecular weight between water and sucrose.


```python
activity_molar = (
    par.particles.ActivityIdealMolarBuilder()
    .set_molar_mass(
        np.array([18.01528, 342.29648]) * 1e-3, "kg/mol"
    )  # water, sucrose
    .build()
)

# mixture
activities = activity_molar.activity(mass_concentration=mass_mixture)
print(f"Activities: {activities}")

# partial pressures
pure_pressure = np.array([100, 10])  # water, sucrose
partial_pressures = activity_molar.partial_pressure(
    pure_vapor_pressure=pure_pressure, mass_concentration=mass_mixture
)

print(f"Partial pressures: {partial_pressures}")
```

    Activities: [0.82608954 0.17391046]
    Partial pressures: [82.6089542   1.73910458]


## Factory: Kappa Value Parameterization

Lastly, we will use the factory method to create a kappa value parameterization for predicting water activity in non-ideal solutions. This method is more complex and requires additional parameters to be defined. We will use the same water-sucrose mixture to demonstrate the kappa value approach.


```python
parameters_input = {
    "density": np.array(
        [1000, 1500]
    ),  # water, sucrose, kg/m^3 are the base SI default units
    "density_units": "kg/m^3",  # this tells the factory what the input units are so it can convert to kg/m^3
    "molar_mass": np.array([18.01528, 342.29648]),  # water, sucrose
    "molar_mass_units": "g/mol",  # this tells the factory what the input units are so it can convert to kg/mol
    "kappa": np.array([0, 0.3]),  # water, sucrose
    "water_index": 0,  # water is the first component
}

activity_kappa = par.particles.ActivityFactory().get_strategy(
    strategy_type="kappa", parameters=parameters_input
)

# mixture
activities = activity_kappa.activity(mass_concentration=mass_mixture)
print(f"Activities: {activities}")

# partial pressures
pure_pressure = np.array([100, 10])  # water, sucrose
partial_pressures = activity_kappa.partial_pressure(
    pure_vapor_pressure=pure_pressure, mass_concentration=mass_mixture
)

print(f"Partial pressures: {partial_pressures}")
```

    Activities: [0.55555556 0.17391046]
    Partial pressures: [55.55555556  1.73910458]


## Plotting Mixing Rules

We will plot the activity and partial pressure reduction for the water-sucrose mixture using the ideal mass-based mixing rule, molar-based mixing rule, and kappa value parameterization. This will help us visualize the differences between the three methods and understand how they affect the prediction of water activity in the solution.

Note: Only water is treated non-ideally in the kappa value parameterization. The other species are treated in a molar-based ideal mixing rule.


```python
mass_water = np.linspace(0.001, 0.9999, 100)
mass_sucrose = 1 - mass_water
mass_mixture = np.array([mass_water, mass_sucrose]).T

activities_mass = np.zeros_like(mass_mixture)
activities_molar = np.zeros_like(mass_mixture)
activities_kappa = np.zeros_like(mass_mixture)


for i, mass in enumerate(mass_mixture):
    activities_mass[i] = activity_mass.activity(mass_concentration=mass)
    activities_molar[i] = activity_molar.activity(mass_concentration=mass)
    activities_kappa[i] = activity_kappa.activity(mass_concentration=mass)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(mass_water, activities_mass[:, 0], label="Mass: Water")
ax.plot(
    mass_water, activities_mass[:, 1], label="Mass: Sucrose", linestyle="--"
)
ax.plot(mass_water, activities_molar[:, 0], label="Molar: Water")
ax.plot(
    mass_water,
    activities_molar[:, 1],
    label="Molar: Sucrose",
    linestyle="--",
    linewidth=3,
    alpha=0.85,
)
ax.plot(mass_water, activities_kappa[:, 0], label="Kappa: Water")
ax.plot(
    mass_water, activities_kappa[:, 1], label="Kappa: Sucrose", linestyle="--"
)
ax.set_xlabel("Mass fraction water")
ax.set_ylabel("Activity")
ax.legend()
plt.show()
```


    
![png](output_9_0.png)
    


## Summary

By the end of this notebook, you should have a better understanding of mixing rules, mass-based vs. molar-based calculations, and the kappa value parameterization for predicting water activity in non-ideal solutions. You will also have learned how to apply these concepts to practical examples and visualize the results using plots.

You saw how different mixing rules can be used to predict the properties of solutions and how they can affect the accuracy of the predictions. You also learned about the kappa value parameterization and how it can be used to model water activity in non-ideal solutions. These concepts are essential for condensation and phase equilibrium calculations when aerosol particles are present in the atmosphere.


---
# Adding_Particles_During_Simulation.md

# Adding Particles During Simulation

In this tutorial, we demonstrate how add particles to an aerosol object. This is useful when you want to modify a custom aerosol process during a simulation.

The example is for a particle resolved simulation, the same approach can be used for the other types of particle representations (*but it has not been tested yet*).

**Imports**


```python
import numpy as np
import matplotlib.pyplot as plt

# particula imports
import particula as par
```

## Aerosol Setup

We need to first make the aerosol object. Details on this can be found in the Aerosol Tutorial.


```python
# Preset gas species that does not condense in the atmosphere
# AtmosphereBuilder constructs the atmosphere with predefined species
atmosphere = (
    par.gas.AtmosphereBuilder()
    .add_species(
        par.gas.PresetGasSpeciesBuilder().build()
    )  # Add a preset gas species
    .set_temperature(25, temperature_units="degC")  # Set temperature to 25°C
    .set_pressure(1, pressure_units="atm")  # Set pressure to 1 atmosphere
    .build()  # Finalize the atmosphere object
)

# Generate a particle distribution using a lognormal sample distribution
# This distribution has a mean particle diameter (mode) and geometric standard deviation (GSD)
particle_sample = par.particles.get_lognormal_sample_distribution(
    mode=np.array([100e-9]),  # Mean particle diameter of 100 nm
    geometric_standard_deviation=np.array([1.3]),  # GSD of 1.3
    number_of_particles=np.array([1e4]),  # Total number of particles
    number_of_samples=100_000,  # Number of samples for particle distribution
)

# Calculate the mass of each particle in the sample, assuming density of 1500 kg/m^3
particle_mass_sample = (
    4 / 3 * np.pi * particle_sample**3 * 1500
)  # Particle mass in kg

# Build a resolved mass representation for each particle
# This defines how particle mass, activity, and surface are represented
resolved_masses = (
    par.particles.ResolvedParticleMassRepresentationBuilder()
    .set_distribution_strategy(
        par.particles.ParticleResolvedSpeciatedMass()
    )  # Use speciated mass distribution
    .set_activity_strategy(
        par.particles.ActivityIdealMass()
    )  # Define activity based on ideal mass
    .set_surface_strategy(
        par.particles.SurfaceStrategyVolume()
    )  # Define surface area based on particle volume
    .set_mass(particle_mass_sample, "kg")  # Assign mass of particles (in kg)
    .set_density(1500, "kg/m^3")  # Set particle density to 1500 kg/m^3
    .set_charge(0)  # Assume neutral particles with no charge
    .set_volume(1, "cm^3")  # Set volume of particle distribution
    .build()  # Finalize the resolved mass representation
)

# Create an aerosol object with the defined atmosphere and resolved particles
aerosol = par.Aerosol(atmosphere=atmosphere, particles=resolved_masses)

# Print the properties of the atmosphere
print(aerosol)
```

    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['Preset100']
    [0]: Particle Representation:
    	Strategy: ParticleResolvedSpeciatedMass
    	Activity: ActivityIdealMass
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 8.556e-07 [kg/m^3]
    	Number Concentration: 1.000e+11 [#/m^3]


## Particles to Add

For the particle resolved representation, the particles to add must be the provide an array of new particle masses and the concentrations. 

Currently the concentrations should all be one, as this is the particle resolved representation.

If you have multiple species, then the shape of the `added_distribution` should be `(number of particles, number of species)`. But `added_concentration` is still `(number of particles,)`.


```python
# particles to add

# Generate a particle distribution using a lognormal sample distribution
# This distribution has a mean particle diameter (mode) and geometric standard deviation (GSD)
particles_to_add = par.particles.get_lognormal_sample_distribution(
    mode=np.array([1e-9]),  # Mean particle diameter of 100 nm
    geometric_standard_deviation=np.array([1.1]),  # GSD of 1.3
    number_of_particles=np.array([1e4]),  # Total number of particles
    number_of_samples=10_000,  # Number of samples for particle distribution
)

# Calculate the mass of each particle in the sample, assuming density of 1500 kg/m^3
particle_mass_add = (
    4 / 3 * np.pi * particles_to_add**3 * 1500
)  # Particle mass in kg
concentration_add = np.ones_like(particle_mass_add)


# print shapes
print(f"Particles to add: {particle_mass_add.shape}")
```

    Particles to add: (10000,)



```python
# Get initial particle radii before adding particle
initial_radii = aerosol.particles[0].get_radius(clone=True)
print(
    f"Initial concentration: {aerosol.particles[0].get_total_concentration()}"
)

# Perform the add process
aerosol.particles[
    0
].add_concentration(  # select the particle representation and call add_concentration
    added_concentration=concentration_add,
    added_distribution=particle_mass_add,
)
radii_after_step_1 = aerosol.particles[0].get_radius(clone=True)
print(
    f"Concentration after step 1: {aerosol.particles[0].get_total_concentration()}"
)

# Perform the add process
aerosol.particles[0].add_concentration(
    added_concentration=concentration_add,
    added_distribution=particle_mass_add,
)
radii_after_step_2 = aerosol.particles[0].get_radius(clone=True)

print(
    f"Concentration after step 2: {aerosol.particles[0].get_total_concentration()}"
)
concentration_value = aerosol.particles[0].concentration
```

    Initial concentration: 99999999999.99998
    Concentration after step 1: 109999999999.99998
    Concentration after step 2: 119999999999.99998


## Graphing

We now visualize the two particle add steps


```python
# Define lognormal bins for particle radius histogram
bins_lognormal = np.logspace(-10, -6, 100)

# Create figure for visualizing the histogram of particle radii
fig, ax = plt.subplots(figsize=(8, 6))

# Plot radii distribution after step 2
bins, edges = np.histogram(radii_after_step_2, bins=bins_lognormal)
ax.bar(
    edges[:-1],
    bins,
    width=np.diff(edges),
    align="edge",
    label="After 2 steps",
    alpha=0.6,
)
# Plot radii distribution after step 1

bins, edges = np.histogram(radii_after_step_1, bins=bins_lognormal)
ax.bar(
    edges[:-1],
    bins,
    width=np.diff(edges),
    align="edge",
    label="After 1 step",
    alpha=0.5,
)

# Plot initial radii distribution
bins, edges = np.histogram(initial_radii, bins=bins_lognormal)
ax.bar(
    edges[:-1],
    bins,
    width=np.diff(edges),
    align="edge",
    label="Initial",
    alpha=0.4,
    edgecolor="black",
)

# Set axes to logarithmic scale for x-axis (particle radius)
ax.set_xscale("log")
ax.set_yscale("log")

# Add labels and legend
ax.set_xlabel("Radius (m)")
ax.set_ylabel("Number of particles")
plt.legend()

# Show the plot
plt.show()
```


    
![png](output_8_0.png)
    


## Conclusion

We have demonstrated how to add particles to an aerosol object. This is useful when you want to modify a aerosol object with a custom process during a simulation.


---
# Aerosol_Distributions.md

# Size Distributions Tutorial

In this tutorial, we will explore how to calculate and plot lognormal distributions for aerosol particles. This is commonly used in aerosol science to model the size distribution of particles in different environmental or experimental conditions.



```python
# %%
import numpy as np
import matplotlib.pyplot as plt

import particula as par
```

## Probability Density Function

(fix this, more details for aerosol science)

The probability density function (PDF) of a lognormal distribution is given by:

$$
f(x, s) = \frac{1}{s x \sqrt{2\pi}} \exp\left(-\frac{\log^2(x)}{2s^2}\right)

$$

where $x$ is the particle diameter, and $s$ is the standard deviation of the distribution.
$$
for x > 0, s > 0
$$

The `scale` parameter is defined as the mode of the distribution.

The probability density above is defined in the “standardized” form. To shift and/or scale the distribution use the loc and scale parameters. Specifically, `lognorm.pdf(x, s, loc, scale)` is identically equivalent to `lognorm.pdf(y, s) / scale` with `y = (x - loc) / scale`. Note that shifting the location of a distribution does not make it a “noncentral” distribution; noncentral generalizations of some distributions are available in separate classes.

- [PDF Wikipedia](https://en.wikipedia.org/wiki/probability_density_function)
- [Log-normal Wikipedia](https://en.wikipedia.org/wiki/Log-normal_distribution)
- [Log-normal Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html)

### Define Particle Size Ranges

We use logarithmic spacing for particle diameters to cover a broad size range typically observed in aerosol particles.


```python
# Define the x_values as a range of particle diameters
x_values = np.logspace(-3, 1, 2000)  # From 0.001 to 10 microns
```

### Single Mode PDF Particle Size Distribution

In this section, we explore modeling a distribution with a single mode, typical for scenarios where particle populations are relatively uniform. The single mode represents a common characteristic size (mode) and spread (geometric standard deviation) of aerosol particles. We utilize parameters for the geometric standard deviation and the modal particle diameter to define this distribution. The distribution is scaled such that the area under the probability density function (PDF) equals the total number of particles, ensuring that it accurately reflects the particle count in terms of probability across different sizes. This method is particularly useful for representing aerosol populations where a single predominant size class exists, making it easier to analyze and predict aerosol behavior in environmental or laboratory settings.


```python
# %%
# Single mode distribution
single_mode_gsd = np.array([1.4])
single_mode = np.array([0.02])
single_mode_nparticles = np.array([1e3])

single_mode_distribution = par.particles.get_lognormal_pdf_distribution(
    x_values, single_mode, single_mode_gsd, single_mode_nparticles
)
```

### Multi-Mode PDF Particle Distribution

For more complex scenarios, such as urban air samples, we often observe multiple modes. Here we define and calculate distributions for a two-mode system.


```python
# %%
# Multi-mode distribution
multi_mode_gsd = np.array([1.4, 1.8])
multi_mode = np.array([0.02, 1.0])
multi_mode_nparticles = np.array([1e3, 1e3])

multi_mode_distribution = par.particles.get_lognormal_pdf_distribution(
    x_values, multi_mode, multi_mode_gsd, multi_mode_nparticles
)
```

### Plotting the PDFs of Particle Size Distributions

Visualizing the probability density functions (PDFs) helps in understanding the frequency of different particle sizes.


```python
# %%
plt.figure(figsize=(10, 6))
plt.plot(x_values, single_mode_distribution, label="Single Mode", linewidth=4)
plt.plot(x_values, multi_mode_distribution, label="Multi Mode")
plt.title("Lognormal Particle Size Distribution")
plt.xlabel("Particle Diameter (μm)")
plt.ylabel("Frequency")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](output_10_0.png)
    


### Calculate and Display Total Concentration from PDFs

Total concentration is important for understanding the overall particle load in a sample.


```python
# %%
single_mode_total_concentration = np.trapezoid(
    single_mode_distribution, x_values
)
multi_mode_total_concentration = np.trapezoid(
    multi_mode_distribution, x_values
)

print(
    f"Total Concentration for Single Mode: {single_mode_total_concentration}"
)
print(f"Total Concentration for Multi Mode: {multi_mode_total_concentration}")
```

    Total Concentration for Single Mode: 1000.0
    Total Concentration for Multi Mode: 2000.000000000001


## Probability Mass Function (PMF) for Aerosol Distributions

The Probability Mass Function (PMF) of aerosol distributions provides a distinct perspective compared to the Probability Density Function (PDF). While the PDF represents the probability of particle sizes occurring within a continuous range, ensuring that the integral over all sizes equals the total number of particles, the PMF deals directly with discrete particle counts.

In PMF, each value represents the actual number of particles expected at a specific size interval, rather than the probability density. This approach is particularly advantageous when quantifying and visualizing the actual particle counts across various size classes, making it ideal for detailed statistical analysis and practical applications like filter testing or health impact studies.

Unlike the PDF, where the area under the curve corresponds to the total number of particles (when scaled appropriately), the PMF sums directly to the total number of particles without needing any integral calculation. Each point on the PMF curve directly indicates the number of particles in that particular size class, thus providing a more intuitive grasp of the size distribution's impact, especially in contexts where the exact count of particles is more relevant than their probability density.

[PMF Wikipedia](https://en.wikipedia.org/wiki/probability_mass_function)


```python
single_pmf_distribution = par.particles.get_lognormal_pmf_distribution(
    x_values, single_mode, single_mode_gsd, single_mode_nparticles
)
multi_pmf_distribution = par.particles.get_lognormal_pmf_distribution(
    x_values, multi_mode, multi_mode_gsd, multi_mode_nparticles
)
```

### Plotting the PMFs of Particle Size Distributions

Particle mass functions (PMFs) tell us about the actual number of particles at different sizes.


```python
plt.figure(figsize=(10, 6))
plt.plot(x_values, single_pmf_distribution, label="Single Mode", linewidth=4)
plt.plot(x_values, multi_pmf_distribution, label="Multi Mode")
plt.title("Lognormal PMF Particle Size Distribution")
plt.xlabel("Particle Diameter (μm)")
plt.ylabel("Number of Particles")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](output_16_0.png)
    


### Calculate and Display Total Number of Particles from PMFs

This helps quantify the actual particle count in different modes.


```python
# %%
single_mode_total_concentration = single_pmf_distribution.sum()
multi_mode_total_concentration = multi_pmf_distribution.sum()

print(
    f"Total Concentration for Single Mode: {single_mode_total_concentration}"
)
print(f"Total Concentration for Multi Mode: {multi_mode_total_concentration}")
```

    Total Concentration for Single Mode: 1000.0000000000001
    Total Concentration for Multi Mode: 2000.0


## Summary of PDF vs. PMF in Aerosol Distributions

In this tutorial, we explored two fundamental representations of aerosol size distributions: the Probability Density Function (PDF) and the Particle Mass Function (PMF). Both offer valuable insights into aerosol characteristics but serve different analytical purposes:

- **Probability Density Function (PDF)**: The PDF provides a normalized view of particle size distribution where the area under the curve represents the total number of particles. It is ideal for understanding the relative frequency of different particle sizes within a continuous range and for conducting probability-based analyses. The PDF is particularly useful in theoretical studies and simulations where understanding the likelihood of particle sizes is crucial.

- **Particle Mass Function (PMF)**: Conversely, the PMF directly quantifies the actual number of particles in each size interval. This discrete representation is especially useful for practical applications such as air quality monitoring and aerosol delivery systems where knowing the exact count of particles at different sizes is necessary. The PMF is straightforward as it adds up to the total particle count directly, providing a more tangible grasp of particle distribution without requiring integration.

Both methods play critical roles in aerosol science, each complementing the other by offering different perspectives on particle size distributions. Understanding when to use each can significantly enhance the accuracy and relevance of aerosol studies.


---
# Aerosol_Tutorial.md

# Aerosol Tutorial

Aerosols are complex systems comprising both gaseous components and particulate matter. To accurately model such systems, we introduce the `Aerosol` class, which serves as a collection the `Atmosphere` and `ParticleRepresentation` objects.

In this quick tutorial, we will demonstrate how to create an `Aerosol` object, as this is the key object that will track the state of the aerosol system during dynamics.


```python
import numpy as np
import matplotlib.pyplot as plt

import particula as par
```

## Gas->Atmosphere and Particles

First we'll create a simple `Atmosphere` object, which will represent the gas phase of the aerosol system. We'll also create a `ParticleRepresentation` object, which will represent the particulate phase of the aerosol system.

For the chemical species, we will use a pure component glycerol system.


```python
# Glycerol gas
molar_mass_glycerol = 92.09382e-3  # kg/mol
parameters_clausius = {
    "latent_heat": 71.5 * molar_mass_glycerol,
    "latent_heat_units": "kJ/kg",
    "temperature_initial": 125.5,
    "temperature_initial_units": "degC",
    "pressure_initial": 1,
    "pressure_initial_units": "mmHg",
}
vapor_pressure_strategy = par.gas.VaporPressureFactory().get_strategy(
    "clausius_clapeyron", parameters_clausius
)

sat_concentration = vapor_pressure_strategy.saturation_concentration(
    molar_mass_glycerol, 298.15
)
print(f"Saturation concentration: {sat_concentration:.2e} kg/m^3")

sat_factor = 0.5  # 50% of saturation concentration
glycerol_gas = (
    par.gas.GasSpeciesBuilder()
    .set_molar_mass(molar_mass_glycerol, "kg/mol")
    .set_vapor_pressure_strategy(vapor_pressure_strategy)
    .set_concentration(sat_concentration * sat_factor, "kg/m^3")
    .set_name("Glycerol")
    .set_condensable(True)
    .build()
)
print(glycerol_gas)

atmosphere = (
    par.gas.AtmosphereBuilder()
    .add_species(glycerol_gas)
    .set_temperature(25, temperature_units="degC")
    .set_pressure(1, pressure_units="atm")
    .build()
)
print(atmosphere)

# Glycerol particle distribution
lognormal_rep = (
    par.particles.PresetParticleRadiusBuilder()
    .set_mode(np.array([100]), "nm")
    .set_geometric_standard_deviation(np.array([1.5]))
    .set_number_concentration(np.array([1e4]), "1/cm^3")
    .set_density(1.26, "g/cm^3")
    .build()
)
```

    Saturation concentration: 2.54e-03 kg/m^3
    Glycerol
    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['Glycerol']


## Creating an Aerosol object

With both the `Atmosphere` and `ParticleRepresentation` objects created, we can now create an `Aerosol` object. This object will contain both the gas and particle phase objects, and will be used to track the state of the aerosol system during dynamics.


```python
aerosol = par.Aerosol(atmosphere=atmosphere, particles=lognormal_rep)

print(aerosol)
```

    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['Glycerol']
    [0]: Particle Representation:
    	Strategy: RadiiBasedMovingBin
    	Activity: ActivityIdealMass
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 1.106e-07 [kg/m^3]
    	Number Concentration: 1.000e+10 [#/m^3]


# Summary

In this tutorial, we demonstrated how to create an `Aerosol` object, which is the key object that will track the state of the aerosol system during dynamics. It is pretty simple, as the `Aerosol` object is just a collection of the `Atmosphere` and `ParticleRepresentation` objects and only functions as a container for these objects. It can also iterate over the `Atmosphere` and `ParticleRepresentation` objects.


---
# AtmosphereTutorial.md

# Atmosphere Tutorial

Gases, alongside particles, constitute the essential components of an aerosol system. In their natural state, gases are collections of molecules that move freely, not bound to one another. We introduce the `Atmosphere` class, a composite that encapsolate `GasSpecies`, with addition parameters for the atmospheric state.

- **`Atmosphere`**: This class represents the atmospheric environment by detailing properties such as temperature and pressure, alongside a dynamic list of gas species
    present.
- **`AtmosphericBuilder`**: A builder class that simplifies the creation of `Atmosphere` objects.

We'll continue with our organics and water example, combining the two into a single `Atmosphere` object.


```python
import numpy as np
import matplotlib.pyplot as plt

# import particula
import particula as par
```

## Build Gas Species

First we will build the, `GasSpecies` objects for the organics and water. Following the same procedure from previously in [`Gas Species`](./next_gas_species.ipynb).


```python
# Define the coefficients for Butanol using the Antoine equation.
butanol_coefficients = {"a": 7.838, "b": 1558.19, "c": 196.881}
butanol_antione = par.gas.VaporPressureFactory().get_strategy(
    "antoine", butanol_coefficients
)
styrene_coefficients = {"a": 6.924, "b": 1420, "c": 226}
styrene_antione = par.gas.VaporPressureFactory().get_strategy(
    "antoine", styrene_coefficients
)

# Water uses a different model for vapor pressure calculation called the Buck equation.
water_buck = par.gas.VaporPressureFactory().get_strategy("water_buck")

# Create the GasSpecies using the GasSpeciesBuilder
# water species
water_species = (
    par.gas.GasSpeciesBuilder()
    .set_name("water")
    .set_molar_mass(18.01528e-3, "kg/mol")
    .set_vapor_pressure_strategy(water_buck)
    .set_condensable(True)
    .set_concentration(1e-3, "kg/m^3")
    .build()
)

# organic species
organic_molar_mass = np.array([0.074121, 104.15e-3])
organic_vapor_pressure = [butanol_antione, styrene_antione]
organic_concentration = np.array([2e-6, 1e-9])
organic_names = np.array(["butanol", "styrene"])
organic_species = (
    par.gas.GasSpeciesBuilder()
    .set_name(organic_names)
    .set_molar_mass(organic_molar_mass, "kg/mol")
    .set_vapor_pressure_strategy(organic_vapor_pressure)
    .set_condensable(True)
    .set_concentration(organic_concentration, "kg/m^3")
    .build()
)

# Print the species
print(water_species)
print(organic_species)
```

    water
    ['butanol' 'styrene']


## Atmosphere Builder

The `AtmosphereBuilder` class is a builder class that simplifies the creation of `Atmosphere` objects. It provides a fluent interface for adding `GasSpecies` objects to the `Atmosphere` object. We will use it to build the `Atmosphere` object for the organics and water. The builder requries the following parameters:

- `pressure`: The total pressure of the gas mixture, in Pascals, or provided pressure_units string for conversion.
- `temperature`: The temperature of the gas mixture, in Kelvin, or provided temperature_units string for conversion.
- `species`: A list of `GasSpecies` objects, representing the gases in the mixture. This can be added one by one using the `add_species` method.

### Air

Air is assumed to be the non-specified component of the gas mixture, making up the remainder of the gas mixture. We do not explicitly add air to the gas mixture, but it is implicitly included in most calculations.


```python
gas_mixture = (
    par.gas.AtmosphereBuilder()
    .add_species(water_species)
    .add_species(organic_species)
    .set_temperature(25, temperature_units="degC")
    .set_pressure(1, pressure_units="atm")
    .build()
)

print("Notice the units conversion to base SI:")
print(gas_mixture)
```

    Notice the units conversion to base SI:
    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['water', "['butanol' 'styrene']"]


## Iterating Over Gas Species

Once the `Gas` object has been established, it enables us to iterate over each `GasSpecies` within the mixture. This functionality is particularly valuable for evaluating and adjusting properties dynamically, such as when changes in temperature and pressure occur due to environmental alterations.

### Practical Example: Altitude Impact

Consider a scenario where our gas mixture is transported from sea level to an altitude of 10 kilometers. Such a change in altitude significantly impacts both temperature and pressure, which in turn affects the behavior of each gas species in the mixture.

#### Geopotential Height Equation

The pressure and temperature changes with altitude can be approximated by using the geopotential height equation. Here's how you can calculate these changes:

1. **Pressure Change**: The pressure at a given altitude can be estimated by:
   
$$
   P = P_0 \left(1 - \frac{L \cdot h}{T_0}\right)^{\frac{g \cdot M}{R \cdot L}}
$$

   where:
   - $ P $ is the pressure at altitude $ h $,
   - $ P_0 $ is the reference pressure at sea level (101325 Pa),
   - $ L $ is the standard temperature lapse rate (approximately 0.0065 K/m),
   - $ h $ is the altitude in meters (10000 m for 10 km),
   - $ T_0 $ is the reference temperature at sea level (288.15 K),
   - $ g $ is the acceleration due to gravity (9.80665 m/s²),
   - $ M $ is the molar mass of Earth's air (0.0289644 kg/mol),
   - $ R $ is the universal gas constant (8.314 J/(mol·K)).

1. **Temperature Change**: The temperature decreases linearly with altitude at the lapse rate $ L $:
   
   $$
   T = T_0 - L h
   $$

   Using this formula, we can estimate the temperature at an altitude of 10 km:
   - $T$ = 288.15 K 
   - $L$ 0.0065 K/m
   - $h$ = 10000 m

### Application
By iterating through each `GasSpecies`, we can apply these formulas to adjust their properties based on the calculated pressure and temperature at 10 km altitude, aiding in simulations or real-world applications where altitude plays a crucial role in gas behavior.



```python
# Constants for calculations
sea_level_pressure = 101325  # Reference pressure at sea level (Pa)
sea_level_temperature = 330  # Reference temperature at sea level (K)
gravity = 9.80665  # Acceleration due to gravity (m/s^2)
molar_mass_air = 0.0289644  # Molar mass of Earth's air (kg/mol)
universal_gas_constant = 8.314  # Universal gas constant (J/(mol·K))
temperature_lapse_rate = 0.0065  # Standard temperature lapse rate (K/m)

# Generate an array of altitudes from sea level (0 meters) to 10 km (10000 meters), divided into 100 intervals
altitude_range = np.linspace(0, 10000, 100)

# Calculate the temperature at each altitude based on the linear temperature lapse rate
temperature_at_altitudes = (
    sea_level_temperature - temperature_lapse_rate * altitude_range
)

# Calculate the pressure at each altitude using the barometric formula
pressure_at_altitudes = sea_level_pressure * (
    (1 - temperature_lapse_rate * altitude_range / sea_level_temperature)
    ** (
        gravity
        * molar_mass_air
        / (universal_gas_constant * temperature_lapse_rate)
    )
)


# Initialize a matrix to hold saturation ratios for each species at each
# altitude
saturation_ratio = np.zeros(len(altitude_range))

# Loop over each altitude's temperature and pressure
for index, (temperature, pressure) in enumerate(
    zip(temperature_at_altitudes, pressure_at_altitudes)
):
    # Set the current temperature and pressure of the gas mixture
    gas_mixture.temperature = temperature
    gas_mixture.total_pressure = pressure

    # Loop over water
    saturation_ratio[index] = gas_mixture.species[0].get_saturation_ratio(
        gas_mixture.temperature
    )


# Plot the saturation ratio of water vapor at each altitude
fig, ax = plt.subplots()
ax.plot(saturation_ratio, altitude_range, label="Water")
ax.set_xscale("log")
ax.set_ylabel("Altitude (m)")
ax.set_xlabel("Water Saturation Ratio")
ax.set_title("Saturation Ratio of Water Vapor at Different Altitudes")
ax.legend()
plt.show()
```


    
![png](output_7_0.png)
    


# Summary

In this notebook, we learned how to create an `Atmosphere` object using the `AtmosphereBuilder` class. We also explored how to iterate over each `GasSpecies` within the mixture, enabling us to adjust properties dynamically based on environmental changes. This functionality is particularly useful for simulating real-world scenarios where temperature and pressure variations significantly impact gas behavior.

We now need to build the particle representation, so that combined with the `Atmosphere`, we can create an aerosol system.


---
# Coagulation_1_PMF_Pattern.md

# Coagulation Patterns: PMF Particle Distribution

This notebook explores the coagulation process using a Probability Mass Function (PMF) to define the initial particle size distribution. A PMF provides a representation of discrete particle sizes bins and their associated counts.

The PMF-based distribution allows us to model how the number of particles in each size category changes over time due to coagulation, providing insights into the size distribution's shift toward larger particles.

**PMF Particle Distribution:**
We initialize the particle size distribution using a PMF, where each particle bin has a concentration count. This discrete distribution captures the initial population of particles, categorized by size. The PMF provides flexibility in representing systems where specific particle sizes are dominant or where particles are grouped into size bins.

**Coagulation Process:**
The coagulation process is modeled using a discrete bin approach. We define the coagulation process using a flexible `Coagulation` class, which allows us to choose different coagulation strategies. In this notebook, we employ the `DiscreteSimple` strategy, which tracks each particle's properties as it undergoes coagulation.

**Imports:**


```python
import numpy as np
import matplotlib.pyplot as plt

import particula as par
```

## Aerosol Setup

This section sets up the aerosol system, defining both the atmospheric conditions and the properties of the particles within it. We use the Builder pattern to construct the atmosphere and the particle mass distribution, ensuring that all parameters are defined explicitly and can be validated during the setup process.

**Atmospheric Setup**

The atmosphere is created using the `AtmosphereBuilder`. This class allows us to define key environmental parameters such as the gas species, temperature, and pressure.

- **Gas Species:** We add a preset gas species using the `PresetGasSpeciesBuilder`, which represents a non-condensing gas in the atmosphere.
- **Temperature:** The temperature is set to 25°C, representing typical atmospheric conditions.
- **Pressure:** Atmospheric pressure is set to 1 atm, simulating standard sea-level pressure.

The `build()` method finalizes the atmosphere object, which will be used in the aerosol simulation.

**Resolved Particle Mass Representation**

Next, we define the particle mass distribution using the `PresetParticleRadiusBuilder`. This builder allows for setting up a detailed particle distribution based on physical properties such as particle size (mode), geometric standard deviation (GSD), and number concentration. 

- **Mode:** The particle size modes are set to 100 nm and 800 nm, defining two distinct groups of particles within the aerosol.
- **Geometric Standard Deviation (GSD):** GSD values of 1.2 and 1.5 represent the spread of particle sizes within each mode, with the larger GSD indicating a broader distribution of particle sizes.
- **Number Concentration:** The number concentration for the two modes is defined as 1e5 and 2e5 particles per cm³, respectively.
- **Distribution Type:** We specify that the distribution follows a Probability Mass Function (PMF), which allows for a discrete representation of particle sizes.
- **Radius Bins:** The radius bins are defined using `np.logspace` to create a logarithmic spacing between particle radii ranging from 10 nm to 100 µm. This ensures that the distribution captures a wide range of particle sizes.

Once all parameters are set, the `build()` method finalizes the particle mass representation.

**Aerosol Object Creation**

Finally, the aerosol system is created by combining the atmospheric conditions and the resolved particle masses. The resulting `aerosol` object contains both the gas phase and particle distribution, ready for use in the coagulation simulation.


```python
# Preset gas species that does not condense in the atmosphere
# AtmosphereBuilder constructs the atmosphere with predefined species
atmosphere = (
    par.gas.AtmosphereBuilder()
    .add_species(
        par.gas.PresetGasSpeciesBuilder().build()
    )  # Add a preset gas species
    .set_temperature(25, temperature_units="degC")  # Set temperature to 25°C
    .set_pressure(1, pressure_units="atm")  # Set pressure to 1 atmosphere
    .build()  # Finalize the atmosphere object
)


# Build a resolved mass representation for each particle
# This defines how particle mass, activity, and surface are represented
radius_bins = np.logspace(
    -8, -5, 250
)  # Define the radius bins for the resolved mass representation
resolved_masses = (
    par.particles.PresetParticleRadiusBuilder()
    .set_mode(np.array([100, 800]), mode_units="nm")  # Set the mode radius
    .set_geometric_standard_deviation(
        np.array([1.2, 1.5])
    )  # Set the geometric standard deviation
    .set_number_concentration(
        np.array([1e5, 2e5]), number_concentration_units="cm^-3"
    )  # Set the number concentration
    .set_distribution_type("pmf")  # Set the distribution type to PMF
    .set_radius_bins(radius_bins, radius_bins_units="m")  # Set the radius bins
    .build()  # Finalize the resolved mass representation
)

# Create an aerosol object with the defined atmosphere and resolved particles
aerosol = par.Aerosol(atmosphere=atmosphere, particles=resolved_masses)

# Print the properties
print(aerosol)
```

    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['Preset100']
    [0]: Particle Representation:
    	Strategy: RadiiBasedMovingBin
    	Activity: ActivityIdealMass
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 8.993e-04 [kg/m^3]
    	Number Concentration: 3.000e+11 [#/m^3]


## Simulation

In this section, we define the coagulation process and run it over multiple time steps. Coagulation is the process by which particles in an aerosol collide and merge, resulting in fewer, larger particles over time. The `Coagulation` class is used to simulate this behavior in a stepwise manner, updating the particle size distribution as the simulation progresses.

**Defining the Coagulation Strategy and Process**

We start by selecting a coagulation strategy using `DiscreteSimple()`, which defines how particles will interact and merge during the coagulation process. In this case, the `DiscreteSimple` strategy simplifies the coagulation by treating particle collisions discretely, allowing for straightforward tracking of particle size and number changes.

- **Coagulation Strategy:** The strategy dictates how particle interactions are handled. Here, `DiscreteSimple` offers a simplified, yet effective, approach for discrete particle interactions.
- **Coagulation Process:** The `Coagulation` class orchestrates the entire process, taking the defined strategy and applying it to the aerosol particles over the specified time and sub-steps.

**Simulation Setup: Time and Sub-Steps**

The coagulation process runs over defined time steps and sub-steps:
- **Time Step:** Each time step simulates the evolution of the aerosol system over a specific interval. In this case, it is set to 1000, representing a coarse time resolution.
- **Sub-Steps:** The time step is further divided into 100 sub-steps, which ensures a finer resolution for particle interactions, capturing the nuances of the coagulation process more accurately.

**Running the Coagulation Process**

The coagulation process is executed for the first time step using the `execute()` method. This method updates the aerosol object, modifying the particle size distribution as particles collide and merge. After this step:
- **Radii After Step:** The particle radii are extracted again to observe the changes in size distribution due to coagulation.
- **Concentration After Step:** The concentration of particles in each size bin is updated and saved for comparison.



```python
# Define the coagulation strategy and process
coagulation_strategy = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type(distribution_type="discrete")
    .build()
)
coagulation_process = par.dynamics.Coagulation(
    coagulation_strategy=coagulation_strategy
)

# Set up time and sub-steps for the coagulation process
time_step = 1000
sub_steps = 100

initial_radii = aerosol.particles[0].get_radius()
concentration_initial = np.copy(aerosol.particles[0].concentration)

# # Perform coagulation process for step 1
aerosol = coagulation_process.execute(
    aerosol, time_step=time_step, sub_steps=sub_steps
)
radii_after_step_1 = aerosol.particles[0].get_radius()
concentration_step_1 = np.copy(aerosol.particles[0].concentration)

# Perform coagulation process for step 2
aerosol = coagulation_process.execute(
    aerosol, time_step=time_step, sub_steps=sub_steps
)
radii_after_step_2 = aerosol.particles[0].get_radius()
concentration_step_2 = np.copy(aerosol.particles[0].concentration)
```

## Graphing

In this section, we visualize how the particle size distribution evolves after each step of the coagulation process. The graph displays the number concentration of particles (in m⁻³) as a function of particle radius (in meters). We use three curves to represent the distribution at different stages of the simulation:

1. **Initial**: This curve represents the particle distribution before any coagulation occurs.
2. **Step 1**: This curve shows how the distribution has changed after one step of the coagulation process.
3. **Step 2**: This curve reflects the further evolution of the particle sizes after a second step of coagulation.


**Coagulation Effect**:

   - After the first step (orange line), the peaks shift downward, indicating a reduction in the number concentration of particles across both size ranges. This is due to smaller particles merging through coagulation, resulting in fewer particles overall.
   - After the second step (green line), the number concentration of particles continues to decrease, with the peaks further reducing in height.
   - The shift towards larger particles becomes more evident as the second peak moves slightly to the right. This is a typical result of coagulation, where larger particles grow as smaller particles merge into them.

**Distribution Changes**:

- **Decrease in Number Concentration**: Coagulation leads to a reduction in the number of smaller particles as they combine to form larger ones. This is reflected in the decrease in concentration after each step.
- **Shift Toward Larger Particles**: The coagulation process shifts the distribution toward larger particle sizes. While the first step results in some particles merging, the second step pushes this trend further, as seen in the slight shift of the second peak to larger radii.
- **Wider Distribution**: As the simulation progresses, the particle size distribution becomes broader, indicating increased variability in particle sizes. This suggests that coagulation is affecting particles across a range of sizes, not just those at the peaks.


```python
# Create figure for visualizing the histogram of particle radii
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the histogram of particle radii after step 1
ax.plot(initial_radii, concentration_initial, label="Initial")
ax.plot(radii_after_step_1, concentration_step_1, label="Step 1")
ax.plot(radii_after_step_2, concentration_step_2, label="Step 2")

# Set the labels and title of the plot
ax.set_xlabel("Particle Radius (m)")
ax.set_ylabel(r"Number Concentration ($m^{-3}$)")
ax.set_title("Particle Radius Distribution After Coagulation Steps")
ax.legend()
ax.set_xscale("log")
plt.show()
```


    
![png](output_7_0.png)
    


## Conclusion

This notebook demonstrates how a PMF-based particle distribution can be used to model the coagulation process in an aerosol system. By tracking the changes in particle size distribution over time, we can observe the shift towards larger particles due to coagulation. The discrete representation of particle sizes allows for detailed insights into how particles interact and merge, leading to changes in the aerosol composition.


---
# Coagulation_3_Particle_Resolved_Pattern.md

# Coagulation Patterns: Particle-Resolved Approach

In this notebook, we explore coagulation patterns through a particle-resolved approach. Rather than directly invoking coagulation functions for each calculation, we adopt a more modular and object-oriented programming structure. By employing design patterns, such as the `Builder` pattern, we simplify the setup and validation of parameters, making the process more maintainable and scalable.

This approach allows for the efficient management of both the gas phase and the particle distribution, incorporating the verification of input parameters for accuracy and consistency. The `Builder` classes facilitate setting up the coagulation environment, from defining the atmospheric conditions to creating particle distributions and specifying their resolved masses.

In this example, we preset a non-condensing gas species in the atmosphere and use a lognormal distribution for particles. We represent the particles using a particle-resolved mass framework, where we handle properties like density, charge, and mass with strategies to define the activity, surface area, and distribution.

**Imports**


```python
import numpy as np
import matplotlib.pyplot as plt

# particula imports
import particula as par
```

## Aerosol Setup

In this section, we define the core components of our particle-resolved coagulation system, focusing on the atmosphere setup, particle distribution, and mass resolution. This step-by-step breakdown helps clarify how the `Builder` pattern organizes the construction of complex objects and ensures input parameters are properly verified.

**Atmospheric Setup**
We begin by configuring the atmosphere using the `AtmosphereBuilder`. This allows for flexibility in defining environmental parameters such as temperature and pressure, as well as adding gas species. In this case, we add a preset gas species that does not condense and set the atmospheric conditions to 25°C and 1 atm.

**Particle Distribution**
The particle distribution is generated using a lognormal distribution, a common approach for representing aerosol particle sizes. The parameters for this distribution include:
- A mode of 100 nm, representing the mean particle diameter.
- A geometric standard deviation (GSD) of 1.3, which controls the spread of particle sizes.
- The total number of particles is 100,000 samples taken to capture the variability of the distribution.

**Mass Calculation**
The mass of each particle is calculated assuming a particle density of 1500 kg/m³. This density corresponds to typical aerosol materials like dust or certain types of particulate matter. The particle masses are computed using the formula for the volume of a sphere, multiplied by the density.

**Resolved Particle Mass Representation**
To capture the diversity of the aerosol population, we use a particle-resolved representation for mass. This approach explicitly tracks individual particle masses and assigns properties such as density and charge. The key strategies used are:
- **Distribution strategy**: Defines how mass is distributed among particles.
- **Activity strategy**: Describes how the activity of the particles is represented, in this case, assuming ideal mass behavior.
- **Surface strategy**: Calculates particle surface behavior by volume mixing.

**Final Aerosol Object**
The `Aerosol` object brings together the atmosphere and the resolved particle masses into a cohesive framework. This encapsulated representation can then be used to simulate particle interactions and coagulation events within the atmosphere.

Finally, we print the properties of the `aerosol` object’s atmosphere to verify the correct setup.


```python
# Preset gas species that does not condense in the atmosphere
# AtmosphereBuilder constructs the atmosphere with predefined species
atmosphere = (
    par.gas.AtmosphereBuilder()
    .add_species(
        par.gas.PresetGasSpeciesBuilder().build()
    )  # Add a preset gas species
    .set_temperature(25, temperature_units="degC")  # Set temperature to 25°C
    .set_pressure(1, pressure_units="atm")  # Set pressure to 1 atmosphere
    .build()  # Finalize the atmosphere object
)

# Generate a particle distribution using a lognormal sample distribution
# This distribution has a mean particle diameter (mode) and geometric standard deviation (GSD)
particle_sample = par.particles.get_lognormal_sample_distribution(
    mode=np.array([100e-9]),  # Mean particle diameter of 100 nm
    geometric_standard_deviation=np.array([1.3]),  # GSD of 1.3
    number_of_particles=np.array([1e4]),  # Total number of particles
    number_of_samples=100_000,  # Number of samples for particle distribution
)

# Calculate the mass of each particle in the sample, assuming density of 1500 kg/m^3
particle_mass_sample = (
    4 / 3 * np.pi * particle_sample**3 * 1500
)  # Particle mass in kg

# Build a resolved mass representation for each particle
# This defines how particle mass, activity, and surface are represented
resolved_masses = (
    par.particles.ResolvedParticleMassRepresentationBuilder()
    .set_distribution_strategy(
        par.particles.ParticleResolvedSpeciatedMass()
    )  # Use speciated mass distribution
    .set_activity_strategy(
        par.particles.ActivityIdealMass()
    )  # Define activity based on ideal mass
    .set_surface_strategy(
        par.particles.SurfaceStrategyVolume()
    )  # Define surface area based on particle volume
    .set_mass(particle_mass_sample, "kg")  # Assign mass of particles (in kg)
    .set_density(1500, "kg/m^3")  # Set particle density to 1500 kg/m^3
    .set_charge(0)  # Assume neutral particles with no charge
    .set_volume(0.1, "cm^3")  # Set volume of particle distribution
    .build()  # Finalize the resolved mass representation
)

# Create an aerosol object with the defined atmosphere and resolved particles
aerosol = par.Aerosol(atmosphere=atmosphere, particles=resolved_masses)

# Print the properties of the atmosphere
print(aerosol)
```

    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['Preset100']
    [0]: Particle Representation:
    	Strategy: ParticleResolvedSpeciatedMass
    	Activity: ActivityIdealMass
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 8.546e-06 [kg/m^3]
    	Number Concentration: 1.000e+12 [#/m^3]


## Simulation

In this section, we run the coagulation simulation by first defining the coagulation strategy and the process through which it is executed.

- **Coagulation Strategy:** The strategy for how coagulation is performed is selected using `coagulation.ParticleResolved()`. This specific strategy dictates how particle interactions are handled at the individual particle level, ensuring that the coagulation process respects the details of a particle-resolved approach. In this context, particles are treated as distinct entities, and the merging process is computed explicitly for each pair that interacts.

- **Particle Process:** Once the strategy is defined, the `Coagulation` process is initialized by passing the chosen strategy (`coagulation_strategy`) to the `particle_process.Coagulation` class. This `Coagulation` class is responsible for running the actual simulation. It orchestrates the execution of coagulation by applying the selected strategy over the particle distribution. During each execution step, the particle properties—such as mass, size, and count—are updated according to the rules defined by the particle-resolved strategy.

- **Execution of the Process:** The coagulation process is applied in steps using the `coagulation_process.execute()` method. In each step, the particles' masses are updated based on the time step and sub-steps provided. The time step controls the temporal resolution of the simulation, while the sub-steps break the time step into finer increments to ensure accurate resolution of coagulation events.

For each step:

1. The radii of particles are obtained before and after the coagulation step.
2. The updated particle properties, such as radius and mass, are recorded.
3. After the final step, we count the number of particles that have fully coagulated, i.e., those that have a mass of zero.


```python
# Define the coagulation strategy and process
coagulation_strategy = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type(distribution_type="particle_resolved")
    .build()
)
coagulation_process = par.dynamics.Coagulation(
    coagulation_strategy=coagulation_strategy
)

# Set up time and sub-steps for the coagulation process
time_step = 1000
sub_steps = 100

# Define lognormal bins for particle radius histogram
bins_lognormal = np.logspace(-8, -6, 100)

# Get initial particle radii before the coagulation process
initial_radii = aerosol.particles[0].get_radius()

# Perform coagulation process for step 1
aerosol = coagulation_process.execute(
    aerosol, time_step=time_step, sub_steps=sub_steps
)
radii_after_step_1 = aerosol.particles[0].get_radius()

# Perform coagulation process for step 2
aerosol = coagulation_process.execute(
    aerosol, time_step=time_step, sub_steps=sub_steps
)
radii_after_step_2 = aerosol.particles[0].get_radius()

# Count particles that have coagulated (i.e., have zero mass)
zero_count = np.sum(aerosol.particles[0].get_mass() == 0)
print(f"Particles that coagulated: {zero_count}")
```

    Particles that coagulated: 51721


## Graphing

In this section, we visualize the evolution of the particle size distribution as the coagulation process progresses.

- We use a histogram to show the distribution of particle radii at three stages: initially, after step 1, and after step 2.
- The x-axis is scaled logarithmically to properly represent the range of particle sizes, which can span multiple orders of magnitude.
- The plot helps illustrate the effect of coagulation, where particles merge over time, shifting the distribution towards larger sizes and reducing the number of smaller particles.

This visual representation provides an intuitive understanding of how the coagulation process influences particle sizes, which is key to understanding aerosol dynamics in various atmospheric conditions.


```python
# Create figure for visualizing the histogram of particle radii
fig, ax = plt.subplots(figsize=(8, 6))

# Plot initial radii distribution
bins, edges = np.histogram(initial_radii, bins=bins_lognormal)
ax.bar(edges[:-1], bins, width=np.diff(edges), align="edge", label="Initial")

# Plot radii distribution after step 1
bins, edges = np.histogram(radii_after_step_1, bins=bins_lognormal)
ax.bar(
    edges[:-1],
    bins,
    width=np.diff(edges),
    align="edge",
    label="After 1 step",
    alpha=0.7,
)

# Plot radii distribution after step 2
bins, edges = np.histogram(radii_after_step_2, bins=bins_lognormal)
ax.bar(
    edges[:-1],
    bins,
    width=np.diff(edges),
    align="edge",
    label="After 2 steps",
    alpha=0.6,
)

# Set axes to logarithmic scale for x-axis (particle radius)
ax.set_xscale("log")

# Add labels and legend
ax.set_xlabel("Radius (m)")
ax.set_ylabel("Number of particles")
plt.legend()

# Show the plot
plt.show()
```


    
![png](output_7_0.png)
    


## Conclusion

In this notebook, we have demonstrated a particle-resolved approach to modeling coagulation patterns in aerosol systems. By leveraging the `Builder` pattern and modular design, we have created a flexible and extensible framework for simulating particle interactions and tracking their properties over time.


---
# Coagulation_4_Compared.md

# Coagulation Patterns: Comparison of Number and Mass

In this notebook, we explore and compare three distinct methods for representing particle distributions and modeling the coagulation process: the probability mass function (PMF), the probability density function (PDF), and the particle-resolved approach. The goal is to evaluate how each method impacts the number and mass of particles as coagulation progresses, providing insight into their strengths and limitations.


**Imports**


```python
import numpy as np
import matplotlib.pyplot as plt

import particula as par
```

## Setup Distributions

In this section, we define the common parameters used throughout the notebook for modeling the particle distribution and atmosphere. These parameters include the mode of the particle size distribution, the geometric standard deviation, the number of particles in each mode, and the particle density. Additionally, we set the volume of the system for our simulations. The volume is only needed for the particle resolved approach, as the PMF and PDF methods do not require a volume to be defined.

We also construct a simplified atmospheric environment using an `AtmosphereBuilder`, which includes a preset gas species that does not condense in the atmosphere. The temperature is set to 25°C and the pressure to 1 atmosphere, reflecting typical ambient conditions.


```python
# common parameters
mode = np.array([100e-9, 300e-9])  # m
geometric_standard_deviation = np.array([1.3, 1.3])
number_of_particles = np.array(
    [0.75, 0.25]
)  # effective when pdf has multiple modes
density = np.array([1.0e3])
volume = 1 * par.util.get_unit_conversion("cm^3", "m^3")  # cm^-3 to m^-3

# Preset gas species that does not condense in the atmosphere
# AtmosphereBuilder constructs the atmosphere with predefined species
atmosphere = (
    par.gas.AtmosphereBuilder()
    .add_species(
        par.gas.PresetGasSpeciesBuilder().build()
    )  # Add a preset gas species
    .set_temperature(25, temperature_units="degC")  # Set temperature to 25°C
    .set_pressure(1, pressure_units="atm")  # Set pressure to 1 atmosphere
    .build()  # Finalize the atmosphere object
)
```

**Particle Resolved**

In this section, we generate a particle distribution using a lognormal sample distribution, which is characterized by a specified mode (mean particle diameter) and geometric standard deviation (GSD). We then calculate the mass of each particle, assuming a constant density.

Following this, we create a resolved mass representation for each particle, which defines how properties like mass, activity, and particle surfaces represented. The resolved mass representation is built using predefined strategies for mass distribution, activity, and surface area. We assign particle mass, set density, assume neutral particles (with no charge), and define the volume of the distribution.

Finally, we combine the resolved particle properties with the atmospheric conditions to create an `Aerosol` object, which encapsulates both the particles and the atmosphere. The aerosol properties are then printed to provide an overview of the system.



```python
number_of_samples = 100_000  # Number of samples for particle distribution

# Generate a particle distribution using a lognormal sample distribution
# This distribution has a mean particle diameter (mode) and geometric standard deviation (GSD)
radii_sample = par.particles.get_lognormal_sample_distribution(
    mode=mode,
    geometric_standard_deviation=geometric_standard_deviation,
    number_of_particles=number_of_particles,
    number_of_samples=number_of_samples,  # Number of samples for particle distribution
)

# Calculate the mass of each particle in the sample, assuming density of 1500 kg/m^3
particle_mass_sample = (
    4 / 3 * np.pi * radii_sample**3 * density
)  # Particle mass in kg

print(f"Total mass of particles: {np.sum(particle_mass_sample):.2e} kg")
# Build a resolved mass representation for each particle
# This defines how particle mass, activity, and surface are represented
resolved_masses = (
    par.particles.ResolvedParticleMassRepresentationBuilder()
    # Use specieated mass distribution, ideal mass activity, and volume surface strategy
    .set_distribution_strategy(par.particles.ParticleResolvedSpeciatedMass())
    .set_activity_strategy(par.particles.ActivityIdealMass())
    .set_surface_strategy(par.particles.SurfaceStrategyVolume())
    .set_mass(particle_mass_sample, "kg")  # Assign mass of particles (in kg)
    .set_density(density, "kg/m^3")  # Set particle density
    .set_charge(0)  # Assume neutral particles with no charge
    .set_volume(volume, "m^3")  # Set volume of particle distribution
    .build()  # Finalize the resolved mass representation
)

# Create an aerosol object with the defined atmosphere and resolved particles
aerosol_resolved = par.Aerosol(
    atmosphere=atmosphere, particles=resolved_masses
)

# Print the properties of the aerosol
print(aerosol_resolved)
```

    Total mass of particles: 4.30e-12 kg
    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['Preset100']
    [0]: Particle Representation:
    	Strategy: ParticleResolvedSpeciatedMass
    	Activity: ActivityIdealMass
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 4.303e-06 [kg/m^3]
    	Number Concentration: 1.000e+11 [#/m^3]


**PMF**

Here, we generate a Probability Mass Function (PMF) approach. We define the radius bins using a logarithmic scale and calculate the number concentration of particles based on the total number of particles and the system volume. The PMF distribution is then built by specifying the particle mode, geometric standard deviation, and number concentration.

We set the distribution type to 'PMF' and define the radius bins, which represent the particle size categories. The particle density is also specified, and the PMF-based particle representation is finalized.

After setting up the PMF particle distribution, we create an aerosol object that combines the PMF particle properties with the previously defined atmospheric conditions. The properties of the aerosol object are printed to give a summary of the system configuration.


```python
radius_bins = np.logspace(
    -8, -6, 250
)  # Define the radius bins for the resolved mass representation

number_concentration = number_of_particles * np.array(
    [number_of_samples / volume]
)  # Calculate the number concentration of particles
print(f"Number concentration: {number_concentration[0]:.2e} m^-3")
particle_pmf = (
    par.particles.PresetParticleRadiusBuilder()
    .set_mode(mode, mode_units="m")  # Set the mode of the distribution
    .set_geometric_standard_deviation(geometric_standard_deviation)
    .set_number_concentration(
        number_concentration, "m^-3"
    )  # Set the number concentration
    .set_distribution_type("pmf")  # Set the distribution type to PMF
    .set_radius_bins(radius_bins, radius_bins_units="m")  # Set the radius bins
    .set_density(density, "kg/m^3")  # Set particle density
    .build()  # Finalize the resolved mass representation
)

# Create an aerosol object with the defined atmosphere and resolved particles
aerosol_pmf = par.Aerosol(atmosphere=atmosphere, particles=particle_pmf)

# Print the properties of the aerosol
print(aerosol_pmf)
```

    Number concentration: 7.50e+10 m^-3
    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['Preset100']
    [0]: Particle Representation:
    	Strategy: RadiiBasedMovingBin
    	Activity: ActivityIdealMass
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 4.282e-06 [kg/m^3]
    	Number Concentration: 1.000e+11 [#/m^3]


**PDF**

Lastly, we generate a particle distribution using the Probability Density Function (PDF) approach. Similar to the PMF setup, we specify the particle mode, geometric standard deviation, and number concentration. However, in this case, the distribution type is set to "PDF", which models the particle distribution as a continuous probability density function over the defined radius bins.

We assign the same logarithmic radius bins as before, specify the particle density, and assume the particles are neutral by setting their charge to zero. After defining all necessary parameters, we finalize the PDF-based particle representation.

As with the PMF approach, we create an aerosol object by combining the PDF-based particle distribution with the predefined atmospheric conditions. The properties of the resulting aerosol are printed to summarize the system configuration.



```python
particle_pdf = (
    par.particles.PresetParticleRadiusBuilder()
    .set_mode(mode, mode_units="m")  # Set the mode of the distribution
    .set_geometric_standard_deviation(geometric_standard_deviation)
    .set_number_concentration(
        number_concentration, "m^-3"
    )  # Set the number concentration
    .set_distribution_type("pdf")  # Set the distribution type to PMF
    .set_radius_bins(radius_bins, radius_bins_units="m")  # Set the radius bins
    .set_density(density, "kg/m^3")  # Set particle density
    .set_charge(
        np.zeros_like(radius_bins)
    )  # Assume neutral particles with no charge
    .build()  # Finalize the resolved mass representation
)

# Create an aerosol object with the defined atmosphere and resolved particles
aerosol_pdf = par.Aerosol(atmosphere=atmosphere, particles=particle_pdf)

# Print the properties of the aerosol
print(aerosol_pdf)
```

    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['Preset100']
    [0]: Particle Representation:
    	Strategy: RadiiBasedMovingBin
    	Activity: ActivityIdealMass
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 7.797e+02 [kg/m^3]
    	Number Concentration: 4.663e+19 [#/m^3]


**Plot Initial Distributions**

In this section, we plot the initial particle distributions for the PMF, and particle-resolved approaches. The plots show the number concentration of particles as a function of particle radius for each method. The PMF is a line, and the particle-resolved has been binned into discrete sizes.


```python
# plot both
radius_bins = particle_pmf.get_radius()
radii_resolved = resolved_masses.get_radius()

fig, ax = plt.subplots()
bins, edges = np.histogram(radii_resolved, bins=radius_bins)
ax.bar(
    edges[:-1],
    bins / volume,
    width=np.diff(edges),
    align="edge",
    label="Resolved",
    alpha=0.7,
)
ax.plot(
    radius_bins,
    particle_pmf.get_concentration(),
    label="PMF",
    color="red",
)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Number concentration (m^-3)")
ax.legend()
plt.show()
```


    
![png](output_11_0.png)
    


## Simulate Coagulation

In this section, we simulate the coagulation process for the PMF, PDF, and particle-resolved approaches. We define the time step for the simulation and the total simulation time. The time step is used to update the particle distribution at regular intervals, while the total simulation time determines the duration of the coagulation process.


```python
# simulate aerosols, and save total mass and number distribution

# Define the coagulation process
coagulation_process_pmf = par.dynamics.Coagulation(
    coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
        distribution_type="discrete"
    )
)

coagulation_process_resolved = par.dynamics.Coagulation(
    coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
        distribution_type="particle_resolved"
    )
)
coagulation_process_pdf = par.dynamics.Coagulation(
    coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
        distribution_type="continuous_pdf"
    )
)

# Set up time and sub-steps for the coagulation process
total_time = 10000
time_step = 100
sub_steps = 1

# output arrays
time = np.arange(0, total_time, time_step)
total_mass_pmf = np.zeros_like(time, dtype=np.float64)
total_mass_resolved = np.ones_like(time, dtype=np.float64)
total_mass_pdf = np.zeros_like(time, dtype=np.float64)
number_distribution_pmf = np.zeros((len(time), len(radius_bins)))
number_distribution_resolved = np.zeros((len(time), number_of_samples))
number_distribution_pdf = np.zeros((len(time), len(radius_bins)))
total_number_pmf = np.zeros_like(time, dtype=np.float64)
total_number_resolved = np.ones_like(time, dtype=np.float64)
total_number_pdf = np.zeros_like(time, dtype=np.float64)
```

**Simulation Loop**

We then run a simulation loop that iterates over the specified time range, updating the particle distribution at each time step.


```python
# Simulation loop

for i, t in enumerate(time):
    if i > 0:
        # Perform coagulation for the PDF aerosol
        aerosol_pdf = coagulation_process_pdf.execute(
            aerosol_pdf, time_step, sub_steps
        )
        # Perform coagulation for the PMF aerosol
        aerosol_pmf = coagulation_process_pmf.execute(
            aerosol_pmf, time_step, sub_steps
        )
        # Perform coagulation for the resolved aerosol
        aerosol_resolved = coagulation_process_resolved.execute(
            aerosol_resolved, time_step, sub_steps
        )

    total_mass_resolved[i] = aerosol_resolved.particles[
        0
    ].get_mass_concentration()
    number_distribution_resolved[i, :] = aerosol_resolved.particles[
        0
    ].get_radius(clone=True)
    total_number_resolved[i] = np.sum(number_distribution_resolved[i, :] > 0)

    total_mass_pmf[i] = aerosol_pmf.particles[0].get_mass_concentration()
    number_distribution_pmf[i, :] = aerosol_pmf.particles[0].get_concentration(
        clone=True
    )
    total_number_pmf[i] = np.sum(number_distribution_pmf[i, :])

    total_mass_pdf[i] = aerosol_pdf.particles[0].get_mass_concentration()
    number_distribution_pdf[i, :] = aerosol_pdf.particles[0].get_concentration(
        clone=True
    )
    total_number_pdf[i] = np.trapezoid(
        number_distribution_pdf[i, :], radius_bins
    )
```

## Results

The results of the coagulation simulations are presented in this section. We compare the number and mass of particles for the PMF, PDF, and particle-resolved approaches at different time points during the coagulation process.

The fist thing to check is the final state of the aerosol system.



```python
print(aerosol_resolved)
print(aerosol_pmf)
# print(aerosol_pdf)
```

    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['Preset100']
    [0]: Particle Representation:
    	Strategy: ParticleResolvedSpeciatedMass
    	Activity: ActivityIdealMass
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 4.302e-06 [kg/m^3]
    	Number Concentration: 6.253e+10 [#/m^3]
    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['Preset100']
    [0]: Particle Representation:
    	Strategy: RadiiBasedMovingBin
    	Activity: ActivityIdealMass
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 4.242e-06 [kg/m^3]
    	Number Concentration: 6.259e+10 [#/m^3]



```python
# plot the initial and final distributions
fig, ax = plt.subplots(figsize=(8, 5))

bins, edges = np.histogram(
    number_distribution_resolved[0, :], bins=radius_bins
)
ax.bar(
    edges[:-1],
    bins / volume,
    width=np.diff(edges),
    align="edge",
    label="Resolved initial",
    color="red",
    alpha=0.7,
)
bins, edges = np.histogram(
    number_distribution_resolved[-1, :], bins=radius_bins
)
ax.bar(
    edges[:-1],
    bins / volume,
    width=np.diff(edges),
    align="edge",
    label="Resolved final",
    color="blue",
    alpha=0.7,
)

ax.plot(
    radius_bins,
    number_distribution_pmf[0, :],
    label="PMF initial",
    color="red",
)
ax.plot(
    radius_bins,
    number_distribution_pmf[-1, :],
    label="PMF final",
    color="blue",
)

ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Number concentration (m^-3)")
ax.legend()
plt.show()
```


    
![png](output_18_0.png)
    


**Plot Mass Conservation**

In and ideal system, the mass should be conserved. In this section, we plot the mass conservation for the PMF, PDF, and particle-resolved approaches. The plots show the total mass of particles as a function of time during the coagulation process. The mass should remain constant over time, indicating that mass is conserved in the system.

In our case, there is some numerical error in the mass conservation, this is particularly evident in the PMF method.


```python
# mass conservation plot

mass_pmf_error = (total_mass_pmf - total_mass_pmf[0]) / total_mass_pmf[0]
mass_resolved_error = (
    total_mass_resolved - total_mass_resolved[0]
) / total_mass_resolved[0]


fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(time, mass_pmf_error, label="Probability Mass Function")
ax.plot(time, mass_resolved_error, label="Particle Resolved")
# ax.set_yscale("log")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mass Error (relative to start)")
ax.set_title("Error in mass conservation")
ax.legend()
plt.show()
```


    
![png](output_20_0.png)
    


**Plot Number Error**

For the number concentration, we use the PDF distribution as the reference. We calculate the percent error in the number concentration for the PMF and particle-resolved approaches compared to the PDF distribution.


```python
# sum number concentration and plot

total_pmf = np.sum(number_distribution_pmf, axis=1)
total_resolved_non_zero = number_distribution_resolved > 0
total_resolved = np.sum(total_resolved_non_zero, axis=1) / volume

percent_diff_resolved = (
    (total_number_pdf - total_resolved) / total_number_pdf * 100
)
percent_diff_pmf = (total_number_pdf - total_pmf) / total_number_pdf * 100

print(
    f"Resolved number final: {total_resolved[-1]:.2e}, PMF number final: {total_pmf[-1]:.2e}"
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(time, percent_diff_resolved, label="Particle Resolved", linestyle="--")
ax.plot(
    time, percent_diff_pmf, label="Probability Mass Function", linestyle="--"
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Percent Difference vs Probability Density Function")
ax.set_title("Numerical Error in Number Concentration")
ax.legend()
plt.show()
```

    Resolved number final: 6.25e+10, PMF number final: 6.26e+10



    
![png](output_22_1.png)
    


## Conclusion

In this notebook, we compared the PMF, PDF, and particle-resolved approaches for modeling particle distributions and the coagulation process. We found that each method has its strengths and limitations, with the PMF and PDF approaches providing a more continuous representation of the particle distribution, while the particle-resolved approach offers a more detailed view of individual particles.

They all have numerical errors, but the PDF method is the most accurate in terms of mass conservation and number concentration. The PMF method has the largest error in mass concentration. The particle-resolved method has variable error in number concentration.


---
# Coagulation_Basic_1_PMF.md

# Coagulation Basic 1: PMF Representation

## Direct Implementation for Discrete Particle Size Distributions

In this section, we present a direct implementation of the coagulation process based on the methodology outlined in Seinfeld and Pandis (2006). This notebook is designed to provide a clear understanding of the fundamental code required to simulate particle coagulation, without the abstraction layers provided by more advanced object-oriented approaches.

### Objective

The primary goal is to demonstrate how to calculate the coagulation kernel and the resulting coagulation rates for a given particle size distribution. We will start with a uniform size bin and distribution to establish the basic principles. Then, we will extend this to a more realistic lognormal distribution, which is commonly observed in aerosol science.

### Approach

- **Uniform Distribution**: We first initialize a simple, uniform particle size distribution. This helps in understanding the basic coagulation mechanics.
- **Lognormal Probability Mass Function**: After establishing the basics, we move on to a lognormal size distribution, which better represents real-world aerosol size distributions. Probability mass functions (PMFs) is a bin representation of the lognormal distribution, so the sum of all bins equals total concentration.

This step-by-step approach will provide a foundation for understanding more complex implementations, such as those available in the `particula` library.

### Required Libraries

We will utilize standard scientific libraries such as `numpy` for numerical computations, `matplotlib` for plotting, and `pandas` for data handling. Additionally, we import specific functions from the `particula` library to calculate the coagulation kernel and generate size distributions.



```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import particula as par
```

## Creating a Size Distribution

In this section, we define the size distribution for aerosol particles. The particle sizes are distributed across several bins, allowing us to model the behavior of particles across a wide size range.

### Particle Size Bins

We first define the bins for particle radii using a logarithmic scale. The logarithmic scale (`np.logspace`) is particularly useful when dealing with aerosol particles because their sizes often span several orders of magnitude—from nanometers to micrometers. This approach ensures that we capture the full range of particle sizes with more granularity where it matters.

- **`radius_bins`**: These bins represent the particle radii, ranging from 1 nanometer (1e-9 m) to 10 micrometers (1e-5 m).

### Particle Mass Calculation

Next, we calculate the mass of particles in each size bin. The mass is determined using the formula for the volume of a sphere (`4/3 * π * r^3`), assuming a uniform particle density of 1 g/cm³ (which corresponds to 1000 kg/m³ in SI units).

- **`mass_bins`**: The resulting array contains the masses of particles corresponding to each size bin, which is a key parameter for understanding how these particles will interact and coagulate.

This setup provides a foundation for further analysis of the coagulation process by linking particle size and mass, essential components in determining coagulation rates.



```python
# Create a size distribution for aerosol particles

# Define the bins for particle radius using a logarithmic scale
radius_bins = np.logspace(start=-9, stop=-5, num=10)  # m (1 nm to 10 μm)

# Calculate the mass of particles for each size bin
# The mass is calculated using the formula for the volume of a sphere (4/3 * π * r^3)
# and assuming a particle density of 1 g/cm^3 (which is 1000 kg/m^3 in SI units).
mass_bins = 4 / 3 * np.pi * radius_bins**3 * 1e3  # kg
```

## Calculating the Brownian Coagulation Kernel

In this section, we calculate the Brownian coagulation kernel matrix, which quantifies the rate at which particles of different sizes collide and coagulate due to Brownian motion. The kernel matrix is central to understanding the dynamics of particle coagulation.

### Kernel Calculation

We use the `brownian_coagulation_kernel_via_system_state` function from the `particula` library to compute the kernel matrix. This function requires the following inputs:

- **`particle_radius`**: The array of particle radii, which we previously defined in `radius_bins`.
- **`mass_particle`**: The corresponding array of particle masses from `mass_bins`.
- **`temperature`**: The ambient temperature, set here to 293.15 K (equivalent to 20°C).
- **`pressure`**: The ambient pressure, set to 101325 Pa (standard atmospheric pressure).
- **`alpha_collision_efficiency`**: A dimensionless factor representing the efficiency of particle collisions, assumed to be 1.0 for perfect efficiency.

The output is a matrix (kernel) where each element represents the coagulation rate between two specific particle sizes.

### Analyzing the Kernel

We print the shape of the kernel matrix to verify its dimensions, which should match the number of radius bins (i.e., it will be a square matrix).

### Creating a DataFrame

To facilitate analysis, we convert the kernel matrix into a `pandas` DataFrame. The DataFrame format allows for easy manipulation and visualization of the data. The rows and columns of the DataFrame are indexed by the particle radii, making it straightforward to identify which particle sizes are interacting.

We then print the first five rows of the DataFrame to inspect the calculated values. This provides a quick glimpse into the interaction rates between the smallest particles.

### Optional: Saving the Kernel Matrix

For further analysis or to share with others, the kernel matrix can be saved as a CSV file. This step is optional but useful if you need to persist the results for future work.



```python
# Calculate the Brownian coagulation kernel matrix

# The Brownian coagulation kernel is calculated using the `brownian_coagulation_kernel_via_system_state` function.
# This function takes into account the particle size, mass, temperature, pressure, and collision efficiency
# to compute the coagulation rates between particles of different sizes.
kernel = par.dynamics.get_brownian_kernel_via_system_state(
    particle_radius=radius_bins,
    mass_particle=mass_bins,
    temperature=293.15,  # Temperature in Kelvin (20°C)
    pressure=101325,  # Pressure in Pascals (1 atm)
    alpha_collision_efficiency=1.0,  # Assume perfect collision efficiency
)

# Display the shape of the kernel matrix to confirm its dimensions
print(f"Kernel shape: {kernel.shape}")

# Create a pandas DataFrame from the kernel matrix
# The DataFrame allows for easier analysis and visualization of the coagulation kernel.
# Rows and columns are indexed by the particle radius bins, making it clear which sizes are interacting.
df_kernel = pd.DataFrame(kernel, index=radius_bins, columns=radius_bins)

# Print the first 5 rows of the DataFrame to inspect the computed kernel values
df_kernel.head(5)

# Optional: Save the kernel matrix to a CSV file for further analysis or sharing
# df_kernel.to_csv("kernel.csv")
```

    Kernel shape: (10, 10)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1.000000e-09</th>
      <th>2.782559e-09</th>
      <th>7.742637e-09</th>
      <th>2.154435e-08</th>
      <th>5.994843e-08</th>
      <th>1.668101e-07</th>
      <th>4.641589e-07</th>
      <th>1.291550e-06</th>
      <th>3.593814e-06</th>
      <th>1.000000e-05</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.000000e-09</th>
      <td>8.812734e-16</td>
      <td>2.277171e-15</td>
      <td>1.181152e-14</td>
      <td>7.503580e-14</td>
      <td>4.506909e-13</td>
      <td>2.024284e-12</td>
      <td>6.824175e-12</td>
      <td>2.020759e-11</td>
      <td>5.737860e-11</td>
      <td>1.607645e-10</td>
    </tr>
    <tr>
      <th>2.782559e-09</th>
      <td>2.277171e-15</td>
      <td>1.461722e-15</td>
      <td>3.692369e-15</td>
      <td>1.733024e-14</td>
      <td>8.141373e-14</td>
      <td>3.005388e-13</td>
      <td>9.275265e-13</td>
      <td>2.670597e-12</td>
      <td>7.517319e-12</td>
      <td>2.100038e-11</td>
    </tr>
    <tr>
      <th>7.742637e-09</th>
      <td>1.181152e-14</td>
      <td>3.692369e-15</td>
      <td>2.224511e-15</td>
      <td>4.558035e-15</td>
      <td>1.429060e-14</td>
      <td>4.375392e-14</td>
      <td>1.267816e-13</td>
      <td>3.580220e-13</td>
      <td>1.001331e-12</td>
      <td>2.790647e-12</td>
    </tr>
    <tr>
      <th>2.154435e-08</th>
      <td>7.503580e-14</td>
      <td>1.733024e-14</td>
      <td>4.558035e-15</td>
      <td>2.123764e-15</td>
      <td>3.008409e-15</td>
      <td>7.020882e-15</td>
      <td>1.858550e-14</td>
      <td>5.087944e-14</td>
      <td>1.407135e-13</td>
      <td>3.904922e-13</td>
    </tr>
    <tr>
      <th>5.994843e-08</th>
      <td>4.506909e-13</td>
      <td>8.141373e-14</td>
      <td>1.429060e-14</td>
      <td>3.008409e-15</td>
      <td>1.304527e-15</td>
      <td>1.624475e-15</td>
      <td>3.413800e-15</td>
      <td>8.588241e-15</td>
      <td>2.303139e-14</td>
      <td>6.320152e-14</td>
    </tr>
  </tbody>
</table>
</div>



## Plotting the Brownian Coagulation Kernel

After calculating the Brownian coagulation kernel, the next step is to visualize the results. A plot of the kernel values against the particle radius provides insights into how the coagulation rates vary with particle size.

### Visualization Setup

We use `matplotlib` to create the plot:

- **Figure and Axis**: We begin by creating a figure and an axis object using `plt.subplots()`, which provides a flexible framework for plotting.
- **Kernel Plot**: The kernel values are plotted on the y-axis, and the particle radii on the x-axis. Since the kernel values can span several orders of magnitude, we plot their logarithm (base 10) to better visualize the data.
  
### Axes Configuration

- **X-axis**: The x-axis represents the particle radius in meters. Given the wide range of particle sizes, we use a logarithmic scale (`set_xscale("log")`) to evenly distribute the data across the axis.
- **Y-axis**: The y-axis represents the logarithm of the kernel values (`Log10(Kernel)`). This choice makes it easier to observe trends and differences in the coagulation rates across different particle sizes.

### Optional Y-axis Logarithmic Scale

If a deeper examination of the kernel's range is required, the y-axis can also be set to a logarithmic scale by uncommenting the `ax.set_yscale("log")` line. This is useful when the kernel values span several orders of magnitude and need to be visualized more clearly.



```python
# Plot the Brownian coagulation kernel

# Create a figure and axis object using matplotlib
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the kernel values against the particle radius bins
# The kernel values are plotted on a logarithmic scale (log10) for better visualization.
ax.plot(radius_bins, np.log10(kernel))

# Set the x-axis label to indicate the particle radius in meters
ax.set_xlabel("Particle radius (m)")

# Set the y-axis label to indicate the logarithm of the kernel values
ax.set_ylabel("Log10(Kernel)")

# Use a logarithmic scale for the x-axis to properly represent the wide range of particle sizes
ax.set_xscale("log")

# Optionally, the y-axis can also be set to a logarithmic scale if needed
# ax.set_yscale("log")

# Display the plot
plt.show()
```


    
![png](output_7_0.png)
    


## Adding Particle Concentrations

Now that we have calculated the Brownian coagulation kernel, the next step is to introduce the particle concentrations. While the coagulation kernel itself does not depend on the particle concentration, the concentrations are critical when calculating the actual rate of coagulation, as they determine how many particles are available to interact.

### Initial Concentration Setup

We define an initial particle concentration for each size bin:

- **Uniform Concentration**: In this example, we set a uniform concentration across all size bins. Specifically, we assign a concentration of 100 particles per cubic centimeter (100 cm^-3), which converts to 100 million particles per cubic meter (100 * 1e6 m^-3). This concentration is representative of a typical ambient particle concentration in the atmosphere.

### Visualization of the Initial Concentration

To better understand the initial distribution of particle concentrations, we plot these values against the particle radius:

- **X-axis**: The x-axis represents the particle radius in meters, using a logarithmic scale to cover the wide range of particle sizes.
- **Y-axis**: The y-axis shows the particle concentration in particles per cubic meter (m^-3), also plotted on a logarithmic scale. The logarithmic scales on both axes allow us to clearly see the distribution across several orders of magnitude, which is common in aerosol science.

### Importance of Concentration

While the kernel function determines the rate at which particles of different sizes



```python
# Define the initial particle concentration

# Set the initial concentration for each size bin
# The concentration is set uniformly across all bins at 100 particles per cubic centimeter (100 cm^-3),
# which is equivalent to 100 * 1e6 particles per cubic meter (m^-3).
concentration_0 = np.ones_like(radius_bins) * 100 * 1e6  # m^-3

# Plot the initial concentration distribution

# Create a figure and axis object for the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the concentration against the particle radius
ax.plot(radius_bins, concentration_0)

# Set the x-axis label to indicate the particle radius in meters
ax.set_xlabel("Particle radius (m)")

# Set the y-axis label to indicate the concentration in particles per cubic meter
ax.set_ylabel("Concentration (m^-3)")

# Use a logarithmic scale for both the x-axis and y-axis
# This is because the concentration distribution is typically viewed across several orders of magnitude
ax.set_xscale("log")
ax.set_yscale("log")

# Display the plot
plt.show()
```


    
![png](output_9_0.png)
    


## Coagulation Rate Calculation

With the coagulation kernel and initial concentrations defined, we can now calculate the rates at which particles are gained, lost, and the net change due to coagulation. These rates are essential for understanding how the particle size distribution evolves over time as particles collide and coagulate.

### Gain, Loss, and Net Rate Calculation

- **Gain (`discrete_gain`)**: This function calculates the rate at which particles are gained in each size bin due to coagulation. Gain occurs when two smaller particles collide to form a larger particle, increasing the concentration in the corresponding size bin.
  
- **Loss (`discrete_loss`)**: This function calculates the rate at which particles are lost from each size bin due to coagulation. Loss happens when particles in a particular size bin collide with other particles, thereby decreasing the concentration in that bin.
  
- **Net Rate**: The net rate is the difference between the gain and loss for each size bin (`net_0 = gain_0 - loss_0`). It represents the overall change in concentration for each particle size due to coagulation.

### Displaying Results

We create a `pandas` DataFrame to organize and display the gain, loss, and net coagulation rates for each particle size bin. The DataFrame is indexed by particle radius, which makes it easy to understand the changes in concentration across different sizes.

- **Gain**: The rate at which particles are added to each bin due to coagulation.
- **Loss**: The rate at which particles are removed from each bin due to coagulation.
- **Net**: The overall change in concentration for each bin.

Finally, we display the first five rows of the DataFrame to inspect the initial values for gain, loss, and net change. This provides a quick look at how coagulation is expected to alter the particle size distribution in the system.

By analyzing these rates, we can predict the dynamic behavior of the aerosol particles over time, as smaller particles merge to form larger ones or disappear from the system.




```python
# Coagulation rate calculation

# Calculate the gain, loss, and net change in particle concentration due to coagulation
# `discrete_gain`: Calculates the rate at which particles are gained due to coagulation
# `discrete_loss`: Calculates the rate at which particles are lost due to coagulation
gain_0 = par.dynamics.get_coagulation_gain_rate_discrete(
    radius_bins, concentration_0, kernel
)
loss_0 = par.dynamics.get_coagulation_loss_rate_discrete(
    concentration_0, kernel
)
net_0 = gain_0 - loss_0  # Net change in concentration

# Create a DataFrame to display the gain, loss, and net coagulation rates
# The DataFrame is indexed by particle radius bins for clarity
df = pd.DataFrame(
    data={"Gain": gain_0, "Loss": loss_0, "Net": net_0}, index=radius_bins
)

# Display the first 5 rows of the DataFrame
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gain</th>
      <th>Loss</th>
      <th>Net</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.000000e-09</th>
      <td>4.379653</td>
      <td>2.477398e+06</td>
      <td>-2.477394e+06</td>
    </tr>
    <tr>
      <th>2.782559e-09</th>
      <td>25.830601</td>
      <td>3.252254e+05</td>
      <td>-3.251995e+05</td>
    </tr>
    <tr>
      <th>7.742637e-09</th>
      <td>79.877700</td>
      <td>4.357112e+04</td>
      <td>-4.349124e+04</td>
    </tr>
    <tr>
      <th>2.154435e-08</th>
      <td>1666.815122</td>
      <td>7.097477e+03</td>
      <td>-5.430662e+03</td>
    </tr>
    <tr>
      <th>5.994843e-08</th>
      <td>-12799.281898</td>
      <td>6.505676e+03</td>
      <td>-1.930496e+04</td>
    </tr>
  </tbody>
</table>
</div>



## Plotting the Coagulation Gain, Loss, and Net Rates

After calculating the coagulation rates, it's crucial to visualize how these rates vary across different particle sizes. This plot will show the gain, loss, and net rates of particle concentration as a function of particle radius.

### Plot Details

- **Net Rate**: The net rate of change in particle concentration is plotted as a thick gray line. This line highlights the overall effect of coagulation, showing whether the concentration in each size bin is increasing or decreasing.
- **Gain Rate**: The gain rate is plotted in dark green. This line shows how particles are being added to each size bin as smaller particles coagulate to form larger ones.
- **Loss Rate**: The loss rate is plotted in red. To make it visually distinct and indicate that it's a removal process, the loss rate is plotted as `-1 * loss_0`. This negative value reflects the decrease in particle concentration due to coagulation.

### Axes Configuration

- **X-axis**: The x-axis represents the particle radius in meters, plotted on a logarithmic scale. This scale is essential for accurately representing the wide range of particle sizes, from nanometers to micrometers.
  
- **Y-axis**: The y-axis shows the rate of change in concentration, with units of `m⁻³ s⁻¹`, indicating how quickly particles are being gained, lost, or changing net concentration in the system.

### Interpretation

By analyzing this plot, you can determine which particle sizes are most affected by coagulation. For instance, sizes where the net rate is positive indicate that coagulation is leading to an increase in concentration, while negative values suggest a decrease. This visualization is crucial for understanding the evolution of the particle size distribution over time.



```python
# Plot the coagulation gain, loss, and net rates

# Create a figure and axis object for the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the net coagulation rate
# The net rate is plotted with a thicker gray line for emphasis
ax.plot(radius_bins, net_0, label="Net", color="gray", linewidth=4)

# Plot the gain rate
# The gain rate is plotted in dark green
ax.plot(radius_bins, gain_0, label="Gain", color="darkgreen")

# Plot the loss rate
# The loss rate is plotted in red, and multiplied by -1 for plotting to indicate that it's a removal process
ax.plot(radius_bins, -1 * loss_0, label="Loss", color="red")

# Set the x-axis label to indicate the particle radius in meters
ax.set_xlabel("Particle radius (m)")

# Set the y-axis label to indicate the rate of change in concentration, with appropriate units
ax.set_ylabel(r"Rate $\dfrac{1}{m^{3} s^{1}}$")

# Use a logarithmic scale for the x-axis to account for the wide range of particle sizes
ax.set_xscale("log")

# Add a legend to the plot to identify the gain, loss, and net lines
plt.legend()

# Display the plot
plt.show()
```


    
![png](output_13_0.png)
    


## Simulating the Coagulation Process Over Time

In this section, we manually simulate the coagulation process over a few discrete time steps. This manual simulation allows us to observe how particle concentrations evolve as a result of coagulation.

### Simulation Setup

- **Time Step**: We define a time step of 0.1 seconds (`TIME_STEP = 0.1`). This value represents the interval at which we update the particle concentrations based on the coagulation rates.

### Time Step 1

1. **Calculate Gain and Loss**: Using the initial concentration (`concentration_0`), we calculate the gain and loss rates using the coagulation kernel.
2. **Net Rate**: The net rate of change in concentration is determined by subtracting the loss from the gain.
3. **Update Concentration**: The concentration is updated by adding the net rate multiplied by the time step to the initial concentration, resulting in `concentration_1`.

### Time Step 2

1. **Recalculate Gain and Loss**: With the updated concentration from time step 1 (`concentration_1`), we recalculate the gain and loss rates.
2. **Net Rate**: Again, we calculate the net rate of change.
3. **Update Concentration**: The concentration is updated to `concentration_2` using the same method as in time step 1.

### Time Step 3

1. **Recalculate Gain and Loss**: We perform the same calculations with the concentration from time step 2 (`concentration_2`).
2. **Update Concentration**: The concentration is updated to `concentration_3`.

### Observing Changes

We print the maximum concentration at each time step to observe how the distribution evolves due to coagulation. This can provide insights into how quickly particles are coalescing into larger sizes or being depleted.

### DataFrame Creation

The concentrations at each time step are combined into a `pandas` DataFrame, making it easier to compare how the distribution changes over time. We display the first few rows to inspect these changes.

### Optional: Saving Results

The concentration data can be saved to a CSV file for further analysis or visualization, allowing you to track the evolution of particle concentrations over time.

### Interpretation

By manually simulating the coagulation process, we can see the step-by-step changes in particle concentrations. This approach highlights the dynamic nature of coagulation and how it impacts particle size distributions in aerosols over time.



```python
# Simulating the coagulation process over time manually

# Define the time step for the simulation
TIME_STEP = 0.1  # seconds

# Time step 1: Calculate the gain, loss, and net rate, then update concentration

# Calculate the rate of change in concentration (gain and loss) for the initial concentration
gain = par.dynamics.get_coagulation_gain_rate_discrete(
    radius_bins, concentration_0, kernel
)
loss = par.dynamics.get_coagulation_loss_rate_discrete(concentration_0, kernel)
net = gain - loss  # Net rate of change

# Update the concentration for the next time step
concentration_1 = concentration_0 + net * TIME_STEP

# Time step 2: Recalculate rates with the updated concentration and update again

# Recalculate gain and loss based on the updated concentration from time step 1
gain = par.dynamics.get_coagulation_gain_rate_discrete(
    radius_bins, concentration_1, kernel
)
loss = par.dynamics.get_coagulation_loss_rate_discrete(concentration_1, kernel)
net = gain - loss

# Update the concentration for the next time step
concentration_2 = concentration_1 + net * TIME_STEP

# Time step 3: Recalculate rates again and update concentration

# Recalculate gain and loss based on the updated concentration from time step 2
gain = par.dynamics.get_coagulation_gain_rate_discrete(
    radius_bins, concentration_2, kernel
)
loss = par.dynamics.get_coagulation_loss_rate_discrete(concentration_2, kernel)
net = gain - loss

# Update the concentration for the next time step
concentration_3 = concentration_2 + net * TIME_STEP

# Print the maximum concentration at each time step to observe changes
print(f"Concentration 0 max: {concentration_0.max()}")
print(f"Concentration 1 max: {concentration_1.max()}")
print(f"Concentration 2 max: {concentration_2.max()}")
print(f"Concentration 3 max: {concentration_3.max()}")

# Combine the concentrations at each time step into a DataFrame for easy comparison
df_concentration = pd.DataFrame(
    {
        "0": concentration_0,
        "1": concentration_1,
        "2": concentration_2,
        "3": concentration_3,
    },
    index=radius_bins,
)

# Display the first five rows of the DataFrame to inspect the concentration changes
df_concentration.head(5)

# Optional: Save the concentration data to a CSV file for further analysis
# df_concentration.to_csv("concentration_uniform_sim.csv")
```

    Concentration 0 max: 100000000.0
    Concentration 1 max: 1263633883.0015862
    Concentration 2 max: 11241880825.442844
    Concentration 3 max: 98778247873.61397





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.000000e-09</th>
      <td>100000000.0</td>
      <td>9.975226e+07</td>
      <td>9.765578e+07</td>
      <td>7.995004e+07</td>
    </tr>
    <tr>
      <th>2.782559e-09</th>
      <td>100000000.0</td>
      <td>9.996748e+07</td>
      <td>9.969288e+07</td>
      <td>9.733161e+07</td>
    </tr>
    <tr>
      <th>7.742637e-09</th>
      <td>100000000.0</td>
      <td>9.999565e+07</td>
      <td>9.995912e+07</td>
      <td>9.964448e+07</td>
    </tr>
    <tr>
      <th>2.154435e-08</th>
      <td>100000000.0</td>
      <td>9.999946e+07</td>
      <td>9.999441e+07</td>
      <td>9.995043e+07</td>
    </tr>
    <tr>
      <th>5.994843e-08</th>
      <td>100000000.0</td>
      <td>9.999807e+07</td>
      <td>9.999542e+07</td>
      <td>9.998650e+07</td>
    </tr>
  </tbody>
</table>
</div>



## Plotting the Evolution of Particle Concentration

To visualize how particle concentrations evolve over time due to coagulation, we plot the concentration distributions at different time steps. This allows us to observe the changes in particle size distribution as the coagulation process progresses.

### Plot Details

- **Time Step 0 (`t=0`)**: The initial concentration distribution is plotted as a solid blue line. This serves as the baseline before any coagulation has occurred.
  
- **Time Step 1 (`t=1`)**: After the first time step, the concentration distribution is plotted as a dashed green line. This shows the immediate effect of coagulation on the particle distribution.
  
- **Time Step 2 (`t=2`)**: The concentration distribution at the second time step is plotted as a dash-dot orange line. By this point, we can begin to see more noticeable changes as particles coagulate.
  
- **Time Step 3 (`t=3`)**: Finally, the concentration distribution after the third time step is plotted as a dotted red line, illustrating further evolution of the particle sizes as the coagulation process continues.

### Axes Configuration

- **X-axis**: The x-axis represents the particle radius in meters, and is plotted on a logarithmic scale to capture the wide range of particle sizes.
  
- **Y-axis**: The y-axis shows the concentration in particles per cubic meter (m⁻³), also on a logarithmic scale to reflect the changes in concentration across orders of magnitude.

### Interpretation

As expected, smaller particles tend to coagulate into larger sizes, leading to changes in the overall distribution. This plot provides a visual representation of the dynamic nature of coagulation, making it easier to understand how particle populations evolve in an aerosol system.

The plot provides a visual representation of the dynamic nature of coagulation, making it easier to understand how particle populations evolve in an aerosol system.

### End member Error

Due to the underlying numerical integration assumptions, a flat distribution is not treated correctly at the largest sizes. But in real-world cases where the distribution is not flat, this error is not significant.



```python
# Plot the evolution of particle concentration over time

# Create a figure and axis object for the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the concentration distribution at each time step
ax.plot(radius_bins, concentration_0, label="t=0", linestyle="-", color="blue")
ax.plot(
    radius_bins, concentration_1, label="t=1", linestyle="--", color="green"
)
ax.plot(
    radius_bins, concentration_2, label="t=2", linestyle="-.", color="orange"
)
ax.plot(radius_bins, concentration_3, label="t=3", linestyle=":", color="red")

# Set the x-axis label to indicate the particle radius in meters
ax.set_xlabel("Particle radius (m)")

# Set the y-axis label to indicate the concentration in particles per cubic meter
ax.set_ylabel(r"Concentration $\dfrac{1}{m^{3}}$")

# Use logarithmic scales for both the x-axis and y-axis
# This helps in visualizing the broad range of particle sizes and concentration changes
ax.set_xscale("log")
ax.set_yscale("log")

# Add a legend to differentiate between the time steps
plt.legend()

# Display the plot
plt.show()
```


    
![png](output_17_0.png)
    


## Lognormal Distribution

In this section, we will implement the coagulation process for a lognormal particle size distribution. A lognormal distribution is more representative of real-world aerosol systems, where particles are not uniformly distributed in size but instead follow a distribution where most particles are centered around a particular size with fewer particles in the smaller and larger size ranges.

### Why Lognormal Distribution?

A lognormal distribution is often observed in natural aerosol populations due to the multiplicative processes that govern particle formation and growth. This distribution is characterized by a peak (or mode) at the most common particle size, with the number of particles decreasing logarithmically for sizes smaller and larger than the mode. Implementing coagulation for this distribution will provide a more realistic simulation of how aerosol particles behave in the atmosphere.

We will proceed by defining the lognormal size distribution, calculating the coagulation kernel, and then simulating the coagulation process over time, similar to what we did for the uniform distribution. This approach will allow us to compare the results between the uniform and lognormal distributions, highlighting the differences in coagulation dynamics based on initial particle size distributions.

### Defining the Radius and Mass Bins

- **Radius Bins**: We define the particle radius bins on a logarithmic scale ranging from 1 nanometer (1e-9 m) to 10 micrometers (1e-4 m). Using 250 bins ensures a fine resolution across this range, which is important for accurately representing the lognormal distribution.

### Generating the Lognormal Distribution

- **Lognormal Distribution**: We generate the particle concentration using a lognormal distribution, which is characterized by:
  - A **mode** (most probable size) of 100 nanometers (100 nm).
  - A **geometric standard deviation** of 1.4, which controls the spread of the distribution.
  - A **total number concentration** of 10000 particles per cubic centimeter (10000 cm⁻³), converted to particles per cubic meter for consistency with our units.

### Plotting the Lognormal Distribution

The resulting lognormal distribution is plotted with particle radius on the x-axis (logarithmic scale) and particle concentration on the y-axis. This plot visually demonstrates the lognormal distribution, showing a peak around the mode (100 nm) with concentrations decreasing for both smaller and larger particles.

This setup provides a more realistic starting point for simulating the coagulation process in an aerosol system.


```python
# Define fine scale radius bins and corresponding mass bins for a lognormal distribution

# Create fine scale radius bins on a logarithmic scale from 1 nm to 10 μm
radius_bins = np.logspace(start=-9, stop=-4, num=250)  # m (1 nm to 10 μm)

# Calculate the mass for each particle size bin assuming a density of 1 g/cm^3 (1000 kg/m^3)
mass_bins = 4 / 3 * np.pi * (radius_bins) ** 3 * 1e3  # kg

# Generate a lognormal particle size distribution
# This distribution is characterized by a mode (most probable size) of 100 nm,
# a geometric standard deviation of 1.4, and a total number concentration of 10000 particles per cm^3.
concentration_lognormal_0 = par.particles.get_lognormal_pmf_distribution(
    x_values=radius_bins,
    mode=np.array(100e-9),  # Mode of the distribution (100 nm)
    geometric_standard_deviation=np.array(1.4),  # Geometric standard deviation
    number_of_particles=np.array(
        1e6 * 1e6  # Total concentration (10000 cm^-3 converted to m^-3)
    ),
)

# Plot the lognormal concentration distribution
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(radius_bins, concentration_lognormal_0)

# Set the x-axis to a logarithmic scale to capture the wide range of particle sizes
ax.set_xlabel("Particle radius (m)")

# Label the y-axis to show the concentration in particles per cubic meter
ax.set_ylabel(r"Concentration $\dfrac{1}{m^{3}}$")

# Use a logarithmic scale for the x-axis to better visualize the distribution across particle sizes
ax.set_xscale("log")

# Set Title
ax.set_title("PMF: Lognormal Particle Size Distribution")

# Display the plot
plt.show()
```


    
![png](output_19_0.png)
    


## Simulating the Coagulation Process for a Lognormal Distribution

Having established a lognormal particle size distribution, we now proceed to simulate the coagulation process over time. This simulation will show how the distribution evolves as particles coagulate and form larger particles.

### Simulation Setup

- **Time Step**: We set a time step of 100 seconds (`TIME_STEP = 100`). This interval determines how frequently the particle concentrations are updated based on the calculated coagulation rates.

- **Coagulation Kernel**: The coagulation kernel is calculated using the same parameters as before (temperature, pressure, and perfect collision efficiency). The kernel remains constant throughout the simulation as it only depends on the physical properties of the particles and the environment.

### Time Step Calculations

For each time step, we calculate the gain, loss, and net change in particle concentration:

1. **Time Step 1**:
   - Calculate the gain and loss rates for the initial concentration (`concentration_lognormal_0`).
   - Update the concentration to `concentration_lognormal_1` by applying the net rate of change. Any negative concentrations resulting from numerical errors are set to zero.

2. **Time Step 2**:
   - Recalculate the gain and loss rates based on `concentration_lognormal_1`.
   - Update the concentration to `concentration_lognormal_2` and ensure no negative values.

3. **Time Step 3**:
   - Recalculate the rates again based on `concentration_lognormal_2`.
   - Update the concentration to `concentration_lognormal_3`, correcting any negative values.

### Combining Results

The concentrations at each time step are combined into a `pandas` DataFrame. This structure allows for easy comparison of how the particle size distribution changes over time due to coagulation.

### Optional: Saving Results

The resulting concentration data can be saved to a CSV file for further analysis or visualization. This step is optional but useful for documenting the evolution of the lognormal distribution over time.

### Interpretation

By simulating the coagulation process for a lognormal distribution, we can observe how initially peaked distributions broaden and shift as particles merge. The correction for negative concentrations ensures that the physical constraints of the system (i.e., non-negative particle counts) are respected throughout the simulation.




```python
# Simulating the coagulation process over time for a lognormal distribution

# Define the time step for the simulation
TIME_STEP = 100  # seconds

# Calculate the coagulation kernel
kernel = par.dynamics.get_brownian_kernel_via_system_state(
    particle_radius=radius_bins,
    mass_particle=mass_bins,
    temperature=293.15,  # Temperature in Kelvin
    pressure=101325,  # Pressure in Pascals (1 atm)
    alpha_collision_efficiency=1.0,  # Assume perfect collision efficiency
)

# Time step 1: Calculate gain, loss, and update concentration
gain = par.dynamics.get_coagulation_gain_rate_discrete(
    radius_bins, concentration_lognormal_0, kernel
)
loss = par.dynamics.get_coagulation_loss_rate_discrete(
    concentration_lognormal_0, kernel
)
net = gain - loss
concentration_lognormal_1 = concentration_lognormal_0 + net * TIME_STEP
concentration_lognormal_1[concentration_lognormal_1 < 0] = (
    0  # Ensure no negative concentrations
)

# Time step 2: Recalculate rates and update concentration
gain = par.dynamics.get_coagulation_gain_rate_discrete(
    radius_bins, concentration_lognormal_1, kernel
)
loss = par.dynamics.get_coagulation_loss_rate_discrete(
    concentration_lognormal_1, kernel
)
net = gain - loss
concentration_lognormal_2 = concentration_lognormal_1 + net * TIME_STEP
concentration_lognormal_2[concentration_lognormal_2 < 0] = (
    0  # Ensure no negative concentrations
)

# Time step 3: Recalculate rates and update concentration
gain = par.dynamics.get_coagulation_gain_rate_discrete(
    radius_bins, concentration_lognormal_2, kernel
)
loss = par.dynamics.get_coagulation_loss_rate_discrete(
    concentration_lognormal_2, kernel
)
net = gain - loss
concentration_lognormal_3 = concentration_lognormal_2 + net * TIME_STEP
concentration_lognormal_3[concentration_lognormal_3 < 0] = (
    0  # Ensure no negative concentrations
)

# Combine the concentrations at each time step into a DataFrame for easy comparison
df_concentration = pd.DataFrame(
    {
        "0": concentration_lognormal_0,
        "1": concentration_lognormal_1,
        "2": concentration_lognormal_2,
        "3": concentration_lognormal_3,
    },
    index=radius_bins,
)

# Optional: Save the concentration data to a CSV file for further analysis
# df_concentration.to_csv("concentration_lognormal_sim.csv")
```

## Analyzing Concentration Extremes Over Time

To gain insights into how the particle concentrations evolve during the coagulation process, it's important to track the maximum and minimum concentrations at each time step. These values can provide valuable information about the stability and dynamics of the particle distribution.

### Concentration Extremes

We print the maximum and minimum concentrations for each time step:

- **Time Step 0 (`t=0`)**:
  - **Max Concentration**: This represents the highest concentration of particles at the initial distribution.
  - **Min Concentration**: This represents the lowest concentration at the initial distribution.

- **Time Step 1 (`t=1`)**:
  - **Max Concentration**: After the first time step, we observe the highest concentration to see how it compares with the initial state.
  - **Min Concentration**: Similarly, the lowest concentration is noted, which may indicate the depletion of certain particle sizes.

- **Time Step 2 (`t=2`)**:
  - **Max Concentration**: As the simulation progresses, the peak concentration may shift due to ongoing coagulation.
  - **Min Concentration**: Continued tracking of the minimum concentration helps in understanding the impact of coagulation on smaller particle sizes.

- **Time Step 3 (`t=3`)**:
  - **Max Concentration**: The final maximum concentration provides an overview of how the distribution has evolved.
  - **Min Concentration**: The minimum concentration may highlight which particle sizes have been most affected by coagulation.

### Interpretation

By examining these extremes, we can infer the following:

- **Max Concentration**: Changes in the maximum concentration over time can indicate the formation of larger particles as smaller ones coagulate. A decrease in max concentration suggests that the most abundant particle size at earlier steps is merging with others, leading to a broader or shifted distribution.

- **Min Concentration**: The minimum concentration helps identify whether certain particle sizes are becoming scarce due to coagulation, which may cause those bins to empty out or reduce significantly.

This analysis is crucial for understanding the dynamic behavior of the particle size distribution under coagulation and for ensuring that the simulation reflects realistic physical constraints.


```python
# Print the maximum and minimum concentrations at each time step

print(f"Max concentration at t=0: {concentration_lognormal_0.max()}")
print(f"Min concentration at t=0: {concentration_lognormal_0.min()}")

print(f"Max concentration at t=1: {concentration_lognormal_1.max()}")
print(f"Min concentration at t=1: {concentration_lognormal_1.min()}")

print(f"Max concentration at t=2: {concentration_lognormal_2.max()}")
print(f"Min concentration at t=2: {concentration_lognormal_2.min()}")

print(f"Max concentration at t=3: {concentration_lognormal_3.max()}")
print(f"Min concentration at t=3: {concentration_lognormal_3.min()}")
```

    Max concentration at t=0: 54738275267.16476
    Min concentration at t=0: 1.6445825598006672e-81
    Max concentration at t=1: 51186782324.56813
    Min concentration at t=1: 0.0
    Max concentration at t=2: 48026245801.09982
    Min concentration at t=2: 0.0
    Max concentration at t=3: 45351022236.99084
    Min concentration at t=3: 0.0


## Plotting the Coagulation Gain, Loss, and Net Rates for Lognormal Distribution

To visualize the dynamics of coagulation for a lognormal particle size distribution, we plot the rates of gain, loss, and net change in concentration across the range of particle sizes. This plot provides insights into how particles are interacting during the coagulation process.

### Plot Details

- **Net Rate**: The net rate of change in particle concentration is plotted as a thick gray line. This line represents the overall effect of coagulation, indicating whether the concentration in each size bin is increasing or decreasing.

- **Gain Rate**: The gain rate, plotted in dark green, shows how particles are being added to each size bin as smaller particles coagulate to form larger ones. This rate reflects the accumulation of particles in specific size bins.

- **Loss Rate**: The loss rate is plotted in red, with the values multiplied by -1 to indicate that it represents a reduction in particle concentration. This line shows how particles are being depleted from each size bin due to coagulation.

### Axes Configuration

- **X-axis**: The x-axis represents the particle radius in meters, plotted on a logarithmic scale to capture the wide range of particle sizes present in the distribution.
  
- **Y-axis**: The y-axis shows the rate of change in concentration in units of particles per cubic meter per second ($\dfrac{1}{m^{3} s^{1}}$), providing a clear view of how rapidly particles are being gained, lost, or changing net concentration in the system.

### Legend and Interpretation

A legend is included to clearly differentiate between the net, gain, and loss lines on the plot. By analyzing these rates, we can determine the most active particle sizes in the coagulation process:

- **Positive Net Rate**: Indicates that the particle size bin is gaining particles, likely due to the aggregation of smaller particles.
- **Negative Net Rate**: Indicates that the particle size bin is losing particles, either because they are merging into larger particles or being depleted through other processes.

This plot is essential for understanding the detailed behavior of the particle size distribution during coagulation, highlighting which sizes are most affected by the process.



```python
# Plot the coagulation gain, loss, and net rates for the lognormal distribution

# Create a figure and axis object for the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the net coagulation rate
# The net rate is plotted with a thicker gray line for emphasis
ax.plot(radius_bins, net, label="Net", color="gray", linewidth=4)

# Plot the gain rate
# The gain rate is plotted in dark green
ax.plot(radius_bins, gain, label="Gain", color="darkgreen")

# Plot the loss rate
# The loss rate is plotted in red, and multiplied by -1 for plotting to indicate that it's a removal process
ax.plot(radius_bins, -1 * loss, label="Loss", color="red")

# Set the x-axis label to indicate the particle radius in meters
ax.set_xlabel("Particle radius (m)")

# Set the y-axis label to indicate the rate of change in concentration, with appropriate units
ax.set_ylabel(r"Rate $\dfrac{1}{m^{3} s^{1}}$")

# Use a logarithmic scale for the x-axis to account for the wide range of particle sizes
ax.set_xscale("log")

# set title
ax.set_title("PMF: Coagulation gain, loss, and net rates")

# Add a legend to identify the gain, loss, and net lines
plt.legend()

# Display the plot
plt.show()
```


    
![png](output_25_0.png)
    


## Plotting the Evolution of Concentration for Lognormal Distribution

After simulating the coagulation process over several time steps, it is important to visualize how the particle concentration evolves. This plot shows the concentration distribution at different time steps, allowing us to observe the changes in the lognormal distribution as coagulation progresses.

### Plot Details

- **Time Step 0 (`t=0`)**: The initial concentration distribution is plotted as a solid blue line. This represents the starting point of the simulation, with a lognormal distribution centered around the mode.
  
- **Time Step 1 (`t=1`)**: After the first time step, the concentration distribution is plotted as a dashed green line. This line shows the immediate impact of coagulation on the particle sizes.

- **Time Step 2 (`t=2`)**: The concentration distribution at the second time step is plotted as a dash-dot orange line. By this time, noticeable shifts in the distribution may start to appear as particles coagulate.

- **Time Step 3 (`t=3`)**: Finally, the concentration distribution after the third time step is plotted as a dotted red line. This line illustrates further evolution of the distribution, highlighting the ongoing effects of coagulation.

### Axes Configuration

- **X-axis**: The x-axis represents the particle radius in meters, plotted on a logarithmic scale to cover the wide range of particle sizes in the lognormal distribution.
  
- **Y-axis**: The y-axis shows the concentration in particles per cubic meter (m⁻³), also plotted on a logarithmic scale to reflect the broad range of concentrations.

### Interpretation

By comparing the concentration distributions at different time steps, you can observe how the lognormal distribution shifts and broadens as particles coagulate. Typically, the concentration of smaller particles decreases over time as they merge to form larger particles, leading to an increase in the concentration of larger particles. This visualization provides a clear, temporal view of the coagulation process and its effects on the particle size distribution.

This plot is crucial for understanding the dynamic evolution of aerosol particles under coagulation, particularly when starting with a realistic lognormal distribution.



```python
# Plot the evolution of particle concentration over time for the lognormal distribution

# Create a figure and axis object for the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the concentration distribution at each time step
ax.plot(
    radius_bins,
    concentration_lognormal_0,
    label="t=0",
    linestyle="-",
    color="blue",
)
ax.plot(
    radius_bins,
    concentration_lognormal_1,
    label="t=1",
    linestyle="--",
    color="green",
)
ax.plot(
    radius_bins,
    concentration_lognormal_2,
    label="t=2",
    linestyle="-.",
    color="orange",
)
ax.plot(
    radius_bins,
    concentration_lognormal_3,
    label="t=3",
    linestyle=":",
    color="red",
)

# Set the x-axis label to indicate the particle radius in meters
ax.set_xlabel("Particle radius (m)")

# Set the y-axis label to indicate the concentration in particles per cubic meter
ax.set_ylabel(r"Concentration $\dfrac{1}{m^{3}}$")

# set title
ax.set_title("PMF: Particle concentration evolution over time")

# Use a logarithmic scale for both the x-axis and y-axis
# This is essential for visualizing the wide range of particle sizes and concentration changes
ax.set_xscale("log")
# ax.set_yscale("log")

# Add a legend to differentiate between the time steps
plt.legend()

# Display the plot
plt.show()
```


    
![png](output_27_0.png)
    


# Total Concentration Over Time

To understand the overall impact of coagulation on the particle population, it is essential to track the total concentration of particles over time. This plot shows how the total concentration changes as particles coagulate and form larger sizes.


```python
# Summation of PMF distribution
# Calculate the total concentration of particles at each time step
total_concentration_0 = concentration_lognormal_0.sum()
total_concentration_1 = concentration_lognormal_1.sum()
total_concentration_2 = concentration_lognormal_2.sum()
total_concentration_3 = concentration_lognormal_3.sum()

# Create a figure and axis object for the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the total concentration at each time step
ax.plot(
    [
        total_concentration_0,
        total_concentration_1,
        total_concentration_2,
        total_concentration_3,
    ],
    label="Total concentration",
    marker="o",  # Add markers to indicate each time step
    linestyle="-",  # Use a solid line to connect the markers
    color="blue",  # Set the line color to blue
)

# Set the x-axis label to indicate the time step
ax.set_xlabel("Time step")

# Set the y-axis label to indicate the total concentration in particles per cubic meter
ax.set_ylabel("Total concentration $(m^{-3})$")
ax.set_ylim(bottom=0.84e12)

ax.set_title("PMF: Total concentration at each time step")

# Display the plot
plt.show()
```


    
![png](output_29_0.png)
    


### Summary

In this notebook, we explored the process of particle coagulation in aerosols, focusing on both uniform and lognormal particle size distributions. The notebook provided a step-by-step implementation of the coagulation process, highlighting key concepts and calculations necessary to simulate the dynamic behavior of aerosol particles over time.

#### Key Steps and Findings:

1. **Coagulation Basics**:
   - We began with a uniform particle size distribution to introduce the fundamental concepts of coagulation, including the calculation of the Brownian coagulation kernel and the rates of particle gain, loss, and net change.
   - The initial uniform distribution allowed us to understand the basic mechanics of how particles interact and merge over time.

2. **Transition to Lognormal Distribution**:
   - We then shifted to a more realistic lognormal particle size distribution, which better represents real-world aerosol systems. This distribution was characterized by a mode (most probable particle size), a geometric standard deviation, and a total number concentration.
   - The coagulation process was simulated for this lognormal distribution, with the particle concentrations updated over several discrete time steps.

3. **Simulation and Visualization**:
   - The evolution of the particle size distribution was tracked over time, with plots showing the changes in concentration for different particle sizes. These plots illustrated how smaller particles gradually merge to form larger ones, leading to a shift and broadening of the distribution.
   - The rates of particle gain, loss, and net change were also visualized, providing insights into the most active particle sizes during the coagulation process.

4. **Key Insights**:
   - The notebook demonstrated that coagulation leads to a decrease in the number concentration of smaller particles as they coalesce to form larger particles. This results in a broader size distribution with fewer small particles and an increased concentration of larger particles.
   - The lognormal distribution, due to its realistic representation of aerosol particles, showed more complex dynamics compared to the uniform distribution, emphasizing the importance of starting with an appropriate initial distribution in simulations.

#### Conclusion:

This notebook provided a guide to simulating the coagulation process in aerosol particles, from basic principles to more advanced applications involving realistic size distributions. By comparing the results from uniform and lognormal distributions, we gained a deeper understanding of how particle populations evolve under coagulation, highlighting the critical role of particle size distribution in aerosol dynamics. The methods and visualizations presented here can be extended to further study and analyze aerosol behavior in various environmental and industrial contexts.


---
# Coagulation_Basic_2_PDF.md

# Coagulation Basics 2: PDF Representation

### Continuous Particle Size Distributions

In this notebook, we implement the coagulation process for continuous particle size distributions, following the methodology outlined in Seinfeld and Pandis (2006). This work builds on concepts from `Coagulation Basics 1: PMF Representation` by extending the analysis to probability density function (PDF) representations of particle size distributions.

#### Objective

The primary objective is to demonstrate the calculation of the coagulation kernel and the resulting coagulation rates for a given particle size distribution. We will compare the PDF representation with the probability mass function (PMF) representation, highlighting how the PDF's continuous nature influences the coagulation process.

#### Approach

1. **Uniform Distribution**:
   - We begin with a uniform particle size distribution to illustrate the fundamental mechanics of coagulation.
  
2. **Lognormal Probability Density Function**:
   - After establishing the basics with a uniform distribution, we proceed to a lognormal size distribution, which more accurately reflects real-world aerosol size distributions. The PDF representation is continuous and describes the probability of finding particles within a specific size range. Integrating the PDF across the entire size range yields the total particle concentration.
  
   - **Units**:
     - The units of the PDF are particles per unit volume per unit size, typically expressed as $\dfrac{1}{m^3 \cdot m}$ or $\dfrac{1}{m^4}$. Integration of the PDF over the particle size range gives the total number concentration, expressed in $\dfrac{1}{m^3}$.

This step-by-step approach lays the groundwork for more advanced implementations, such as those in the `particula` library.

---

#### Required Libraries

We will use common scientific libraries, including `numpy` for numerical calculations, `matplotlib` for visualization, and `pandas` for data manipulation. Additionally, we will leverage specific functions from the `particula` library to compute the coagulation kernel and generate size distributions.


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapezoid

# particula imports
import particula as par
```

## Creating a Size Distribution

In this section, we define the size distribution for aerosol particles. The particle sizes are distributed across several fine-scale bins, allowing for detailed modeling of particle behavior over a wide size range.

### Particle Size Bins

- **`radius_bins`**: These bins represent particle radii, spanning from 1 nanometer (1e-9 m) to 10 micrometers (1e-5 m) on a logarithmic scale. Using fine-scale bins (500 in total) provides a more detailed resolution of the size distribution, which is crucial for accurate coagulation modeling.
- **`mass_bins`**: For each radius bin, we calculate the corresponding particle mass, assuming a particle density of 1 g/cm³ (1000 kg/m³ in SI units). This mass calculation is essential for understanding how particles interact and coagulate over time.


```python
# Create fine-scale radius bins on a logarithmic scale from 1 nm to 10 μm
radius_bins = np.logspace(start=-9, stop=-4, num=500)  # m (1 nm to 10 μm)

# Calculate the mass for each particle size bin assuming a density of 1 g/cm³ (1000 kg/m³)
mass_bins = 4 / 3 * np.pi * radius_bins**3 * 1e3  # kg
```

## Calculating the Brownian Coagulation Kernel

In this section, we calculate the Brownian coagulation kernel matrix, which quantifies the rate at which particles of different sizes collide and coagulate due to Brownian motion. Understanding the kernel matrix is crucial for analyzing the dynamics of particle coagulation.

It's important to note that the kernel matrix itself remains consistent whether we're using a Probability Mass Function (PMF) or a Probability Density Function (PDF) representation. The difference lies in how we apply the kernel matrix—using summations in the PMF case and integrations in the PDF case—to calculate coagulation rates.


```python
# The Brownian coagulation kernel is calculated using the `brownian_coagulation_kernel_via_system_state` function.
# This function takes into account particle size, mass, temperature, pressure, and collision efficiency
# to compute the coagulation rates between particles of different sizes.
kernel = par.dynamics.get_brownian_kernel_via_system_state(
    particle_radius=radius_bins,
    mass_particle=mass_bins,
    temperature=293.15,  # Temperature in Kelvin (20°C)
    pressure=101325,  # Pressure in Pascals (1 atm)
    alpha_collision_efficiency=1.0,  # Assume perfect collision efficiency
)

# Display the shape of the kernel matrix to confirm its dimensions
print(f"Kernel shape: {kernel.shape}")

# Create a pandas DataFrame from the kernel matrix
# The DataFrame allows for easier analysis and visualization of the coagulation kernel.
# Rows and columns are indexed by the particle radius bins, making it clear which sizes are interacting.
df_kernel = pd.DataFrame(kernel, index=radius_bins, columns=radius_bins)

# Print the first 5 rows of the DataFrame to inspect the computed kernel values
df_kernel.head(5)

# Optional: Save the kernel matrix to a CSV file for further analysis or sharing
# df_kernel.to_csv("kernel.csv")
```

    Kernel shape: (500, 500)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1.000000e-09</th>
      <th>1.023340e-09</th>
      <th>1.047225e-09</th>
      <th>1.071668e-09</th>
      <th>1.096681e-09</th>
      <th>1.122277e-09</th>
      <th>1.148472e-09</th>
      <th>1.175277e-09</th>
      <th>1.202708e-09</th>
      <th>1.230780e-09</th>
      <th>...</th>
      <th>8.124930e-05</th>
      <th>8.314568e-05</th>
      <th>8.508632e-05</th>
      <th>8.707225e-05</th>
      <th>8.910453e-05</th>
      <th>9.118425e-05</th>
      <th>9.331251e-05</th>
      <th>9.549045e-05</th>
      <th>9.771921e-05</th>
      <th>1.000000e-04</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.000000e-09</th>
      <td>8.812734e-16</td>
      <td>8.867456e-16</td>
      <td>8.930227e-16</td>
      <td>9.001194e-16</td>
      <td>9.080509e-16</td>
      <td>9.168335e-16</td>
      <td>9.264845e-16</td>
      <td>9.370220e-16</td>
      <td>9.484651e-16</td>
      <td>9.608340e-16</td>
      <td>...</td>
      <td>1.310366e-09</td>
      <td>1.340961e-09</td>
      <td>1.372270e-09</td>
      <td>1.404310e-09</td>
      <td>1.437098e-09</td>
      <td>1.470650e-09</td>
      <td>1.504986e-09</td>
      <td>1.540123e-09</td>
      <td>1.576080e-09</td>
      <td>1.612876e-09</td>
    </tr>
    <tr>
      <th>1.023340e-09</th>
      <td>8.867456e-16</td>
      <td>8.914801e-16</td>
      <td>8.970151e-16</td>
      <td>9.033642e-16</td>
      <td>9.105423e-16</td>
      <td>9.185649e-16</td>
      <td>9.274485e-16</td>
      <td>9.372104e-16</td>
      <td>9.478690e-16</td>
      <td>9.594436e-16</td>
      <td>...</td>
      <td>1.251439e-09</td>
      <td>1.280658e-09</td>
      <td>1.310559e-09</td>
      <td>1.341157e-09</td>
      <td>1.372470e-09</td>
      <td>1.404514e-09</td>
      <td>1.437305e-09</td>
      <td>1.470862e-09</td>
      <td>1.505201e-09</td>
      <td>1.540342e-09</td>
    </tr>
    <tr>
      <th>1.047225e-09</th>
      <td>8.930227e-16</td>
      <td>8.970151e-16</td>
      <td>9.018038e-16</td>
      <td>9.074021e-16</td>
      <td>9.138240e-16</td>
      <td>9.210845e-16</td>
      <td>9.291991e-16</td>
      <td>9.381847e-16</td>
      <td>9.480587e-16</td>
      <td>9.588397e-16</td>
      <td>...</td>
      <td>1.195165e-09</td>
      <td>1.223070e-09</td>
      <td>1.251626e-09</td>
      <td>1.280848e-09</td>
      <td>1.310753e-09</td>
      <td>1.341355e-09</td>
      <td>1.372672e-09</td>
      <td>1.404719e-09</td>
      <td>1.437514e-09</td>
      <td>1.471075e-09</td>
    </tr>
    <tr>
      <th>1.071668e-09</th>
      <td>9.001194e-16</td>
      <td>9.033642e-16</td>
      <td>9.074021e-16</td>
      <td>9.122455e-16</td>
      <td>9.179079e-16</td>
      <td>9.244034e-16</td>
      <td>9.317470e-16</td>
      <td>9.399548e-16</td>
      <td>9.490434e-16</td>
      <td>9.590307e-16</td>
      <td>...</td>
      <td>1.141425e-09</td>
      <td>1.168075e-09</td>
      <td>1.195346e-09</td>
      <td>1.223255e-09</td>
      <td>1.251814e-09</td>
      <td>1.281041e-09</td>
      <td>1.310949e-09</td>
      <td>1.341555e-09</td>
      <td>1.372875e-09</td>
      <td>1.404926e-09</td>
    </tr>
    <tr>
      <th>1.096681e-09</th>
      <td>9.080509e-16</td>
      <td>9.105423e-16</td>
      <td>9.138240e-16</td>
      <td>9.179079e-16</td>
      <td>9.228066e-16</td>
      <td>9.285337e-16</td>
      <td>9.351036e-16</td>
      <td>9.425313e-16</td>
      <td>9.508330e-16</td>
      <td>9.600258e-16</td>
      <td>...</td>
      <td>1.090104e-09</td>
      <td>1.115556e-09</td>
      <td>1.141601e-09</td>
      <td>1.168254e-09</td>
      <td>1.195530e-09</td>
      <td>1.223442e-09</td>
      <td>1.252005e-09</td>
      <td>1.281235e-09</td>
      <td>1.311147e-09</td>
      <td>1.341756e-09</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 500 columns</p>
</div>



## Generating Lognormal PDF Size Distribution

In this section, we generate a lognormal particle size distribution and visualize it to understand the particle concentration across different sizes. A lognormal distribution is commonly used to represent aerosol particle size distributions due to its ability to model the skewness observed in real-world data.

### Key Parameters

- **Mode (Most Probable Size)**: The mode of the distribution is set at 100 nanometers (100 nm), representing the most common particle size within the distribution.
- **Geometric Standard Deviation**: A geometric standard deviation of 1.4 is used, which determines the spread of the distribution. This value reflects how broadly the particle sizes are distributed around the mode.
- **Total Number Concentration**: The total particle concentration is set at $1 \times 10^6$ particles per cubic centimeter (cm³), which is converted to $1 \times 10^{12}$ particles per cubic meter (m³) for the calculation.

### Visualization

The resulting distribution is plotted with particle radius on the x-axis and particle concentration on the y-axis. The x-axis is scaled logarithmically to effectively display the wide range of particle sizes. This plot helps in visualizing how particle concentration varies with size, providing insights into the distribution characteristics and potential behavior during coagulation processes.



```python
# Generate a lognormal particle size distribution
# This distribution is characterized by a mode (most probable size) of 100 nm,
# a geometric standard deviation of 1.4, and a total number concentration of 1e6 particles per cm^3.
concentration_lognormal_0 = par.particles.get_lognormal_pdf_distribution(
    x_values=radius_bins,
    mode=np.array(100e-9),  # Mode of the distribution (100 nm)
    geometric_standard_deviation=np.array(1.4),  # Geometric standard deviation
    number_of_particles=np.array(
        1e6 * 1e6
    ),  # Total concentration (1e6 cm^-3 converted to m^-3)
)

# Plot the lognormal concentration distribution
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(radius_bins, concentration_lognormal_0)

# Set the x-axis to a logarithmic scale to capture the wide range of particle sizes
ax.set_xlabel("Particle radius (m)")

# Label the y-axis to show the concentration in particles per cubic meter per unit size
ax.set_ylabel(r"Concentration $\dfrac{1}{m^3 \cdot m}$")

# Use a logarithmic scale for the x-axis to better visualize the distribution across particle sizes
ax.set_xscale("log")

# Set the title of the plot
ax.set_title("Lognormal Particle Size Distribution")

# Display the plot
plt.show()
```


    
![png](output_7_0.png)
    


## Simulating the Coagulation Process for a Lognormal Distribution

After establishing a lognormal particle size distribution, we now simulate the coagulation process over time to observe how the distribution evolves as particles collide and form larger particles.

### Simulation Setup

- **Time Step**: We use a time step of 100 seconds (`TIME_STEP = 100`). This interval determines the frequency at which particle concentrations are updated based on the calculated coagulation rates.

- **Coagulation Kernel**: The coagulation kernel is computed using the same environmental parameters (temperature, pressure, and collision efficiency) as before. The kernel remains unchanged throughout the simulation, as it depends solely on the physical properties of the particles and the surrounding environment.

### Time Step Calculations

For each time step, we perform the following calculations:

1. **Time Step 1**:
   - **Gain and Loss Rates**: Calculate the gain and loss rates for the initial concentration (`concentration_lognormal_0`) based on the coagulation kernel.
   - **Update Concentration**: Apply the net rate of change to update the particle concentration to `concentration_lognormal_1`. Any negative concentrations, which may result from numerical errors, are set to zero.

2. **Time Step 2**:
   - **Recalculate Rates**: Compute the gain and loss rates using the updated concentration (`concentration_lognormal_1`).
   - **Update Concentration**: Update the concentration to `concentration_lognormal_2`, ensuring no negative values.

3. **Time Step 3**:
   - **Recalculate Rates**: Calculate the rates again based on the most recent concentration (`concentration_lognormal_2`).
   - **Update Concentration**: Update the concentration to `concentration_lognormal_3`, correcting any negative concentrations.

### Combining Results

The concentrations at each time step are combined into a `pandas` DataFrame. This structure facilitates easy comparison of the particle size distribution changes over time, offering insights into the coagulation process.

### Optional: Saving Results

For further analysis or documentation, the concentration data can be saved to a CSV file. This step is optional but can be useful for detailed post-simulation analysis.



```python
# Simulating the coagulation process over time for a lognormal distribution

# Define the time step for the simulation
TIME_STEP = 100  # seconds

# Calculate the coagulation kernel
kernel = par.dynamics.get_brownian_kernel_via_system_state(
    particle_radius=radius_bins,
    mass_particle=mass_bins,
    temperature=293.15,  # Temperature in Kelvin
    pressure=101325,  # Pressure in Pascals (1 atm)
    alpha_collision_efficiency=1.0,  # Assume perfect collision efficiency
)

# Time step 1: Calculate gain, loss, and update concentration
gain = par.dynamics.get_coagulation_gain_rate_continuous(
    radius=radius_bins, concentration=concentration_lognormal_0, kernel=kernel
)
loss = par.dynamics.get_coagulation_loss_rate_continuous(
    radius=radius_bins, concentration=concentration_lognormal_0, kernel=kernel
)
net = gain - loss
concentration_lognormal_1 = concentration_lognormal_0 + net * TIME_STEP
concentration_lognormal_1[concentration_lognormal_1 < 0] = (
    0  # Ensure no negative concentrations
)

# Time step 2: Recalculate rates and update concentration
gain = par.dynamics.get_coagulation_gain_rate_continuous(
    radius=radius_bins, concentration=concentration_lognormal_1, kernel=kernel
)
loss = par.dynamics.get_coagulation_loss_rate_continuous(
    radius=radius_bins, concentration=concentration_lognormal_1, kernel=kernel
)
net = gain - loss
concentration_lognormal_2 = concentration_lognormal_1 + net * TIME_STEP
concentration_lognormal_2[concentration_lognormal_2 < 0] = (
    0  # Ensure no negative concentrations
)

# Time step 3: Recalculate rates and update concentration
gain = par.dynamics.get_coagulation_gain_rate_continuous(
    radius=radius_bins, concentration=concentration_lognormal_2, kernel=kernel
)
loss = par.dynamics.get_coagulation_loss_rate_continuous(
    radius=radius_bins, concentration=concentration_lognormal_2, kernel=kernel
)
net = gain - loss
concentration_lognormal_3 = concentration_lognormal_2 + net * TIME_STEP
concentration_lognormal_3[concentration_lognormal_3 < 0] = (
    0  # Ensure no negative concentrations
)

# Combine the concentrations at each time step into a DataFrame for easy comparison
df_concentration = pd.DataFrame(
    {
        "0": concentration_lognormal_0,
        "1": concentration_lognormal_1,
        "2": concentration_lognormal_2,
        "3": concentration_lognormal_3,
    },
    index=radius_bins,
)

# Optional: Save the concentration data to a CSV file for further analysis
# df_concentration.to_csv("concentration_lognormal_sim.csv")
```

## Plotting the Coagulation Gain, Loss, and Net Rates for Lognormal Distribution

To visualize the dynamics of coagulation for a lognormal particle size distribution, we plot the rates of gain, loss, and net change in concentration across the range of particle sizes. This plot provides insights into how particles are interacting during the coagulation process.

### Plot Details

- **Net Rate**: The net rate of change in particle concentration is plotted as a thick gray line. This line represents the overall effect of coagulation, indicating whether the concentration in each size bin is increasing or decreasing.

- **Gain Rate**: The gain rate, plotted in dark green, shows how particles are being added to each size bin as smaller particles coagulate to form larger ones. This rate reflects the accumulation of particles in specific size bins.

- **Loss Rate**: The loss rate is plotted in red, with the values multiplied by -1 to indicate that it represents a reduction in particle concentration. This line shows how particles are being depleted from each size bin due to coagulation.

### Axes Configuration

- **X-axis**: The x-axis represents the particle radius in meters, plotted on a logarithmic scale to capture the wide range of particle sizes present in the distribution.
  
- **Y-axis**: The y-axis shows the rate of change in concentration in units of particles per cubic meter per second ($\dfrac{1}{m^{3} s^{1} \cdot m}$), providing a clear view of how rapidly particles are being gained, lost, or changing net concentration in the system.

### Legend and Interpretation

A legend is included to clearly differentiate between the net, gain, and loss lines on the plot. By analyzing these rates, we can determine the most active particle sizes in the coagulation process:

- **Positive Net Rate**: Indicates that the particle size bin is gaining particles, likely due to the aggregation of smaller particles.
- **Negative Net Rate**: Indicates that the particle size bin is losing particles, either because they are merging into larger particles or being depleted through other processes.

This plot is essential for understanding the detailed behavior of the particle size distribution during coagulation, highlighting which sizes are most affected by the process.



```python
# Plot the coagulation gain, loss, and net rates for the lognormal distribution

# Create a figure and axis object for the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the net coagulation rate
# The net rate is plotted with a thicker gray line for emphasis
ax.plot(radius_bins, net, label="Net", color="gray", linewidth=4)

# Plot the gain rate
# The gain rate is plotted in dark green
ax.plot(radius_bins, gain, label="Gain", color="darkgreen")

# Plot the loss rate
# The loss rate is plotted in red, and multiplied by -1 for plotting to indicate that it's a removal process
ax.plot(radius_bins, -1 * loss, label="Loss", color="red")

# Set the x-axis label to indicate the particle radius in meters
ax.set_xlabel("Particle radius (m)")

# Set the y-axis label to indicate the rate of change in concentration, with appropriate units
ax.set_ylabel(r"Rate $\dfrac{1}{m^{3} s^{1} \cdot m}$")

# Use a logarithmic scale for the x-axis to account for the wide range of particle sizes
ax.set_xscale("log")

# title
ax.set_title("PDF: Coagulation Gain, Loss, and Net Rates")

# Add a legend to identify the gain, loss, and net lines
plt.legend()

# Display the plot
plt.show()
```


    
![png](output_11_0.png)
    


## Plotting the Evolution of Concentration for Lognormal Distribution

After simulating the coagulation process over several time steps, it is important to visualize how the particle concentration evolves. This plot shows the concentration distribution at different time steps, allowing us to observe the changes in the lognormal distribution as coagulation progresses.



```python
# Plot the evolution of particle concentration over time for the lognormal distribution

# Create a figure and axis object for the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the concentration distribution at each time step
ax.plot(
    radius_bins,
    concentration_lognormal_0,
    label="t=0",
    linestyle="-",
    color="blue",
)
ax.plot(
    radius_bins,
    concentration_lognormal_1,
    label="t=1",
    linestyle="--",
    color="green",
)
ax.plot(
    radius_bins,
    concentration_lognormal_2,
    label="t=2",
    linestyle="-.",
    color="orange",
)
ax.plot(
    radius_bins,
    concentration_lognormal_3,
    label="t=3",
    linestyle=":",
    color="red",
)

# Set the x-axis label to indicate the particle radius in meters
ax.set_xlabel("Particle radius (m)")

# Set the y-axis label to indicate the concentration in particles per cubic meter
ax.set_ylabel(r"Concentration $\dfrac{1}{m^3 \cdot m}$")

# Set the title of the plot to describe the concentration evolution over time
ax.set_title("PDF: Particle concentration evolution over time")

# Use a logarithmic scale for both the x-axis and y-axis
# This is essential for visualizing the wide range of particle sizes and concentration changes
ax.set_xscale("log")
# ax.set_yscale("log")

# Add a legend to differentiate between the time steps
plt.legend()

# Display the plot
plt.show()
```


    
![png](output_13_0.png)
    


## Calculating and Visualizing Total Particle Concentration Over Time

In this section, we calculate the total number of particles at each time step by integrating the particle concentration across the size distribution. This provides a clear view of how the total particle concentration evolves as coagulation progresses.

### Integration of Particle Concentration

To determine the total number of particles at each time step, we integrate the concentration across all particle sizes using the trapezoidal rule. This integration gives us the total particle concentration (in particles per cubic meter) at each specific time step.

### Plotting Total Concentration Over Time

We then plot the total particle concentration for each time step to visualize how the overall concentration decreases as particles coagulate into larger sizes. The x-axis represents the time steps, while the y-axis shows the total particle concentration in particles per cubic meter. Markers are used to highlight the concentration at each time step, connected by a solid line to indicate the trend over time.

This plot provides a straightforward representation of the coagulation process's impact on the total number of particles, illustrating the decrease in concentration as particles merge and grow.

### Note:

You can compare the results obtained from the PDF representation with those from the PMF representation in the previous notebook to observe how similar the results are despite the different representations. They are not exactly equal.


```python
# Integrate the concentration to calculate the total number of particles at each time step
total_concentration_lognormal_0 = trapezoid(
    concentration_lognormal_0, x=radius_bins
)
total_concentration_lognormal_1 = trapezoid(
    concentration_lognormal_1, x=radius_bins
)
total_concentration_lognormal_2 = trapezoid(
    concentration_lognormal_2, x=radius_bins
)
total_concentration_lognormal_3 = trapezoid(
    concentration_lognormal_3, x=radius_bins
)

# Plot the total concentration at each time step for the lognormal distribution
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the total concentration at each time step
ax.plot(
    [
        total_concentration_lognormal_0,
        total_concentration_lognormal_1,
        total_concentration_lognormal_2,
        total_concentration_lognormal_3,
    ],
    label="Total concentration",
    marker="o",  # Add markers to indicate each time step
    linestyle="-",  # Use a solid line to connect the markers
    color="blue",  # Set the line color to blue
)

# Set the x-axis label to indicate the time step
ax.set_xlabel("Time step")

# Set the y-axis label to indicate the total concentration in particles per cubic meter
ax.set_ylabel(r"Total concentration $\dfrac{1}{m^3}$")
ax.set_ylim(bottom=0.84e12)

# Set the title of the plot to describe the total concentration evolution over time
ax.set_title("Total concentration at each time step")

# Display the plot
plt.show()
```


    
![png](output_15_0.png)
    


### Summary

In this notebook, we explored the process of particle coagulation in aerosols, with a particular focus on probability density function (PDF) representations of lognormal particle size distributions. 

We began by defining a lognormal distribution, which is commonly used to represent aerosol particle sizes due to its ability to model the skewed distributions observed in real-world scenarios. We then calculated the Brownian coagulation kernel, a crucial element that quantifies the rate at which particles of different sizes collide and coagulate due to Brownian motion. 

Through a series of simulations, we observed how the particle size distribution evolves over time as coagulation occurs. By applying the coagulation kernel to the initial distribution, we calculated the gain and loss rates for particles and updated the particle concentrations at each time step. This iterative process illustrated the gradual decrease in total particle concentration as smaller particles combined to form larger ones.

Finally, we integrated the particle concentrations across the size distribution at each time step to determine the total number of particles present. This allowed us to visualize the overall reduction in particle concentration over time, providing a clear demonstration of the impact of coagulation on aerosol dynamics.

This notebook not only highlights the fundamental principles of aerosol coagulation but also provides a practical framework for modeling and analyzing these processes using PDF representations. The methodologies and tools used here, such as the integration of concentration data and the use of a coagulation kernel, are essential for understanding the behavior of aerosols in various environmental and industrial contexts.


---
# Coagulation_Basic_3_compared.md

# Coagulation Basic 3: PMF vs. PDF

In this section, we will compare two fundamental approaches to modeling aerosol particle coagulation: the Probability Mass Function (PMF) and the Probability Density Function (PDF) methods. Both methods offer different perspectives on representing particle size distributions and calculating coagulation rates, which are critical for understanding how particles in aerosols interact and evolve over time.

## Probability Mass Function (PMF) vs. Probability Density Function (PDF)

- **PMF Representation**: The PMF method discretizes the particle size distribution into distinct bins, each representing a specific particle size or mass. This approach counts the number of particles within each bin, making it a straightforward method for tracking how particle populations change due to coagulation. PMF is particularly useful when dealing with discrete particle sizes or when particle number concentrations are of primary interest.

- **PDF Representation**: In contrast, the PDF method provides a continuous representation of the particle size distribution. Instead of counting particles in discrete bins, PDF describes the likelihood of finding particles within a given size range. This approach is well-suited for scenarios where a smooth distribution of particle sizes is expected or when dealing with very fine size resolutions.

## Comparison Objectives

The objective of this comparison is to demonstrate how the choice between PMF and PDF representations affects the calculation of coagulation rates and the resulting particle size distributions. By applying both methods to a lognormal size distribution, we will analyze the differences in how each method handles the evolution of particle populations during coagulation.

To facilitate this comparison, we will:
1. **Initialize Lognormal Distributions**: Generate lognormal particle size distributions using both PMF and PDF methods, ensuring that both distributions share the same initial parameters (e.g., mode, geometric standard deviation, and total particle concentration).
  
2. **Calculate Coagulation Kernel**: Compute the Brownian coagulation kernel using identical environmental conditions (e.g., temperature, pressure, collision efficiency) for both methods. This will allow us to isolate the effect of the distribution representation on the coagulation rates.

3. **Simulate Coagulation**: Simulate the coagulation process over several time steps for both PMF and PDF representations, tracking how the particle size distributions evolve and comparing the results.

By the end of this section, we aim to highlight the strengths and limitations of each method, providing insights into when and why one approach might be preferred over the other in aerosol research and modeling.



```python
# Import necessary libraries

import numpy as np  # For numerical operations and array manipulations
import matplotlib.pyplot as plt  # For plotting graphs and visualizations
import pandas as pd  # For data manipulation and analysis
from scipy.integrate import (
    trapezoid,
)  # For numerical integration using the trapezoidal rule

import particula as par
```

## Setting Up and Visualizing

In this section, we define the parameters for a lognormal particle size distribution and generate both Probability Density Function (PDF) and Probability Mass Function (PMF) representations. We then visualize these distributions to compare how each method represents particle concentrations across different size ranges.

**Distribution Parameters**

We start by defining the key parameters for the lognormal distribution:

- **Mode**: The most probable particle size is set to 200 nanometers (200 nm or \(200 \times 10^{-9}\) meters).
- **Geometric Standard Deviation**: The spread of the distribution is controlled by a geometric standard deviation of 1.5.
- **Total Number Concentration**: The total concentration of particles is \(1 \times 10^6\) particles per cubic centimeter (cm³), which we convert to \(1 \times 10^{12}\) particles per cubic meter (m³) for our calculations.

We also define the radius bins, which span a wide range from 1 nanometer to 10 micrometers, using a logarithmic scale to capture the distribution across different particle sizes.

**Explanation**

- **Parameter Definitions**: The mode, geometric standard deviation, and number concentration are clearly defined to set up the lognormal distribution.
- **Distribution Creation**: We generate both PDF and PMF representations of the distribution using the defined parameters and radius bins. This allows us to see how each method captures the particle concentrations.
- **Visualization**: The plot compares the PDF and PMF distributions on a logarithmic scale, which is essential for accurately displaying the wide range of particle sizes and concentrations. The use of log scales on both axes helps to highlight the differences between the two methods.

By visualizing these distributions side by side, we gain a better understanding of how the PDF and PMF methods differ in representing aerosol particle sizes and concentrations.


```python
# Define distribution parameters
mode = np.array([200e-9])  # Mode of the distribution (200 nm)
std = np.array([1.5])  # Geometric standard deviation
number = np.array([1e6]) * 1e6  # 1e6 particles per cm^3 converted to m^3

# Define radius bins on a logarithmic scale from 1 nm to 10 μm
radius_bins = np.logspace(start=-9, stop=-4, num=500)

# Create the lognormal PDF distribution
distribution_pdf = par.particles.get_lognormal_pdf_distribution(
    x_values=radius_bins,
    mode=mode,
    geometric_standard_deviation=std,
    number_of_particles=number,
)

# Create the lognormal PMF distribution
distribution_pmf = par.particles.get_lognormal_pmf_distribution(
    x_values=radius_bins,
    mode=mode,
    geometric_standard_deviation=std,
    number_of_particles=number,
)

# Plot the PDF and PMF distributions for comparison
plt.figure(figsize=(8, 6))
plt.plot(radius_bins, distribution_pdf, label="PDF")
plt.plot(radius_bins, distribution_pmf, label="PMF")
plt.xlabel("Particle Radius [m]")
plt.ylabel(r"Concentration $\dfrac{1}{m^3}$ or $\dfrac{1}{m^3 \cdot m}$")
plt.xscale(
    "log"
)  # Logarithmic scale for the x-axis to capture wide size range
plt.yscale(
    "log"
)  # Logarithmic scale for the y-axis to highlight distribution differences
plt.legend()
plt.show()
```


    
![png](output_3_0.png)
    


## Rescaling PDF

In this section, we convert the previously generated Probability Density Function (PDF) into a Probability Mass Function (PMF) to directly compare it with the original PMF distribution. This rescaling is necessary because PDF and PMF represent the distribution in different ways, and aligning them allows for a more accurate comparison.

**Rescaling the PDF**

The conversion from PDF to PMF involves rescaling the distribution so that the integral of the PDF over each bin corresponds to the particle count in that bin, similar to what is represented in the PMF. This is done using the `distribution_convert_pdf_pms` function.



```python
# Rescale the PDF to PMF
distribution_pdf_rescaled = par.util.get_pdf_distribution_in_pmf(
    x_array=radius_bins,
    distribution=distribution_pdf,
    to_pdf=False,  # Convert PDF to PMF
)

# Plot the rescaled PDF and the original PMF for comparison
plt.figure(figsize=(8, 6))
plt.plot(radius_bins, distribution_pdf_rescaled, label="PDF Rescaled to PMF")
plt.plot(radius_bins, distribution_pmf, label="PMF")
plt.xlabel("Particle Radius [m]")
plt.ylabel(r"Concentration $\dfrac{1}{m^3}$")
plt.xscale("log")  # Use a logarithmic scale for the x-axis
# plt.yscale('log')  # Optionally use a logarithmic scale for the y-axis
plt.legend()
plt.show()
```


    
![png](output_5_0.png)
    


## Rescaling PMF

After converting the PDF to a PMF for direct comparison, we now perform the reverse operation: rescaling the PMF to a PDF. This allows us to compare the original PDF with the PMF that has been adjusted to match the continuous representation of particle concentrations.

**Rescaling the PMF**

To rescale the PMF to a PDF, we use the `distribution_convert_pdf_pms` function. This conversion ensures that the PMF, which originally represented discrete particle counts in each bin, is transformed into a continuous probability density function, aligning it with the original PDF format.


```python
# Rescale the PMF to PDF
distribution_pmf_rescaled = par.util.get_pdf_distribution_in_pmf(
    x_array=radius_bins,
    distribution=distribution_pmf,
    to_pdf=True,  # Convert PMF to PDF
)

# Plot the original PDF and the rescaled PMF for comparison
plt.figure()
plt.plot(radius_bins, distribution_pdf, label="PDF")
plt.plot(radius_bins, distribution_pmf_rescaled, label="PMF Rescaled to PDF")
plt.xlabel("Particle Radius [m]")
plt.ylabel(r"Concentration $\dfrac{1}{m^3 \cdot m}$")
plt.xscale("log")  # Logarithmic scale for the x-axis to capture size range
plt.legend()
plt.show()
```


    
![png](output_7_0.png)
    


## Verifying Number Concentration

In this section, we validate the consistency of the number concentration across different representations (PDF and PMF) by performing numerical integrations and summations. This step ensures that the transformations between PDF and PMF maintain the expected total particle concentrations.

**Integrating Number Concentration for PDF**

We first integrate the original PDF distribution and the PMF that has been rescaled to a PDF to check if they yield the same total number concentration. The trapezoidal rule is used for this integration.


```python
# Integrate the original PDF to calculate the total number concentration
number_concentration_pdf = trapezoid(distribution_pdf, x=radius_bins)

# Integrate the PMF rescaled to PDF to calculate the total number concentration
number_concentration_pmf_rescaled = trapezoid(
    distribution_pmf_rescaled, x=radius_bins
)

# Print the results to compare
print(f"Number concentration from PDF: {number_concentration_pdf}")
print(
    f"Number concentration from PMF rescaled to PDF: {number_concentration_pmf_rescaled}"
)
```

    Number concentration from PDF: 1000000000000.0
    Number concentration from PMF rescaled to PDF: 988596064185.8999


**Verifying Number Concentration for PMF**

Next, we sum the original PMF distribution and the PDF that has been rescaled to a PMF. Summing these values directly gives the total number concentration, allowing us to verify consistency across the different representations.


```python
# Sum the original PMF to calculate the total number concentration
number_concentration_pmf = distribution_pmf.sum()

# Sum the PDF rescaled to PMF to calculate the total number concentration
number_concentration_pdf_rescaled = distribution_pdf_rescaled.sum()

# Print the results to compare
print(f"Number concentration from PMF: {number_concentration_pmf}")
print(
    f"Number concentration from PDF rescaled to PMF: {number_concentration_pdf_rescaled}"
)
```

    Number concentration from PMF: 1000000000000.0
    Number concentration from PDF rescaled to PMF: 1011535485753.2141


## Small Errors

When comparing the total number concentrations derived from the PDF and PMF representations, as well as their rescaled versions, we observe small discrepancies between the values.

**Sources of Error**

These small differences arise from several factors inherent in the process of transforming and integrating discrete and continuous distributions:

1. **Numerical Approximation**:
   - The integration of the PDF and the summation of the PMF involve numerical approximations, which can introduce small errors. The trapezoidal rule, used for integrating the PDF, is an approximation method that may not perfectly capture the area under the curve, especially when dealing with finely spaced bins or distributions that change rapidly in certain regions.

2. **Discretization of Continuous Distributions**:
   - When rescaling a PDF to a PMF or vice versa, we are essentially converting a continuous function into a discrete one, or vice versa. This discretization process can lead to slight inconsistencies because the continuous distribution is approximated by a finite number of bins. The exact alignment of these bins with the underlying distribution is rarely perfect, leading to small errors.

3. **Cumulative Effect of Small Differences**:
   - Small differences across many bins can accumulate, resulting in a noticeable discrepancy when summing or integrating over the entire distribution. Even if each individual difference is minute, the total error can be more significant when considering the entire size range.

**Significance of the Errors**

While these errors are small relative to the total number concentration (less than 1% in this case), they are important to acknowledge when performing precision calculations. In practical applications, these discrepancies are often considered acceptable, but they highlight the importance of understanding the limitations of numerical methods and transformations between different types of distributions.

**Mitigating the Errors**

- **Increasing the Number of Bins**: Using a higher resolution (more bins) can help reduce the error by more closely approximating the continuous distribution.
- **Refining the Interpolation Method**: More sophisticated interpolation methods may provide better alignment between the PDF and PMF during the rescaling process, further minimizing errors.
- **Error Analysis**: Incorporating error analysis into the calculations can help quantify and understand the impact of these discrepancies on the overall results.

Overall, these small errors are a natural consequence of the numerical techniques used and do not significantly detract from the accuracy of the coagulation modeling. However, being aware of their existence is crucial for interpreting results with a full understanding of the underlying processes.

## Brownian Coagulation Kernel

Before comparing the coagulation rates between the Probability Mass Function (PMF) and Probability Density Function (PDF) representations, it is essential to calculate the Brownian coagulation kernel. The kernel quantifies the rate at which particles of different sizes collide and coagulate due to Brownian motion. This matrix is a key component in determining how quickly particles in an aerosol system merge to form larger particles.

**Calculation of Particle Masses**

To calculate the coagulation kernel, we first need to determine the mass of particles in each size bin. The mass of a particle is calculated using the formula for the volume of a sphere:

$$
m = \frac{4}{3} \pi r^3 \times 1000 \, \text{kg/m}^3
$$

where $r$  is the particle radius and 1000 kg/m³ is the assumed density of the particles.


```python
# Calculate the mass of particles for each size bin
mass_bins = 4 / 3 * np.pi * radius_bins**3 * 1e3  # kg

# Calculate the Brownian coagulation kernel matrix
kernel = par.dynamics.get_brownian_kernel_via_system_state(
    particle_radius=radius_bins,
    mass_particle=mass_bins,
    temperature=293.15,  # Temperature in Kelvin (20°C)
    pressure=101325,  # Pressure in Pascals (1 atm)
    alpha_collision_efficiency=1.0,  # Assume perfect collision efficiency
)
```

## Volume Conservation

In aerosol coagulation processes, one check is to ensure that the total particle volume is conserved. While the number of particles decreases as they coagulate to form larger particles, the total volume of material should remain constant (assuming no other processes such as condensation or evaporation are occurring).

In this section, we calculate the gain, loss, and net coagulation rates using the PDF representation and verify that the total volume remains consistent.

**Calculating Coagulation Rates**

We start by calculating the gain, loss, and net coagulation rates based on the initial PDF concentration. These rates describe how particles in different size bins gain or lose mass due to coagulation.

**Converting to Volume Distribution
**
To verify volume conservation, we convert the particle concentration rates (gain, loss, and net) into volume rates by multiplying them by the volume of particles in each size bin. The volume of a particle is given by:

$$
V = \frac{4}{3} \pi r^3
$$


```python
concentration_0 = distribution_pdf

# Calculate gain and loss rates for the PDF
gain_pdf = par.dynamics.get_coagulation_gain_rate_continuous(
    radius=radius_bins,
    concentration=concentration_0,
    kernel=kernel,
)
loss_pdf = par.dynamics.get_coagulation_loss_rate_continuous(
    radius=radius_bins, concentration=concentration_0, kernel=kernel
)
net_pdf = gain_pdf - loss_pdf

# Convert gain, loss, and net rates to volume distribution
gain_volume = gain_pdf * 4 / 3 * np.pi * radius_bins**3
loss_volume = loss_pdf * 4 / 3 * np.pi * radius_bins**3
net_volume = net_pdf * 4 / 3 * np.pi * radius_bins**3

# Integrate the gain, loss, and net rates to get the total volume
gain_total_volume = trapezoid(gain_volume, x=radius_bins)
loss_total_volume = trapezoid(loss_volume, x=radius_bins)
net_total_volume = trapezoid(net_volume, x=radius_bins)

# Print the results to verify volume conservation
print(f"Gain total volume: {gain_total_volume}")
print(f"Loss total volume: {loss_total_volume}")
print(f"Net total volume: {net_total_volume}")
```

    Gain total volume: 6.75165104833036e-11
    Loss total volume: 6.752244952802652e-11
    Net total volume: -5.9390447229141716e-15


**Ensuring Volume Conservation in Coagulation Rates for PMF**

Similar to the PDF representation, it is essential to ensure that the total particle volume is conserved in the PMF (Probability Mass Function) representation during the coagulation process. The number of particles may decrease as they coagulate into larger particles, but the total volume of particles should remain constant if no other processes (like condensation or evaporation) are involved.

**Calculating Coagulation Rates for PMF**

We calculate the gain, loss, and net coagulation rates for the initial PMF concentration. These rates describe how particle concentrations change in each size bin due to coagulation.



```python
# Initial concentration for the PMF distribution
concentration_0_pmf = distribution_pmf

# Calculate gain and loss rates for the PMF
gain_pmf = par.dynamics.get_coagulation_gain_rate_discrete(
    radius=radius_bins,
    concentration=concentration_0_pmf,
    kernel=kernel,
)
loss_pmf = par.dynamics.get_coagulation_loss_rate_discrete(
    concentration=concentration_0_pmf, kernel=kernel
)
net_pmf = gain_pmf - loss_pmf

# Convert gain, loss, and net rates to volume distribution
gain_volume_pmf = gain_pmf * 4 / 3 * np.pi * radius_bins**3
loss_volume_pmf = loss_pmf * 4 / 3 * np.pi * radius_bins**3
net_volume_pmf = net_pmf * 4 / 3 * np.pi * radius_bins**3

# Sum the gain, loss, and net volumes
gain_total_volume_pmf = gain_volume_pmf.sum()
loss_total_volume_pmf = loss_volume_pmf.sum()
net_total_volume_pmf = net_volume_pmf.sum()

# Print the results to verify volume conservation
print(f"Gain total volume PMF: {gain_total_volume_pmf}")
print(f"Loss total volume PMF: {loss_total_volume_pmf}")
print(f"Net total volume PMF: {net_total_volume_pmf}")
```

    Gain total volume PMF: 6.674655653136e-11
    Loss total volume PMF: 6.75224495280265e-11
    Net total volume PMF: -7.758929966665061e-13


## Gain and Loss Rates Comparison

To understand the differences and similarities between the PDF and PMF representations in the context of particle coagulation, we convert the PMF gain and loss rates to their corresponding PDF forms. This allows for a direct comparison of how each representation handles particle concentration changes across different size ranges.

**Converting PMF to PDF**

The gain and loss rates calculated for the PMF representation are converted to PDF using the `distribution_convert_pdf_pms` function. This conversion enables us to plot and compare the results on the same scale and with the same units as the original PDF.

**Explanation**

- **Comparison of Gain and Loss Rates**: The first plot compares the gain and loss rates between the original PDF and the converted PMF. By plotting these on the same axes, we can observe how closely the PMF (when converted to PDF) matches the behavior of the original PDF. The solid lines represent the PDF results, while the dashed lines represent the PMF converted to PDF.

- **Comparison of Net Rates**: The second plot focuses on the net gain and loss rates, which are calculated as the difference between the gain and loss rates. This plot helps in understanding whether the differences in the gain and loss rates between the PDF and converted PMF lead to any significant discrepancies in the overall net rate of particle concentration change.

**Interpretation**

- **Matching Behavior**: Ideally, the converted PMF should closely match the original PDF, indicating that both representations handle coagulation rates similarly. Any differences observed in the plots can provide insights into the nuances of each method, such as how they handle small particle sizes or how numerical approximations might affect the results.
  
- **Significance of Differences**: While small differences between the PDF and converted PMF may arise due to numerical methods, these differences can highlight the strengths and limitations of each approach in representing particle size distributions and their evolution during coagulation.


**PMF Gain Error**

The PMF gain rate is slightly off, and we are still looking into the issue. We will update this section once we have resolved the discrepancy.



```python
# Convert PMF gain and loss rates to PDF
gain_pmf_to_pdf = par.util.get_pdf_distribution_in_pmf(
    x_array=radius_bins, distribution=gain_pmf, to_pdf=True
)
loss_pmf_to_pdf = par.util.get_pdf_distribution_in_pmf(
    x_array=radius_bins, distribution=loss_pmf, to_pdf=True
)
net_pmf_to_pdf = gain_pmf_to_pdf - loss_pmf_to_pdf

# Plot the gain and loss rates for both PDF and converted PMF
plt.figure()
plt.plot(radius_bins, gain_pdf, label="Gain (PDF)", linewidth=4)
plt.plot(radius_bins, -1 * loss_pdf, label="Loss (PDF)", linewidth=4)
plt.plot(
    radius_bins, gain_pmf_to_pdf, label="Gain (PMF to PDF)", linestyle="--"
)
plt.plot(
    radius_bins,
    -1 * loss_pmf_to_pdf,
    label="Loss (PMF to PDF)",
    linestyle="--",
)
plt.xlabel("Particle Radius [m]")
plt.ylabel(r"Rate $\dfrac{1}{m^3 s \cdot m}$")
plt.xscale("log")
plt.title("PDF: Gain and Loss Comparison")
plt.legend()
plt.show()

# Plot the net gain and loss rates for both PDF and converted PMF
plt.figure()
plt.plot(radius_bins, net_pdf, label="Net (PDF)")
plt.plot(radius_bins, net_pmf_to_pdf, label="Net (PMF to PDF)")
plt.xlabel("Particle Radius [m]")
plt.ylabel(r"Rate $\dfrac{1}{m^3 s \cdot m}$")
plt.xscale("log")
plt.title("PDF: Net Gain and Loss Comparison")
plt.legend()
plt.show()
```


    
![png](output_20_0.png)
    



    
![png](output_20_1.png)
    


## Simulating PDF Coagulation

In this section, we simulate the evolution of a particle size distribution over time as particles undergo coagulation, using the Probability Density Function (PDF) representation. The simulation tracks how the distribution changes at different time steps, providing insight into how the particle population evolves under the influence of Brownian coagulation.

**Simulation Setup**

- **Initial Distribution**: The simulation begins with the initial particle size distribution (`distribution_0`), which is based on the PDF calculated earlier.
- **Time Array**: The simulation runs over a time span from 0 to 1000 seconds, with 50 discrete time steps. The time interval between each step is calculated to update the distribution as coagulation progresses.


```python
# Initial distribution for the simulation
distribution_0 = distribution_pdf
distribution_i = distribution_0

# Define the time array for the simulation
time_array = np.linspace(
    start=0, stop=1000, num=50
)  # Time span of 1000 seconds
time_interval = (
    time_array[1] - time_array[0]
)  # Time interval between each step

# Array to store the distribution at each time step
distribution_time = np.zeros([len(time_array), len(distribution_0)])

# Simulate the coagulation process over time
for i, dpa in enumerate(time_array):
    if i > 0:
        # Calculate coagulation gain and loss at the current time step
        coag_gain_i = par.dynamics.get_coagulation_gain_rate_continuous(
            radius=radius_bins,
            concentration=distribution_i,
            kernel=kernel,
        )
        coag_loss_i = par.dynamics.get_coagulation_loss_rate_continuous(
            radius=radius_bins, concentration=distribution_i, kernel=kernel
        )

        # Calculate the net change in distribution due to coagulation
        net_change = (coag_gain_i - coag_loss_i) * time_interval
        distribution_i = distribution_i + net_change

        # Ensure no negative concentrations (set to zero if less than zero)
        distribution_i[distribution_i < 0] = 0

    # Store the updated distribution for the current time step
    distribution_time[i, :] = distribution_i
```

**Visualizing the Evolution of the Particle Size Distribution**

After simulating the coagulation process over time, we can visualize how the particle size distribution evolves at different time steps. This plot compares the initial distribution with the distribution at a mid-point and at the end of the simulation, highlighting the changes that occur due to coagulation.


```python
# Set up the plot
fig, ax = plt.subplots(1, 1, figsize=[9, 6])

# Define the radius bins
radius = radius_bins

# Plot the initial distribution, mid-point distribution, and final distribution
ax.semilogx(
    radius, distribution_0, "-b", label="Initial"
)  # Initial distribution
ax.semilogx(
    radius, distribution_time[25, :], "--", label="t-step=25"
)  # Mid-point
ax.semilogx(
    radius, distribution_time[-1, :], "-r", label="t=end"
)  # Final distribution

# Set the limits for the x-axis to focus on the relevant size range
ax.set_xlim([2e-8, 1e-6])

# Add legend to distinguish between different time steps
ax.legend()

# Label the y-axis to indicate concentration units
ax.set_ylabel(r"Concentration $\dfrac{1}{m^3 \cdot m}$")

# Label the x-axis for particle radius
ax.set_xlabel("Radius [m]")

# Add grid lines for better readability
ax.grid(True, alpha=0.5)

# Show the plot
plt.show()
```


    
![png](output_24_0.png)
    


**Visualizing Particle Size Distribution Evolution Over Time**

To further understand how the particle size distribution evolves during the coagulation process, we can create a 2D image plot. In this plot, time is represented on the x-axis, particle size (radius) on the y-axis, and the concentration is color-coded. This type of plot provides a comprehensive view of how both small and large particles change in concentration over the entire simulation period.



```python
# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create an image plot (2D histogram) with time on the x-axis, radius on the y-axis, and concentration as color
c = ax.pcolormesh(
    time_array,
    radius_bins,
    distribution_time.T,
    shading="auto",
    cmap="viridis",
)

# Set the y-axis to a logarithmic scale to capture the wide range of particle sizes
ax.set_ylim([5e-8, 1e-6])
ax.set_yscale("log")

# Label the axes
ax.set_xlabel("Time (s)")
ax.set_ylabel("Particle Radius (m)")

# Add a color bar to indicate the concentration levels
cbar = fig.colorbar(c, ax=ax)
cbar.set_label(r"Concentration $\dfrac{1}{m^3 \cdot m}$")

# Add a title to the plot
ax.set_title("Evolution of Particle Size Distribution Over Time")

# Show the plot
plt.show()
```


    
![png](output_26_0.png)
    


## ODE solver

In this section, we simulate the coagulation process using an ordinary differential equation (ODE) solver to track the evolution of particle concentrations in different size bins over time.


```python
from scipy.integrate import solve_ivp
import numpy as np


# Define the coagulation ODE system
def coagulation_ode(t, distribution, radius_bins, kernel):
    """
    Compute the derivative of the distribution with respect to time
    (i.e., the rate of change due to coagulation).

    Arguments:
        t: Time variable (not used explicitly here, but required by solve_ivp).
        distribution: The current particle size distribution (array).
        radius_bins: The bins for the particle radius.
        kernel: The coagulation kernel.

    Returns:
        The time derivative of the particle distribution.
    """
    coag_gain = par.dynamics.get_coagulation_gain_rate_continuous(
        radius=radius_bins, concentration=distribution, kernel=kernel
    )
    coag_loss = par.dynamics.get_coagulation_loss_rate_continuous(
        radius=radius_bins, concentration=distribution, kernel=kernel
    )

    # Net change in distribution due to coagulation
    net_change = coag_gain - coag_loss

    return net_change


# Initial distribution for the simulation
distribution_0 = distribution_pdf

# Define the time array for the simulation
time_array = np.linspace(
    start=0, stop=1000, num=50
)  # Time span of 1000 seconds

# Use scipy's solve_ivp to solve the ODE system
solution = solve_ivp(
    fun=coagulation_ode,  # ODE function
    t_span=(time_array[0], time_array[-1]),  # Time span
    y0=distribution_0,  # Initial distribution
    t_eval=time_array,  # Time points to store the solution
    args=(
        radius_bins,
        kernel,
    ),  # Additional arguments to coagulation_ode
    method="BDF",  # Integration method (default)
    max_step=1e2,
)

# The solution is stored in solution.y (each column is a time step)
distribution_solver = solution.y.T  # Transpose to match original format
```

## Comparison of Different Integration Methods


```python
# Set up the plot
fig, ax = plt.subplots(1, 1, figsize=[9, 6])

# Define the radius bins
radius = radius_bins

# Plot the initial distribution, mid-point distribution, and final distribution
ax.semilogx(
    radius, distribution_solver[0, :], "-b", label="Initial"
)  # Initial distribution
ax.semilogx(
    radius, distribution_solver[25, :], "--", label="t-step=25"
)  # Mid-point
ax.semilogx(
    radius, distribution_solver[-1, :], "-r", label="t=end"
)  # Final distribution

# Plot the initial distribution, mid-point distribution, and final distribution
ax.semilogx(
    radius, distribution_0, "-b", label="Initial-loop"
)  # Initial distribution
ax.semilogx(
    radius, distribution_time[25, :], "--", label="t-step=25 loop"
)  # Mid-point
ax.semilogx(
    radius, distribution_time[-1, :], "-r", label="t=end loop"
)  # Final distribution

# Set the limits for the x-axis to focus on the relevant size range
ax.set_xlim([2e-8, 1e-6])

# Add legend to distinguish between different time steps
ax.legend()

# Label the y-axis to indicate concentration units
ax.set_ylabel(r"Concentration $\dfrac{1}{m^3 \cdot m}$")

# Label the x-axis for particle radius
ax.set_xlabel("Radius [m]")

# Add grid lines for better readability
ax.grid(True, alpha=0.5)

# Show the plot
plt.show()
```


    
![png](output_30_0.png)
    



---
# Coagulation_Basic_4_ParticleResolved.md

# Coagulation Basic 4: Particle Resolved

**Introduction**

In aerosol science, understanding particle-particle interactions is crucial for predicting the evolution of particle size distributions. One such interaction is **coagulation**, where two particles collide and merge into a larger particle. Accurately modeling coagulation at the level of individual particles is known as the **particle-resolved method**.

The particle-resolved method tracks each particle individually, considering its unique properties and interactions. This method provides the most detailed representation of aerosol dynamics, making it ideal for cases where precision is paramount, such as in cloud microphysics or laboratory-scale studies.

However, this approach is computationally intensive because it requires simulating every individual particle and its interactions. Unlike the **super droplet method**, which uses statistical representations to reduce computational load, the direct particle-resolved method does not aggregate particles into larger groups. Instead, every particle is treated independently, ensuring that every interaction is explicitly modeled.

This notebook provides a step-by-step guide to simulating coagulation using a pure particle-resolved approach, demonstrating how individual particles evolve over time without any simplifications or approximations in particle grouping.

**Setup and Imports**

We'll start by importing the necessary libraries and setting up the environment.


```python
# %% particle resolved coagulation example
import numpy as np  # For numerical operations and array manipulations
import matplotlib.pyplot as plt  # For plotting graphs and visualizations

import particula as par  # Import the particula module
```

## Generating Distribution

In this section, we generate a sample particle size distribution following a lognormal distribution. The lognormal distribution is commonly used in aerosol science to describe particle size distributions. 

**Coagulation Kernel**

We also calculate the Brownian coagulation kernel for these particles, which quantifies the probability of coagulation between particles of different sizes.

**Random seed**

We set a random seed to ensure reproducibility of the results.


```python
# lognormal spacing

radius_bins = np.logspace(
    -9, -6, num=20
)  # Define the radius bins for the distribution
mass_bins = (
    4 / 3 * np.pi * radius_bins**3 * 1000
)  # Calculate the mass of the particles in the bins

kernel = par.dynamics.get_brownian_kernel_via_system_state(
    particle_radius=radius_bins,
    mass_particle=mass_bins,
    temperature=298.15,
    pressure=101325,
)  # Calculate the Brownian coagulation kernel for the radius bins

random_generator = np.random.default_rng(12345)
```

**Sampling the Particle Distribution**

We then sample particles from the lognormal distribution. These particles will be sorted by size to prepare for the coagulation step.


```python
# %% sample particle distribution
particle_radius = par.particles.get_lognormal_sample_distribution(
    mode=np.array([1e-8, 1e-7]),
    geometric_standard_deviation=np.array([1.4, 1.4]),
    number_of_particles=np.array([5000, 1000]),
    number_of_samples=10_000,
)
# particle_radius = np.sort(particle_radius)
particles_original = particle_radius.copy()
```

## Coagulation Step

In the coagulation step, particles collide and merge over a given time step. The super droplet method efficiently simulates this process by adjusting the particle sizes and concentrations based on the calculated kernel and the specified volume and time step.


```python
# %% Coagulation step

delta_t = 100  # time step in seconds
total_number_concentration = 1_000_000 * 1e6  # particles per m^3
total_number_tracked = len(particle_radius)
volume_sim = total_number_tracked / total_number_concentration

loss_gain_index = par.dynamics.get_particle_resolved_coagulation_step(
    particle_radius=particle_radius,
    kernel=kernel,
    kernel_radius=radius_bins,
    volume=volume_sim,
    time_step=delta_t,
    random_generator=random_generator,
)
particle_radius, gain, loss = par.dynamics.get_particle_resolved_update_step(
    particle_radius=particle_radius,
    gain=np.zeros_like(particle_radius),
    loss=np.zeros_like(particle_radius),
    small_index=loss_gain_index[:, 0],
    large_index=loss_gain_index[:, 1],
)

print(loss_gain_index.shape)
print(loss_gain_index)
```

    (3201, 2)
    [[3074 9486]
     [3535 9551]
     [2105 3968]
     ...
     [9772 8800]
     [9111 8947]
     [9496 8414]]


**Plotting the New Distribution**

Finally, we plot the particle size distribution before and after coagulation. This visualization helps us understand the effect of the coagulation process on the particle size distribution.



```python
# %% plot new distribution
fig, ax = plt.subplots()
ax.hist(
    particles_original, bins=100, histtype="step", color="black", density=True
)
ax.hist(
    particle_radius[particle_radius > 0],
    bins=100,
    histtype="step",
    color="blue",
    density=True,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Frequency")
plt.show()
```


    
![png](output_9_0.png)
    


**Plotting the Loss and Gain of Particles**

We also plot the loss and gain of particles due to coagulation. This visualization provides insights into the coagulation process and how it affects the particle population.


```python
fig, ax = plt.subplots()
ax.hist(loss[loss > 0], bins=100, histtype="step", color="red", density=True)
ax.hist(gain[gain > 0], bins=100, histtype="step", color="green", density=True)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Frequency")
plt.show()
```


    
![png](output_11_0.png)
    


## Direct Time Stepping

With the first coagulation step completed, we can now proceed to the next time step. We repeat the coagulation process for the new particle distribution, updating the particle sizes and concentrations accordingly. This iterative approach allows us to simulate the evolution of the particle size distribution over time.

Here we use a simple for loop to perform multiple coagulation steps. In practice, more sophisticated time-stepping methods may be used to improve efficiency and accuracy.


```python
# Initial distribution for the simulation
particles_0 = particles_original
particles_i = particles_0

# Define the time array for the simulation
time_array = np.linspace(
    start=0, stop=1000, num=100
)  # Time span of 1000 seconds
time_interval = (
    time_array[1] - time_array[0]
)  # Time interval between each step

# Array to store the distribution at each time step
particles_matrix = np.zeros([len(time_array), len(particles_0)])

# Simulate the coagulation process over time
for i, dpa in enumerate(time_array):
    if i > 0:

        loss_gain_index = par.dynamics.get_particle_resolved_coagulation_step(
            particle_radius=particles_i,
            kernel=kernel,
            kernel_radius=radius_bins,
            volume=volume_sim,
            time_step=time_interval,
            random_generator=random_generator,
        )
        particles_i, _, _ = par.dynamics.get_particle_resolved_update_step(
            particle_radius=particles_i,
            gain=np.zeros_like(particles_i),
            loss=np.zeros_like(particles_i),
            small_index=loss_gain_index[:, 0],
            large_index=loss_gain_index[:, 1],
        )

        # Ensure no negative concentrations (set to zero if less than zero)
        particles_i[particles_i < 0] = 0

    # Store the updated distribution for the current time step
    particles_matrix[i, :] = particles_i
```

**Plotting the Final Distribution**

Finally, we plot the final particle size distribution after multiple coagulation steps. This visualization shows how the particle size distribution evolves over time due to coagulation.


```python
filtered = particles_matrix[-1, :] > 0

fig, ax = plt.subplots()
ax.hist(
    particles_original, bins=100, histtype="step", color="black", density=True
)
ax.hist(
    particles_matrix[-1, filtered],
    bins=100,
    histtype="step",
    color="blue",
    density=True,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Frequency")
plt.show()
```


    
![png](output_15_0.png)
    



```python
# plot total number of particles
total_number = np.sum(particles_matrix > 0, axis=1)

fig, ax = plt.subplots()
ax.plot(time_array, total_number)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Number of particles")
plt.show()
```


    
![png](output_16_0.png)
    



```python
# convert to concentration
total_concentration = total_number / volume_sim

fig, ax = plt.subplots()
ax.plot(time_array, total_concentration)
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"Concentration $(m^{-3})$")
plt.show()
```


    
![png](output_17_0.png)
    


## Conclusion

This notebook demonstrated the use of the particle resolved coagulation method to simulate particle coagulation.



---
# Coagulation_with_Charge_functional.md

# Coagulation with Charge Effects
In this tutorial, we explore how electrical charge influences the coagulation (collisions and coalescence) of aerosol particles. We will:
- Create a size distribution for aerosol particles.
- Calculate the Coulomb potential ratio and related properties.
- Compute the coagulation kernel considering charge effects.
- Plot the coagulation kernel.
- Simulate the time evolution of the particle size distribution due to coagulation.
- Plot the particle concentration and coagulation rates.

*Import Necessary Libraries*
We import standard libraries like `numpy` and `matplotlib` for numerical computations and plotting. We also import specific functions from the `particula` package, which is used for aerosol dynamics simulations.


```python
import numpy as np
import matplotlib.pyplot as plt

# particula imports
import particula as par
```

## Define the Particle Size Distribution
We create a size distribution for aerosol particles using a logarithmic scale for particle radius, ranging from 1 nm to 10 μm. We calculate the mass of particles in each size bin assuming they are spherical and have a standard density.
 Define the bins for particle radius using a logarithmic scale


```python
radius_bins = np.logspace(start=-9, stop=-4, num=250)  # m (1 nm to 10 μm)

# Calculate the mass of particles for each size bin
# The mass is calculated using the formula for the volume of a sphere (4/3 * π * r^3)
# and assuming a particle density of 1 g/cm^3 (which is 1000 kg/m^3 in SI units).
mass_bins = 4 / 3 * np.pi * radius_bins**3 * 1e3  # kg
```

## Define Particle Charges
We assign charges to the particles. In this example, we assign negative charges to the first 33% of particles and positive charges to the remaining 66% of particles. The charges are assigned based on the particle radius, with negative charges ranging from 10 to 1 and positive charges ranging from 1 to 500. We then plot the charge distribution against the particle radius.
 Determine the number of radius bins


```python
n_bins = len(radius_bins)

# Define the split index where charges transition from negative to positive
split_index = n_bins // 3  # Assign the first 25% of particles negative charges

# Generate logarithmically spaced magnitudes for negative charges from 10 to 1
neg_magnitudes = np.logspace(np.log10(10), np.log10(1), num=split_index)
neg_charges = -neg_magnitudes  # Assign negative sign

# Generate logarithmically spaced magnitudes for positive charges from 1 to 500
pos_magnitudes = np.logspace(
    np.log10(1), np.log10(500), num=n_bins - split_index
)
pos_charges = pos_magnitudes  # Positive charges

# Combine the negative and positive charges into one array
charge_array = np.concatenate((neg_charges, pos_charges))

# Plot charge vs. radius
fig, ax = plt.subplots()
ax.plot(radius_bins, charge_array, marker="o", linestyle="none")
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=1)
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Particle charge (elementary charges)")
ax.set_title("Particle Charge vs. Radius")
ax.grid(True, which="both", ls="--")
plt.show()

temperature = 298.15
```


    
![png](output_6_0.png)
    


## Calculate Coulomb Potential Ratio and Related Properties
In this section, we compute several properties necessary for calculating the coagulation kernel with charge effects:
- **Coulomb Potential Ratio**: Using `coulomb_enhancement.ratio`, we calculate the dimensionless Coulomb potential ratio, which quantifies the electrostatic interaction between charged particles.
- **Dynamic Viscosity**: Obtained from `get_dynamic_viscosity`, needed for calculating friction factors.
- **Mean Free Path**: Calculated using `molecule_mean_free_path`, important for determining the Knudsen number.
- **Knudsen Number**: Computed with `calculate_knudsen_number`, it characterizes the flow regime of the particles.
- **Slip Correction Factor**: Using `cunningham_slip_correction`, accounts for non-continuum effects at small particle sizes.
- **Friction Factor**: Calculated with `friction_factor`, needed for determining particle mobility.
- **Diffusive Knudsen Number**: Using `diffusive_knudsen_number`, combines the effects of particle diffusion and electrostatic interactions.


```python
coulomb_potential_ratio = par.particles.get_coulomb_enhancement_ratio(
    radius_bins, charge_array, temperature=temperature
)
dynamic_viscosity = par.gas.get_dynamic_viscosity(temperature=temperature)
mol_free_path = par.gas.get_molecule_mean_free_path(
    temperature=temperature, dynamic_viscosity=dynamic_viscosity
)
knudsen_number = par.particles.get_knudsen_number(
    mean_free_path=mol_free_path, particle_radius=radius_bins
)
slip_correction = par.particles.get_cunningham_slip_correction(
    knudsen_number=knudsen_number
)


friction_factor_value = par.particles.get_friction_factor(
    particle_radius=radius_bins,
    dynamic_viscosity=dynamic_viscosity,
    slip_correction=slip_correction,
)

diffusive_knudsen_values = par.particles.get_diffusive_knudsen_number(
    particle_radius=radius_bins,
    particle_mass=mass_bins,
    friction_factor=friction_factor_value,
    coulomb_potential_ratio=coulomb_potential_ratio,
    temperature=temperature,
)
```

## Compute the Non-Dimensional Coagulation Kernel
The non-dimensional coagulation kernel is calculated using `coulomb_chahl2019`, which incorporates charge effects into the rate at which particles collide.


```python
non_dimensional_kernel = par.dynamics.get_coulomb_kernel_chahl2019(
    diffusive_knudsen=diffusive_knudsen_values,
    coulomb_potential_ratio=coulomb_potential_ratio,
)
```

## Calculate Coulomb Enhancement Factors
We compute the Coulomb enhancement factors in both the kinetic and continuum limits:
- **Kinetic Limit**: Using `coulomb_enhancement.kinetic`, applicable when particle motions are dominated by random thermal motion.
- **Continuum Limit**: Using `coulomb_enhancement.continuum`, applicable when particles are larger and motions are influenced by continuous fluid flow.


```python
coulomb_kinetic_limit = par.particles.get_coulomb_kinetic_limit(
    coulomb_potential_ratio
)
coulomb_continuum_limit = par.particles.get_coulomb_continuum_limit(
    coulomb_potential_ratio
)

sum_of_radii = radius_bins[:, np.newaxis] + radius_bins[np.newaxis, :]
reduced_mass = par.util.get_reduced_self_broadcast(mass_bins)
```

## Compute the Dimensional Coagulation Kernel
The dimensional coagulation kernel combines all the previously calculated factors and gives the actual rate at which particles of different sizes collide and stick together due to coagulation, considering charge effects.


```python
dimensional_kernel = (
    non_dimensional_kernel
    * friction_factor_value
    * sum_of_radii**3
    * coulomb_kinetic_limit**2
    / (reduced_mass * coulomb_continuum_limit)
)
```

## Plot the Coagulation Kernel
We plot the coagulation kernel as a function of particle radius to visualize how charge affects the coagulation rates across different particle sizes.


```python
fig, ax = plt.subplots()

# Plot negative charges in blue
ax.plot(
    radius_bins,
    dimensional_kernel[:, :split_index],
    linewidth=0.2,
    color="blue",
    label="Negative Charges",
)

# Plot positive charges in red
ax.plot(
    radius_bins,
    dimensional_kernel[:, split_index:],
    linewidth=0.2,
    color="red",
    label="Positive Charges",
)

# ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-30, 1e4)
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Coagulation kernel (m^3/s)")
ax.set_title("Coagulation kernel vs particle radius")
ax.text(1e-9, 1e-22, "Negative Charges", color="blue")
ax.text(1e-6, 1e-6, "Positive Charges", color="red")
plt.show()
```


    
![png](output_16_0.png)
    


## Simulate Coagulation Over a Time Step
We calculate the gain and loss rates of particle concentrations due to coagulation using the previously computed kernel. The net rate of change in particle concentration is obtained by subtracting the loss rate from the gain rate.
 get rates of coagulation dn/dt
 make a number concentration distribution


```python
number_concentration = par.particles.get_lognormal_pmf_distribution(
    x_values=radius_bins,
    mode=np.array([10e-9, 200e-9, 1000e-9]),  # m
    geometric_standard_deviation=np.array([1.4, 1.5, 1.8]),
    number_of_particles=np.array([1e12, 1e12, 1e12]),  # per m^3
)

gain_rate = par.dynamics.get_coagulation_gain_rate_discrete(
    radius=radius_bins,
    concentration=number_concentration,
    kernel=dimensional_kernel,
)
loss_rate = par.dynamics.get_coagulation_loss_rate_discrete(
    concentration=number_concentration,
    kernel=dimensional_kernel,
)

net_rate = gain_rate - loss_rate
```

## Plot Particle Concentration and Coagulation Rates
We visualize the initial particle concentration distribution and the coagulation rates to understand how charge effects influence the coagulation process over time.
- **Particle Concentration**: Shows the number concentration of particles across different sizes.
- **Coagulation Rates**: Displays the gain, loss, and net rates of coagulation, highlighting how particles of different sizes contribute to the overall process.


```python
fig, ax = plt.subplots()
ax.plot(radius_bins, number_concentration)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Number concentration (1/m^3)")
ax.set_title("Number concentration vs particle radius")
plt.show()

# plot the rates
fig, ax = plt.subplots()
ax.plot(radius_bins, gain_rate, label="Gain rate")
ax.plot(radius_bins, -1 * loss_rate, label="Loss rate")
ax.plot(radius_bins, net_rate, label="Net rate", linestyle="--")
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Coagulation rate 1/ (m^3 s)")
ax.set_title("Coagulation rate vs particle radius")
ax.legend()
plt.show()
```


    
![png](output_20_0.png)
    



    
![png](output_20_1.png)
    



---
# Coagulation_with_Charge_objects.md

# Coagulation with Various Kernel Strategies
In this tutorial, we demonstrate how to compute coagulation kernels using different kernel strategies, including the effects of electrical charge on aerosol particles.
We will:
- Define a particle size distribution.
- Assign charges to particles.
- Calculate necessary particle properties.
- Compute the coagulation kernel using different `KernelStrategy` classes.
- Plot and compare the results.

## Import Necessary Libraries
We import standard libraries and specific functions from the `particula` package that will be used in this tutorial.


```python
import numpy as np
import matplotlib.pyplot as plt

# Particula imports
import particula as par
```

## Define the Particle Size Distribution
We create a size distribution for aerosol particles using a logarithmic scale for particle radius, ranging from 1 nm to 10 μm. We calculate the mass of particles in each size bin assuming they are spherical and have a standard density.


```python
# Define the bins for particle radius using a logarithmic scale
radius_bins = np.logspace(start=-9, stop=-4, num=250)  # m (1 nm to 100 μm)

# Calculate the mass of particles for each size bin
mass_bins = 4 / 3 * np.pi * radius_bins**3 * 1e3  # kg
```

## Define Particle Charges
We assign charges to the particles. In this example, we assign negative charges to the smallest particles and positive charges to the larger particles. The charges are assigned based on the particle radius, with negative charges ranging
from -10 to -1 and positive charges ranging from +1 to +500 on a logarithmic scale.
We then plot the charge distribution against the particle radius.


```python
# Determine the number of radius bins
n_bins = len(radius_bins)

# Define the split index where charges transition from negative to positive
split_index = (
    n_bins // 3
)  # Assign the first third of particles negative charges

# Generate logarithmically spaced magnitudes for negative charges from 10 to 1
neg_magnitudes = np.logspace(np.log10(10), np.log10(1), num=split_index)
neg_charges = -neg_magnitudes  # Assign negative sign

# Generate logarithmically spaced magnitudes for positive charges from 1 to 500
pos_magnitudes = np.logspace(
    np.log10(1), np.log10(500), num=n_bins - split_index
)
pos_charges = pos_magnitudes  # Positive charges

# Combine the negative and positive charges into one array
charge_array = np.concatenate((neg_charges, pos_charges))

# Plot charge vs. radius
fig, ax = plt.subplots()
ax.plot(radius_bins, charge_array, marker="o", linestyle="none")
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=1)
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Particle charge (elementary charges)")
ax.set_title("Particle Charge vs. Radius")
ax.grid(True, which="both", ls="--")
plt.show()
```


    
![png](output_6_0.png)
    


## Calculate Coulomb Potential Ratio and Related Properties
In this section, we compute several properties necessary for calculating the coagulation kernel with charge effects:
- **Coulomb Potential Ratio**: Using `coulomb_enhancement.ratio`, we calculate the dimensionless Coulomb potential ratio, which quantifies the electrostatic interaction between charged particles.
- **Dynamic Viscosity**: Obtained from `get_dynamic_viscosity`, needed for calculating friction factors.
- **Mean Free Path**: Calculated using `molecule_mean_free_path`, important for determining the Knudsen number.
- **Knudsen Number**: Computed with `calculate_knudsen_number`, it characterizes the flow regime of the particles.
- **Slip Correction Factor**: Using `cunningham_slip_correction`, accounts for non-continuum effects at small particle sizes.
- **Friction Factor**: Calculated with `friction_factor`, needed for determining particle mobility.
- **Diffusive Knudsen Number**: Using `diffusive_knudsen_number`, combines the effects of particle diffusion and electrostatic interactions.


```python
temperature = 298.15  # Temperature in Kelvin

# Calculate Coulomb potential ratio
coulomb_potential_ratio = par.particles.get_coulomb_enhancement_ratio(
    radius_bins, charge_array, temperature=temperature
)

# Calculate gas properties
dynamic_viscosity = par.gas.get_dynamic_viscosity(temperature=temperature)
mol_free_path = par.gas.get_molecule_mean_free_path(
    temperature=temperature, dynamic_viscosity=dynamic_viscosity
)

# Calculate Knudsen number
knudsen_number = par.particles.get_knudsen_number(
    mean_free_path=mol_free_path, particle_radius=radius_bins
)

# Calculate slip correction factor
slip_correction = par.particles.get_cunningham_slip_correction(
    knudsen_number=knudsen_number
)

# Calculate friction factor
friction_factor_value = par.particles.get_friction_factor(
    particle_radius=radius_bins,
    dynamic_viscosity=dynamic_viscosity,
    slip_correction=slip_correction,
)

# Calculate diffusive Knudsen number
diffusive_knudsen_values = par.particles.get_diffusive_knudsen_number(
    particle_radius=radius_bins,
    particle_mass=mass_bins,
    friction_factor=friction_factor_value,
    coulomb_potential_ratio=coulomb_potential_ratio,
    temperature=temperature,
)

# Prepare quantities for kernel calculations
sum_of_radii = radius_bins[:, np.newaxis] + radius_bins[np.newaxis, :]
reduced_mass = par.util.get_reduced_self_broadcast(mass_bins)
reduced_friction_factor = (
    friction_factor_value[:, np.newaxis]
    * friction_factor_value[np.newaxis, :]
    / (
        friction_factor_value[:, np.newaxis]
        + friction_factor_value[np.newaxis, :]
    )
)
```

## Hard Sphere Kernel Strategy
The **Hard Sphere** kernel strategy assumes that particles interact as hard spheres without any additional forces like electrostatic interactions. This is the simplest kernel and serves as a baseline for comparison.
**References:**
- This strategy is based on classical collision theory.


```python
# Instantiate the Hard Sphere kernel strategy
kernel_strategy_hs = par.dynamics.HardSphereKernelStrategy()

# Compute the dimensionless kernel
dimensionless_kernel_hs = kernel_strategy_hs.dimensionless(
    diffusive_knudsen=diffusive_knudsen_values,
    coulomb_potential_ratio=None,  # Not used in this strategy
)

# Compute the dimensional kernel
dimensional_kernel_hs = kernel_strategy_hs.kernel(
    dimensionless_kernel=dimensionless_kernel_hs,
    coulomb_potential_ratio=coulomb_potential_ratio,
    sum_of_radii=sum_of_radii,
    reduced_mass=reduced_mass,
    reduced_friction_factor=reduced_friction_factor,
)

# Plot the Hard Sphere kernel
fig, ax = plt.subplots()
c = ax.pcolormesh(
    radius_bins, radius_bins, np.log10(dimensional_kernel_hs), shading="auto"
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius r_i (m)")
ax.set_ylabel("Particle radius r_j (m)")
ax.set_title("Hard Sphere Coagulation Kernel")
fig.colorbar(c, ax=ax, label="log10(Kernel) (m³/s)")
plt.show()
```


    
![png](output_10_0.png)
    


## Coulomb Dyachkov 2007 Kernel Strategy
The **Coulomb Dyachkov 2007** kernel strategy accounts for the Coulomb potential between charged particles, as described by Dyachkov et al. (2007). It modifies the collision kernel to include electrostatic interactions.
**References:**
- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of particles in the transition regime: The effect of the Coulomb potential. *Journal of Chemical Physics*, 126(12). [Link](https://doi.org/10.1063/1.2713719)


```python
# %%

# Instantiate the Coulomb Dyachkov 2007 kernel strategy
kernel_strategy_dyachkov = par.dynamics.CoulombDyachkov2007KernelStrategy()

# Compute the dimensionless kernel
dimensionless_kernel_dyachkov = kernel_strategy_dyachkov.dimensionless(
    diffusive_knudsen=diffusive_knudsen_values,
    coulomb_potential_ratio=coulomb_potential_ratio,
)

# Compute the dimensional kernel
dimensional_kernel_dyachkov = kernel_strategy_dyachkov.kernel(
    dimensionless_kernel=dimensionless_kernel_dyachkov,
    coulomb_potential_ratio=coulomb_potential_ratio,
    sum_of_radii=sum_of_radii,
    reduced_mass=reduced_mass,
    reduced_friction_factor=reduced_friction_factor,
)

# Plot the Coulomb Dyachkov 2007 kernel
fig, ax = plt.subplots()
c = ax.pcolormesh(
    radius_bins,
    radius_bins,
    np.log10(dimensional_kernel_dyachkov),
    shading="auto",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius r_i (m)")
ax.set_ylabel("Particle radius r_j (m)")
ax.set_title("Coulomb Dyachkov 2007 Coagulation Kernel")
fig.colorbar(c, ax=ax, label="Kernel (m³/s)")
plt.show()
```


    
![png](output_12_0.png)
    


## Coulomb Gatti 2008 Kernel Strategy
The **Coulomb Gatti 2008** kernel strategy introduces an analytical model for particle charging in plasmas over a wide range of collisionality, as presented by Gatti and Kortshagen (2008).
**References:**
- Gatti, M., & Kortshagen, U. (2008). Analytical model of particle charging in plasmas over a wide range of collisionality. *Physical Review E*, 78(4). [Link](https://doi.org/10.1103/PhysRevE.78.046402)


```python
# %%

# Instantiate the Coulomb Gatti 2008 kernel strategy
kernel_strategy_gatti = par.dynamics.CoulombGatti2008KernelStrategy()

# Compute the dimensionless kernel
dimensionless_kernel_gatti = kernel_strategy_gatti.dimensionless(
    diffusive_knudsen=diffusive_knudsen_values,
    coulomb_potential_ratio=coulomb_potential_ratio,
)

# Compute the dimensional kernel
dimensional_kernel_gatti = kernel_strategy_gatti.kernel(
    dimensionless_kernel=dimensionless_kernel_gatti,
    coulomb_potential_ratio=coulomb_potential_ratio,
    sum_of_radii=sum_of_radii,
    reduced_mass=reduced_mass,
    reduced_friction_factor=reduced_friction_factor,
)

# Plot the Coulomb Gatti 2008 kernel
fig, ax = plt.subplots()
c = ax.pcolormesh(
    radius_bins,
    radius_bins,
    np.log10(dimensional_kernel_gatti),
    shading="auto",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius r_i (m)")
ax.set_ylabel("Particle radius r_j (m)")
ax.set_title("Coulomb Gatti 2008 Coagulation Kernel")
fig.colorbar(c, ax=ax, label="Kernel (m³/s)")
plt.show()
```


    
![png](output_14_0.png)
    


## Coulomb Gopalakrishnan 2012 Kernel Strategy
The **Coulomb Gopalakrishnan 2012** kernel strategy accounts for Coulomb-influenced collisions in aerosols and dusty plasmas, as described by Gopalakrishnan and Hogan (2012).
**References:**
- Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions in aerosols and dusty plasmas. *Physical Review E*, 85(2). [Link](https://doi.org/10.1103/PhysRevE.85.026410)


```python
# Instantiate the Coulomb Gopalakrishnan 2012 kernel strategy
kernel_strategy_gopalakrishnan = (
    par.dynamics.CoulombGopalakrishnan2012KernelStrategy()
)

# Compute the dimensionless kernel
dimensionless_kernel_gopalakrishnan = (
    kernel_strategy_gopalakrishnan.dimensionless(
        diffusive_knudsen=diffusive_knudsen_values,
        coulomb_potential_ratio=coulomb_potential_ratio,
    )
)

# Compute the dimensional kernel
dimensional_kernel_gopalakrishnan = kernel_strategy_gopalakrishnan.kernel(
    dimensionless_kernel=dimensionless_kernel_gopalakrishnan,
    coulomb_potential_ratio=coulomb_potential_ratio,
    sum_of_radii=sum_of_radii,
    reduced_mass=reduced_mass,
    reduced_friction_factor=reduced_friction_factor,
)

# Plot the Coulomb Gopalakrishnan 2012 kernel
fig, ax = plt.subplots()
c = ax.pcolormesh(
    radius_bins,
    radius_bins,
    np.log10(dimensional_kernel_gopalakrishnan),
    shading="auto",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius r_i (m)")
ax.set_ylabel("Particle radius r_j (m)")
ax.set_title("Coulomb Gopalakrishnan 2012 Coagulation Kernel")
fig.colorbar(c, ax=ax, label="Kernel (m³/s)")
plt.show()
```


    
![png](output_16_0.png)
    


## Coulomb Chahl 2019 Kernel Strategy
The **Coulomb Chahl 2019** kernel strategy provides an approximation for high potential, near free molecular regime Coulombic collisions in aerosols and dusty plasmas, as detailed by Chahl and Gopalakrishnan (2019).
**References:**
- Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free molecular regime Coulombic collisions in aerosols and dusty plasmas. *Aerosol Science and Technology*, 53(8), 933-957.
[Link](https://doi.org/10.1080/02786826.2019.1614522)


```python
# %%

# Instantiate the Coulomb Chahl 2019 kernel strategy
kernel_strategy_chahl = par.dynamics.CoulumbChahl2019KernelStrategy()

# Compute the dimensionless kernel
dimensionless_kernel_chahl = kernel_strategy_chahl.dimensionless(
    diffusive_knudsen=diffusive_knudsen_values,
    coulomb_potential_ratio=coulomb_potential_ratio,
)

# Compute the dimensional kernel
dimensional_kernel_chahl = kernel_strategy_chahl.kernel(
    dimensionless_kernel=dimensionless_kernel_chahl,
    coulomb_potential_ratio=coulomb_potential_ratio,
    sum_of_radii=sum_of_radii,
    reduced_mass=reduced_mass,
    reduced_friction_factor=reduced_friction_factor,
)

# Plot the Coulomb Chahl 2019 kernel
fig, ax = plt.subplots()
c = ax.pcolormesh(
    radius_bins,
    radius_bins,
    np.log10(dimensional_kernel_chahl),
    shading="auto",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius r_i (m)")
ax.set_ylabel("Particle radius r_j (m)")
ax.set_title("Coulomb Chahl 2019 Coagulation Kernel")
fig.colorbar(c, ax=ax, label="Kernel (m³/s)")
plt.show()
```


    
![png](output_18_0.png)
    


## Compare Different Kernel Strategies
We compare the coagulation kernels obtained from different strategies to understand the influence of the chosen kernel on the coagulation process.

We step through small to large particles.


```python
# Select a pair of particles for comparison (e.g., index 100)
index = 50

# Plot the kernel values for a selected particle size
fig, ax = plt.subplots()
ax.plot(
    radius_bins,
    dimensional_kernel_hs[index, :],
    label="Hard Sphere",
    linestyle="--",
    linewidth=5,
    alpha=0.6,
)
ax.plot(
    radius_bins,
    dimensional_kernel_dyachkov[index, :],
    label="Dyachkov 2007",
    linestyle="-.",
    linewidth=4,
    alpha=0.7,
)
ax.plot(
    radius_bins,
    dimensional_kernel_gatti[index, :],
    label="Gatti 2008",
    linestyle=":",
    linewidth=4,
    alpha=0.5,
)
ax.plot(
    radius_bins,
    dimensional_kernel_gopalakrishnan[index, :],
    label="Gopalakrishnan 2012",
    linestyle="-",
    linewidth=3,
    alpha=0.5,
)
ax.plot(
    radius_bins,
    dimensional_kernel_chahl[index, :],
    label="Chahl 2019",
    linestyle="-",
    linewidth=2,
    alpha=0.9,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius r_j (m)")
ax.set_ylabel("Kernel K(r_i, r_j) (m³/s)")
ax.set_title(
    f"Comparison of Coagulation Kernels, radius_i = {radius_bins[index]:.2e} m"
)
ax.legend()
plt.show()
```


    
![png](output_20_0.png)
    



```python
# Select a pair of particles for comparison (e.g., index 100)
index = 100

# Plot the kernel values for a selected particle size
fig, ax = plt.subplots()
ax.plot(
    radius_bins,
    dimensional_kernel_hs[index, :],
    label="Hard Sphere",
    linestyle="--",
    linewidth=5,
    alpha=0.6,
)
ax.plot(
    radius_bins,
    dimensional_kernel_dyachkov[index, :],
    label="Dyachkov 2007",
    linestyle="-.",
    linewidth=4,
    alpha=0.7,
)
ax.plot(
    radius_bins,
    dimensional_kernel_gatti[index, :],
    label="Gatti 2008",
    linestyle=":",
    linewidth=4,
    alpha=0.5,
)
ax.plot(
    radius_bins,
    dimensional_kernel_gopalakrishnan[index, :],
    label="Gopalakrishnan 2012",
    linestyle="-",
    linewidth=3,
    alpha=0.5,
)
ax.plot(
    radius_bins,
    dimensional_kernel_chahl[index, :],
    label="Chahl 2019",
    linestyle="-",
    linewidth=2,
    alpha=0.9,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius r_j (m)")
ax.set_ylabel("Kernel K(r_i, r_j) (m³/s)")
ax.set_title(
    f"Comparison of Coagulation Kernels, radius_i = {radius_bins[index]:.2e} m"
)
ax.legend()
plt.show()
```


    
![png](output_21_0.png)
    



```python
# Select a pair of particles for comparison (e.g., index 100)
index = 200

# Plot the kernel values for a selected particle size
fig, ax = plt.subplots()
ax.plot(
    radius_bins,
    dimensional_kernel_hs[index, :],
    label="Hard Sphere",
    linestyle="--",
    linewidth=5,
    alpha=0.6,
)
ax.plot(
    radius_bins,
    dimensional_kernel_dyachkov[index, :],
    label="Dyachkov 2007",
    linestyle="-.",
    linewidth=4,
    alpha=0.7,
)
ax.plot(
    radius_bins,
    dimensional_kernel_gatti[index, :],
    label="Gatti 2008",
    linestyle=":",
    linewidth=4,
    alpha=0.5,
)
ax.plot(
    radius_bins,
    dimensional_kernel_gopalakrishnan[index, :],
    label="Gopalakrishnan 2012",
    linestyle="-",
    linewidth=3,
    alpha=0.5,
)
ax.plot(
    radius_bins,
    dimensional_kernel_chahl[index, :],
    label="Chahl 2019",
    linestyle="-",
    linewidth=2,
    alpha=0.9,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius r_j (m)")
ax.set_ylabel("Kernel K(r_i, r_j) (m³/s)")
ax.set_title(
    f"Comparison of Coagulation Kernels, radius_i = {radius_bins[index]:.2e} m"
)
ax.legend()
plt.show()
```


    
![png](output_22_0.png)
    


## Simulate Coagulation Over a Time Step Using Coulomb Chahl 2019 Kernel
We use the Coulomb Chahl 2019 kernel to simulate the coagulation process over a time step. This involves calculating the gain and loss rates of particle concentrations due to coagulation and the net rate of change.


```python
# Make a number concentration distribution
number_concentration = par.particles.get_lognormal_pmf_distribution(
    x_values=radius_bins,
    mode=np.array([10e-9, 200e-9, 1000e-9]),  # m
    geometric_standard_deviation=np.array([1.4, 1.5, 1.8]),
    number_of_particles=np.array([1e12, 1e12, 1e12]),  # per m^3
)

# Calculate gain and loss rates
gain_rate = par.dynamics.get_coagulation_gain_rate_discrete(
    radius=radius_bins,
    concentration=number_concentration,
    kernel=dimensional_kernel_chahl,
)
loss_rate = par.dynamics.get_coagulation_loss_rate_discrete(
    concentration=number_concentration,
    kernel=dimensional_kernel_chahl,
)

net_rate = gain_rate - loss_rate
```



## Plot Particle Concentration and Coagulation Rates
We visualize the initial particle concentration distribution and the coagulation rates to understand how charge effects influence the coagulation process over time.
- **Particle Concentration**: Shows the number concentration of particles across different sizes.
- **Coagulation Rates**: Displays the gain, loss, and net rates of coagulation, highlighting how particles of different sizes contribute to the overall process.


```python
# Plot the number concentration
fig, ax1 = plt.subplots()
ax1.plot(radius_bins, number_concentration, color="green")
ax1.set_xscale("log")
ax1.set_xlabel("Particle radius (m)")
ax1.set_ylabel("Number concentration (1/m³)")
ax1.set_title("Particle Number Concentration vs. Radius")
ax1.grid(True, which="both", ls="--")
plt.show()

# Plot the coagulation rates
fig, ax2 = plt.subplots()
ax2.plot(radius_bins, gain_rate, label="Gain Rate")
ax2.plot(radius_bins, -loss_rate, label="Loss Rate")
ax2.plot(radius_bins, net_rate, label="Net Rate", linestyle="--")
ax2.set_xscale("log")
# ax2.set_yscale('log')
ax2.set_xlabel("Particle radius (m)")
ax2.set_ylabel("Coagulation rate (1/(m³·s))")
ax2.set_title("Coagulation Rates vs. Particle Radius")
ax2.legend()
ax2.grid(True, which="both", ls="--")
plt.show()
```


    
![png](output_26_0.png)
    



    
![png](output_26_1.png)
    



---
# Condensation_1_Bin.md

# Condensation Tutorial: Radius Bin

*Work in progress, probably split into multiple notebooks, need to find a model system to test this on*

Condensation, is the first process where this framework we have been building up is applied. Here we need to account for the gas phase, and the particle phase. Then ensure that the partial pressures of species at the surface of the particle are equal to the partial pressure in the gas.

**Core Concepts:**

- **Runnable**: An abstract base class defining the interface for aerosol transformation processes.
  - Here and aerosol object is passed to the process, and the process is expected to modify the aerosol object in place, returning the modified object.
- **MassCondensation**: A concrete class implementing the RunnableProcess interface for the condensation process. Is an implementation of a `Runnable` process that adds mass to the aerosol object based on the partial pressures of the gas phase and the particle phase. Then removes the mass from the gas phase.



```python
import numpy as np
import matplotlib.pyplot as plt

# particula imports
import particula as par
```

## Setup Aerosol

First we will repeat the aerosol object that we have been using in the previous notebooks. This object will be passed to the `Runnable` processes `MassCondensation`, and modified in place.


```python
# Glycerol gas
molar_mass_glycerol = 92.09382e-3  # kg/mol
parameters_clausius = {
    "latent_heat": 71.5 * molar_mass_glycerol,
    "latent_heat_units": "kJ/kg",
    "temperature_initial": 125.5,
    "temperature_initial_units": "degC",
    "pressure_initial": 1,
    "pressure_initial_units": "mmHg",
}
vapor_pressure_strategy = par.gas.VaporPressureFactory().get_strategy(
    "clausius_clapeyron", parameters_clausius
)

sat_concentration = vapor_pressure_strategy.saturation_concentration(
    molar_mass_glycerol, 298.15
)
print(f"Saturation concentration: {sat_concentration:.2e} kg/m^3")

sat_factor = 0.01  # 50% of saturation concentration
glycerol_gas = (
    par.gas.GasSpeciesBuilder()
    .set_molar_mass(molar_mass_glycerol, "kg/mol")
    .set_vapor_pressure_strategy(vapor_pressure_strategy)
    .set_concentration(sat_concentration * sat_factor, "kg/m^3")
    .set_name("Glycerol")
    .set_condensable(True)
    .build()
)

atmosphere = (
    par.gas.AtmosphereBuilder()
    .add_species(glycerol_gas)
    .set_temperature(25, temperature_units="degC")
    .set_pressure(1, pressure_units="atm")
    .build()
)

# Glycerol particle distribution
bins = np.logspace(-8, -5, 500)
lognormal_rep = (
    par.particles.PresetParticleRadiusBuilder()
    .set_mode(np.array([100]), "nm")
    .set_geometric_standard_deviation(np.array([1.3]))
    .set_number_concentration(np.array([1e4]), "1/cm^3")
    .set_density(1.26, "g/cm^3")
    .set_distribution_type("pmf")
    .set_radius_bins(bins, "m")
    .build()
)

aerosol = par.Aerosol(atmosphere=atmosphere, particles=lognormal_rep)

print(aerosol)
```

    Saturation concentration: 2.54e-03 kg/m^3
    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['Glycerol']
    [0]: Particle Representation:
    	Strategy: RadiiBasedMovingBin
    	Activity: ActivityIdealMass
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 7.194e-08 [kg/m^3]
    	Number Concentration: 1.000e+10 [#/m^3]


## Condensation Process (Isothermal)

In code this process is implemented as a `Runnable` process. This means that the process is expected to modify the aerosol object in place, returning the modified aerosol object. This is defined in `Particle_processes.py` as the `MassCondensation` class.

The `MassCondensation` class takes a `CondensationStrategy` object as an input. This object defines and evaluates the $dm_{i}/dt$ equation for the condensation process. More strategies can be added into the `condensation.py` file.

For now, let's just run it for a few time steps and see what happens.

Note: We have a moving bin particle representation, so we would expect all the bins to move.


```python
# define the condensation process
condensation_isothermal = par.dynamics.CondensationIsothermal(
    molar_mass=molar_mass_glycerol,
    diffusion_coefficient=2e-5,
    accommodation_coefficient=0.1,  # makes things go slower/faster
)
condensation_process = par.dynamics.MassCondensation(
    condensation_strategy=condensation_isothermal
)

# define the time array in seconds
time_step = 0.5
time_array = np.arange(0, 10, time_step)
total_mass = np.zeros_like(time_array)

# output arrays
aerosol_sim = []

rate = condensation_process.rate(aerosol)

# print(f"Inital rate: {rate[:5]}...")
# print(f"Initial rate shape: {rate.shape}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogx(
    aerosol.particles[0].get_radius() * 1e9,
    aerosol.particles[0].concentration,
    label="Initial",
)
# singe step
aerosol = condensation_process.execute(aerosol, time_step)
ax.semilogx(
    aerosol.particles[0].get_radius() * 1e9,
    aerosol.particles[0].concentration,
    label="After 1 step",
)
# second step
aerosol = condensation_process.execute(aerosol, time_step)
ax.semilogx(
    aerosol.particles[0].get_radius() * 1e9,
    aerosol.particles[0].concentration,
    label="After 2 steps",
)
# 5th step
aerosol = condensation_process.execute(aerosol, time_step)
aerosol = condensation_process.execute(aerosol, time_step)
aerosol = condensation_process.execute(aerosol, time_step)
ax.semilogx(
    aerosol.particles[0].get_radius() * 1e9,
    aerosol.particles[0].concentration,
    label="After 5 steps",
)
plt.legend()
ax.set_xlabel("Radius (nm)")
ax.set_ylabel("Concentration (1/m^3)")
plt.show()
```


    
![png](output_5_0.png)
    


## Summary

We built out the condensation process, and the equations that define the process. We also defined the inputs and outputs of the process. Next we will build out coagulation and nucleation processes, to complete the aerosol dynamics framework.


---
# Condensation_2_MassBin.md

# Condensation Tutorial: Mass Binned

**NEEDS REVISION: integration unstable**

A mass binned model is relaxes the assumption of a single composition for all particles in a given bin. Instead, it allows for a distribution of compositions within each bin. This is useful when the composition of particles is separated by masses. This does not account for the same sized particles having different compositions, but rather different sized particles having different compositions.



```python
import numpy as np
import matplotlib.pyplot as plt

import particula as par
```

## Aerosol Setup

First we'll draw from a lognormal distribution to create a set of particles. We'll will then build an aerosol object to represent the aerosol population.



```python
# Ammonium sulfate and water vapor pressure
molar_mass_ammonium_sulfate = 132.14e-3  # kg/mol
molar_mass_water = 18.015e-3  # kg/mol
parameters_vapor = {
    "vapor_pressure": 1e-20,
    "vapor_pressure_units": "Pa",
}
vapor_pressure_ammonium = par.gas.VaporPressureFactory().get_strategy(
    "constant", parameters_vapor
)
vapor_pressure_water = par.gas.VaporPressureFactory().get_strategy(
    "water_buck"
)

water_sat = vapor_pressure_water.saturation_concentration(
    molar_mass=molar_mass_water, temperature=298.15
)
water_concentration = 0.8 * water_sat

glycerol_gas = (
    par.gas.GasSpeciesBuilder()
    .set_molar_mass(
        np.array([molar_mass_water, molar_mass_ammonium_sulfate]), "kg/mol"
    )
    .set_vapor_pressure_strategy(
        [vapor_pressure_water, vapor_pressure_ammonium]
    )
    .set_concentration(np.array([water_concentration, 0.0]), "kg/m^3")
    .set_name(["H2O", "NH4HSO4"])
    .set_condensable([True, True])
    .build()
)

atmosphere = (
    par.gas.AtmosphereBuilder()
    .add_species(glycerol_gas)
    .set_temperature(25, temperature_units="degC")
    .set_pressure(1, pressure_units="atm")
    .build()
)
```

**Sample Distribution**

Next we'll sample the distribution to get a set of particles. We'll then build an aerosol object to represent the aerosol population.


```python
# sample
particles_sample = par.particles.get_lognormal_sample_distribution(
    mode=np.array([100, 1000]) * 1e-9,
    geometric_standard_deviation=np.array([1.3, 1.5]),
    number_of_particles=np.array([1e3, 1e2]),
    number_of_samples=1000,
)

# histogram lognormal
bins_lognormal = np.logspace(-8, -4, 100)
bins, edges = np.histogram(particles_sample, bins=bins_lognormal, density=True)
# plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(edges[:-1], bins, width=np.diff(edges), align="edge")
ax.set_xscale("log")
ax.set_xlabel("Diameter (m)")
ax.set_ylabel("Count")
plt.show()
```


    
![png](output_5_0.png)
    



```python
# particle radis to mass
density = 1.26e3  # kg/m^3
particle_mass = density * 4 / 3 * np.pi * particles_sample**3
mass_speciation = np.array(
    [particle_mass * 0, particle_mass]
).T  # water, ammonium sulfate
concentration = np.ones_like(particles_sample) * 1e1
densities = np.array([1000, 1.26e3])  # kg/m^3

print(mass_speciation.shape)

activity_strat = (
    par.particles.ActivityKappaParameterBuilder()
    .set_density(densities, "kg/m^3")
    .set_kappa(np.array([0.0, 0.61]))
    .set_molar_mass(
        np.array([molar_mass_water, molar_mass_ammonium_sulfate]), "kg/mol"
    )
    .set_water_index(0)
    .build()
)

surface_strat = (
    par.particles.SurfaceStrategyVolumeBuilder()
    .set_density(densities, "kg/m^3")
    .set_surface_tension(np.array([0.072, 0.092]), "N/m")
    .build()
)

paritcle_rep = (
    par.particles.ParticleMassRepresentationBuilder()
    .set_distribution_strategy(
        par.particles.SpeciatedMassMovingBinBuilder().build()
    )
    .set_surface_strategy(surface_strat)
    .set_activity_strategy(activity_strat)
    .set_density(densities, "kg/m^3")
    .set_charge(0.0)
    .set_mass(mass_speciation, "kg")
    .set_concentration(concentration, "1/cm^3")
    .build()
)

aerosol = par.Aerosol(atmosphere=atmosphere, particles=paritcle_rep)

print(aerosol)
```

    (1000, 2)
    Gas mixture at 298.15 K and 101325.0 Pa consisting of ["['H2O', 'NH4HSO4']"]
    [0]: Particle Representation:
    	Strategy: SpeciatedMassMovingBin
    	Activity: ActivityKappaParameter
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 1.003e-05 [kg/m^3]
    	Number Concentration: 1.000e+10 [#/m^3]


## Condensation Process

Using the same iso thermal condensation process as in the bulk model, we'll update the properties of the particles in the aerosol object. In this cas we will change the water saturation ratio to be 80% and simulate the condensation process.


```python
# define the condensation process
condensation_isothermal = par.dynamics.CondensationIsothermal(
    molar_mass=np.array(
        [molar_mass_water, molar_mass_ammonium_sulfate]
    ),  # kg/mol
    accommodation_coefficient=0.1,  # makes things go slower/faster
    update_gases=False,
)
condensation_process = par.dynamics.MassCondensation(
    condensation_strategy=condensation_isothermal
)

# define the time array in seconds
time_step = 11
sub_steps = 10000
time_array = np.arange(0, 5, time_step)
total_mass = np.zeros_like(time_array)

# output arrays
aerosol_sim = []


fig, ax = plt.subplots(figsize=(8, 6))
bins, edges = np.histogram(
    aerosol.particles[0].get_radius(), bins=bins_lognormal
)
ax.bar(edges[:-1], bins, width=np.diff(edges), align="edge", label="Initial")

print(aerosol)
# one step
aerosol = condensation_process.execute(aerosol, time_step, sub_steps)
bins, edges = np.histogram(
    aerosol.particles[0].get_radius(), bins=bins_lognormal
)
ax.bar(
    edges[:-1],
    bins,
    width=np.diff(edges),
    align="edge",
    label="After 1 step",
    alpha=0.8,
)

print(aerosol)
# 10 seconds
aerosol = condensation_process.execute(aerosol, time_step, sub_steps)
bins, edges = np.histogram(
    aerosol.particles[0].get_radius(), bins=bins_lognormal
)
ax.bar(
    edges[:-1],
    bins,
    width=np.diff(edges),
    align="edge",
    label="After 2 steps",
    alpha=0.7,
)
print(aerosol)

ax.set_xscale("log")
# ax.set_yscale("log")
plt.legend()
ax.set_xlabel("Radius (m)")
ax.set_ylabel("Concentration (1/m^3)")
plt.show()
```

    Gas mixture at 298.15 K and 101325.0 Pa consisting of ["['H2O', 'NH4HSO4']"]
    [0]: Particle Representation:
    	Strategy: SpeciatedMassMovingBin
    	Activity: ActivityKappaParameter
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 1.003e-05 [kg/m^3]
    	Number Concentration: 1.000e+10 [#/m^3]
    Gas mixture at 298.15 K and 101325.0 Pa consisting of ["['H2O', 'NH4HSO4']"]
    [0]: Particle Representation:
    	Strategy: SpeciatedMassMovingBin
    	Activity: ActivityKappaParameter
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 2.938e-05 [kg/m^3]
    	Number Concentration: 1.000e+10 [#/m^3]
    Gas mixture at 298.15 K and 101325.0 Pa consisting of ["['H2O', 'NH4HSO4']"]
    [0]: Particle Representation:
    	Strategy: SpeciatedMassMovingBin
    	Activity: ActivityKappaParameter
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 2.938e-05 [kg/m^3]
    	Number Concentration: 1.000e+10 [#/m^3]



    
![png](output_8_1.png)
    


## Summary

We built out the condensation process, and the equations that define the process. We also defined the inputs and outputs of the process. Next we will build out coagulation and nucleation processes, to complete the aerosol dynamics framework.


---
# Condensation_3_MassResolved.md

# Condensation Tutorial: Particle Resolved

A particle resolved model is a model that tracks the properties of individual particles or collection of particles (e.g., super droplets). This is in contrast to a bulk model, which tracks the properties of the entire aerosol population. The particle resolved model can be more computationally expensive, but can provide more detailed information about the aerosol population.

To run this type of model we will need to use a speciated distribution representation. This is so that we can track the properties of individual particles.



```python
import numpy as np
import matplotlib.pyplot as plt

import particula as par
```

## Setup Aerosol

First we'll draw from a lognormal distribution to create a set of particles. We'll will then build an aerosol object to represent the aerosol population.



```python
initial_water_vapor_activity = 1.025  # Relative humidity/100

# Ammonium sulfate and water vapor pressure
molar_mass_ammonium_sulfate = 132.14e-3  # kg/mol
molar_mass_water = 18.015e-3  # kg/mol
parameters_vapor = {
    "vapor_pressure": 1e-24,
    "vapor_pressure_units": "Pa",
}
vapor_pressure_ammonium = par.gas.VaporPressureFactory().get_strategy(
    "constant", parameters_vapor
)
vapor_pressure_water = par.gas.VaporPressureFactory().get_strategy(
    "water_buck"
)

water_sat = vapor_pressure_water.saturation_concentration(
    molar_mass=molar_mass_water, temperature=298.15
)
water_concentration = water_sat * initial_water_vapor_activity

gas_phase = (
    par.gas.GasSpeciesBuilder()
    .set_molar_mass(
        np.array([molar_mass_water, molar_mass_ammonium_sulfate]), "kg/mol"
    )
    .set_vapor_pressure_strategy(
        [vapor_pressure_water, vapor_pressure_ammonium]
    )
    .set_concentration(np.array([water_concentration, 1e-30]), "kg/m^3")
    .set_name(["H2O", "NH4HSO4"])
    .set_condensable([True, True])
    .build()
)

atmosphere = (
    par.gas.AtmosphereBuilder()
    .add_species(gas_phase)
    .set_temperature(25, temperature_units="degC")
    .set_pressure(1, pressure_units="atm")
    .build()
)
```

**Sample Distribution**

Next we'll sample the distribution to get a set of particles. We'll then build an aerosol object to represent the aerosol population.


```python
density = 1.77e3  # kg/m^3
volume_sim = 1 * par.util.get_unit_conversion("cm^3", "m^3")  # m^3
number_of_samples = 10_000

# Generate a particle distribution using a lognormal sample distribution
# This distribution has a mean particle diameter (mode) and geometric standard deviation (GSD)
particle_sample = par.particles.get_lognormal_sample_distribution(
    mode=np.array([100, 400]) * 1e-9,
    geometric_standard_deviation=np.array([1.3, 1.4]),
    number_of_particles=np.array([1, 0.5]),  # relative to each mode
    number_of_samples=number_of_samples,  # Number of samples for particle distribution
)

# Calculate the mass of each particle in the sample, assuming density of 1500 kg/m^3
particle_mass = (
    4 / 3 * np.pi * particle_sample**3 * density
)  # Particle mass in kg

mass_speciation = np.column_stack(
    [particle_mass * 0, particle_mass]
)  # water, ammonium sulfate
densities = np.array([1000, density])  # kg/m^3

# kappa activity
activity_strat = (
    par.particles.ActivityKappaParameterBuilder()
    .set_density(densities, "kg/m^3")
    .set_kappa(np.array([0.0, 0.61]))
    .set_molar_mass(
        np.array([molar_mass_water, molar_mass_ammonium_sulfate]), "kg/mol"
    )
    .set_water_index(0)
    .build()
)

surface_strat = (
    par.particles.SurfaceStrategyVolumeBuilder()
    .set_density(densities, "kg/m^3")
    .set_surface_tension(np.array([0.072, 0.092]), "N/m")
    .build()
)


# Build a resolved mass representation for each particle
# This defines how particle mass, activity, and surface are represented
resolved_masses = (
    par.particles.ResolvedParticleMassRepresentationBuilder()
    .set_distribution_strategy(
        par.particles.ParticleResolvedSpeciatedMass()
    )  # Use speciated mass distribution
    .set_activity_strategy(
        activity_strat
    )  # Define activity based on ideal mass
    .set_surface_strategy(
        surface_strat
    )  # Define surface area based on particle volume
    .set_mass(mass_speciation, "kg")  # Assign mass of particles (in kg)
    .set_density(densities, "kg/m^3")  # Set particle density to 1500 kg/m^3
    .set_charge(0)  # Assume neutral particles with no charge
    .set_volume(volume_sim, "m^3")  # Set volume of particle distribution
    .build()  # Finalize the resolved mass representation
)

# Create an aerosol object with the defined atmosphere and resolved particles
aerosol_resolved = par.Aerosol(
    atmosphere=atmosphere, particles=resolved_masses
)

# Print the properties of the atmosphere
print(aerosol_resolved)
```

    Gas mixture at 298.15 K and 101325.0 Pa consisting of ["['H2O', 'NH4HSO4']"]
    [0]: Particle Representation:
    	Strategy: ParticleResolvedSpeciatedMass
    	Activity: ActivityKappaParameter
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 2.647e-06 [kg/m^3]
    	Number Concentration: 1.000e+10 [#/m^3]


## Condensation Process

Using the same iso thermal condensation process we now setup the particle resolved simulation. We'll track the properties of each particle as they grow.


```python
# define the condensation process
condensation_isothermal = par.dynamics.CondensationIsothermal(
    molar_mass=np.array(
        [molar_mass_water, molar_mass_ammonium_sulfate]
    ),  # kg/mol
    diffusion_coefficient=2e-5,  # m^2/s
    accommodation_coefficient=1,  # makes things go slower/faster
    update_gases=True,
)
condensation_process = par.dynamics.MassCondensation(
    condensation_strategy=condensation_isothermal
)

# Set up time and sub-steps for the coagulation process
total_time = 10
time_step = 0.01
sub_steps = 10

# bins
bins_lognormal = np.logspace(-8, -4, 200)


# output arrays
time = np.arange(0, total_time, time_step)
total_mass_resolved = np.ones_like(time, dtype=np.float64)
number_distribution_resolved = np.zeros((len(time), number_of_samples))
number_distribution_binned = np.zeros((len(time), len(bins_lognormal) - 1))
total_number_resolved = np.ones_like(time, dtype=np.float64)
water_saturation_in_time = np.ones_like(time, dtype=np.float64)

print(f"Total iterations to do: {len(time)*sub_steps}")
```

    Total iterations to do: 10000



```python
# Simulation loop
for i, t in enumerate(time):
    if i > 0:
        # Perform condensation for the resolved aerosol
        aerosol_resolved = condensation_process.execute(
            aerosol_resolved, time_step, sub_steps
        )

    total_mass_resolved[i] = aerosol_resolved.particles[
        0
    ].get_mass_concentration()
    number_distribution_resolved[i, :] = aerosol_resolved.particles[
        0
    ].get_radius(clone=True)
    number_distribution_binned[i, :], edges = np.histogram(
        number_distribution_resolved[i, :], bins=bins_lognormal
    )
    total_number_resolved[i] = np.sum(number_distribution_resolved[i, :] > 0)
    water_saturation_in_time[i] = aerosol_resolved.atmosphere.species[
        0
    ].get_saturation_ratio(temperature=298.15)[0]


number_distribution_binned = number_distribution_binned / volume_sim

print(aerosol_resolved)
```

    Gas mixture at 298.15 K and 101325.0 Pa consisting of ["['H2O', 'NH4HSO4']"]
    [0]: Particle Representation:
    	Strategy: ParticleResolvedSpeciatedMass
    	Activity: ActivityKappaParameter
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 6.029e-04 [kg/m^3]
    	Number Concentration: 1.000e+10 [#/m^3]


## Visualization

Finally we'll visualize the results of the simulation. The first plot is a histogram of the particle size distribution. The second plot is 2D distribution plot vs time. Third is our limiting varible of water vapor saturation ratio, and the mass transferred to the particles.


```python
# plot the initial and final distributions
fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    edges[:-1],
    number_distribution_binned[0, :],
    width=np.diff(edges),
    align="edge",
    label="Resolved initial",
    color="red",
    alpha=0.7,
)
plot_index = 100
ax.bar(
    edges[:-1],
    number_distribution_binned[plot_index, :],
    width=np.diff(edges),
    align="edge",
    label="CCN overshoot time: {:.1f} s".format(time[plot_index]),
    color="purple",
    alpha=0.5,
)
ax.bar(
    edges[:-1],
    number_distribution_binned[-1, :],
    width=np.diff(edges),
    align="edge",
    label="Resolved final",
    color="blue",
    alpha=0.7,
)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Number concentration (m^-3)")
ax.legend()
plt.show()
```


    
![png](output_10_0.png)
    



```python
fig, ax = plt.subplots(figsize=(8, 5))

# Swap X and Y to reverse axes
X, Y = np.meshgrid(
    time, edges[:-1]
)  # Now time is on the x-axis and edges on the y-axis

# Plot the contour with updated X and Y
log_of_number_distribution_binned = np.log10(
    number_distribution_binned,
    out=np.nan * np.ones_like(number_distribution_binned),
    where=number_distribution_binned > 0,
)
contour = ax.contourf(
    X,
    Y,
    log_of_number_distribution_binned.T,
    cmap="viridis",
    vmin=5,
)

# Add the color bar
cbar = fig.colorbar(contour)
cbar.set_label("Log10 of Number concentration (m^-3)")

ax.set_ylim([1e-8, 1e-5])  # Set limits for y-axis

# Set axis labels
ax.set_yscale("log")  # Log scale for particle radius on y-axis
ax.set_xlabel("Time (s)")
ax.set_ylabel("Particle radius (m)")
fig.tight_layout()
plt.show()
```


    
![png](output_11_0.png)
    



```python
# plot the total mass and water saturation on twin y-axis
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(time, total_mass_resolved, label="Total mass", color="blue")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Total Particle mass (kg/m^3)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(time, water_saturation_in_time, label="Water saturation", color="red")
ax2.set_ylabel("Water saturation", color="red")
ax2.tick_params(axis="y", labelcolor="red")

fig.tight_layout()
plt.show()
```


    
![png](output_12_0.png)
    


## Summary

This tutorial demonstrates how to run a particle resolved model. We performed a cloud condensation simulation and visualized the results. We can see once the aerosol particles activate there is a redistribution of water vapor to the larger particles as the smaller ones are out of equilibrium and evaporate.


---
# Distribution_Tutorial.md

# Distribution Strategy Tutorial

The representation of particle distributions is core to the simulation, but it can vary depending on what you are trying to achieve. In this tutorial, we will cover the  distribution strategies currently implemented.

The distribution strategies, define how to calculate properties derived from the particle distribution. These include particle mass, radius, and total mass. All of which can have different methods depending if the distribution is mass-based, radius-based, or speciated-mass based.

We will cover the following distribution strategies:

- `MassBasedMovingBin`
- `RadiiBasedMovingBin`
- `SpeciatedMassMovingBin`

As they are just operational strategies, they do not have any specific parameters to be set. They are just used to calculate the properties of the particles.


```python
import numpy as np
import matplotlib.pyplot as plt

import particula as par
```

## Strategy: MassBasedMovingBin

The `MassBasedMovingBin` strategy is used when the distribution is mass-based. This
means that the mass of the particles is known and the radius is calculated from the mass. The `MassBasedMovingBin` strategy calculates the radius of the particles using the following equation:

$$
r = \left(\frac{3m}{4\pi\rho}\right)^{1/3}
$$

where $r$ is the radius of the particle, $m$ is the mass of the particle, and $\rho$ is the density of the particle.



```python
mass_distribution = np.linspace(0, 10, 5)  # kg
density = 1000  # kg/m^3

radius = par.particles.MassBasedMovingBin().get_radius(
    mass_distribution, density
)
print(f"Radius of the particles: {radius} m")

total_mass = par.particles.MassBasedMovingBin().get_total_mass(
    mass_distribution,
    concentration=np.ones_like(mass_distribution),
    density=density,
)
print(f"Total mass of the particles: {total_mass} kg")
print(f"Same as the sum*concentration: {np.sum(mass_distribution)} kg")
```

    Radius of the particles: [0.         0.08419452 0.10607844 0.1214295  0.13365046] m
    Total mass of the particles: 25.0 kg
    Same as the sum*concentration: 25.0 kg


## Builder: RadiiBasedMovingBin

The `RadiiBasedMovingBin` strategy is used when the distribution is radius-based. This means that the radius of the particles is known and the mass is calculated from the radius. The `RadiiBasedMovingBin` strategy calculates the mass of the particles using the following equation:

$$
m = \frac{4\pi\rho r^3}{3}
$$

where $m$ is the mass of the particle, $r$ is the radius of the particle, and $\rho$ is the density of the particle.

The builder does nothing in this case, as we just have no parameters to set. We use the builder pattern here to keep the code consistent with the other strategies.


```python
radii_distribution = np.linspace(0, 0.1, 5)  # m
density = 1000  # kg/m^3

radii_strategy = par.particles.RadiiBasedMovingBinBuilder().build()

mass_distribution = radii_strategy.get_mass(radii_distribution, density)
print(f"Mass of the particles: {mass_distribution} kg")

total_mass = radii_strategy.get_total_mass(
    radii_distribution,
    concentration=np.ones_like(radii_distribution),
    density=density,
)
print(f"Total mass of the particles: {total_mass} kg")
```

    Mass of the particles: [0.         0.06544985 0.52359878 1.76714587 4.1887902 ] kg
    Total mass of the particles: 6.544984694978737 kg


## Factory: SpeciatedMassMovingBin

The `SpeciatedMassMovingBin` strategy is used when the distribution is speciated-mass based. This means that the mass of the particles is known and the radius is calculated from the mass. The `SpeciatedMassMovingBin` has multiple species, and the mass of each species is known for that given bin or particle.


```python
mass_distribution1 = np.linspace(0, 10, 5)  # kg
mass_distribution2 = np.linspace(0, 10, 5)  # kg
masses_combined = np.vstack((mass_distribution1, mass_distribution2)).T
density = np.array([1000.0, 2000.0])  # kg/m^3

speciated_mass = par.particles.DistributionFactory().get_strategy(
    "speciated_mass_moving_bin"
)

radius = speciated_mass.get_radius(masses_combined, density)

print(f"Radius of the particles: {radius} m")

total_mass = speciated_mass.get_total_mass(
    masses_combined,
    concentration=np.ones_like(mass_distribution1),
    density=density,
)
print(f"Total mass of the particles: {total_mass} kg")
```

    Radius of the particles: [0.         0.09637866 0.1214295  0.13900208 0.15299159] m
    Total mass of the particles: 50.0 kg


## Summary

In this tutorial, we covered the distribution strategies implemented in the simulation. We covered the `MassBasedMovingBin`, `RadiiBasedMovingBin`, and `SpeciatedMassMovingBin` strategies. These strategies are used to calculate the properties of the particles based on the distribution type.


---
# Gas_Species.md

# Gas Species Tutorial

The `GasSpecies` is a class that represents a gas species in a simulation or calculation. It includes properties such as the species' name, molar mass, vapor pressure, and whether it is condensable. The class provides methods to set and retrieve these properties, ensuring that each species is fully defined and manageable within simulations.

In this notebook, we will demonstrate how to use the `GasSpecies` class to create and manage gas species. This includes creating new species, setting their properties, and retrieving vapor pressure, concentration, and other properties.

## Key Classes

- `GasSpecies`: Represents a gas species in a simulation or calculation.
- `GasSpeciesBuilder`: A builder class that constructs instances of `GasSpecies` with well-defined properties.


```python
import numpy as np
import matplotlib.pyplot as plt

# Particula imports
import particula as par
```

## Define Vapor Pressure Strategies

In this section, we'll focus on defining vapor pressure strategies for gas species, specifically Butanol, Styrene, and Water, which were used in our previous examples. To streamline our analysis, we will group Butanol and Styrene into a single organic category, and consider Water separately.

### Strategy Assignment

For calculating vapor pressures:

- **Organics (Butanol and Styrene)**: We will utilize the Antoine equation, a widely recognized method for estimating the vapor pressure of organic compounds based on temperature.
- **Water**: We will apply the Buck equation, which is specifically tailored to accurately calculate the vapor pressure of water across a range of temperatures.


```python
# Define the coefficients for Butanol using the Antoine equation.
# 'a', 'b', and 'c' are coefficients specific to the Antoine equation used to calculate vapor pressure.
butanol_coefficients = {"a": 7.838, "b": 1558.19, "c": 196.881}
# Create a vapor pressure strategy for Butanol using the Antoine equation.
butanol_antione = par.gas.VaporPressureFactory().get_strategy(
    strategy_type="antoine", parameters=butanol_coefficients
)

# Define the coefficients for Styrene, similar to Butanol, using the
# Antoine equation.
styrene_coefficients = {"a": 6.924, "b": 1420, "c": 226}
# Create a vapor pressure strategy for Styrene using the Antoine equation.
styrene_antione = par.gas.VaporPressureFactory().get_strategy(
    strategy_type="antoine", parameters=styrene_coefficients
)

# Water uses a different model for vapor pressure calculation called the Buck equation.
# The Buck equation is particularly suited for water vapor calculations.
# No additional parameters are required to be passed for the Buck equation
# in this instance.
water_buck = par.gas.VaporPressureFactory().get_strategy(
    strategy_type="water_buck"
)
```

## Using `GasSpeciesBuilder` to Construct Gas Species

Now that we have defined the appropriate vapor pressure strategies for our gas species, we can proceed to construct the individual species using the `GasSpeciesBuilder`. This builder simplifies the process of defining and validating the properties of each gas species before their creation. We'll begin with Water, as it involves a straightforward application of the Buck equation.

### Building the Water Gas Species

The `GasSpeciesBuilder` facilitates a structured approach to setting up a gas species. To build a Water gas species, the builder requires the following properties to be set:

1. **Name**: Identifies the species, which in this case is "Water".
2. **Molar Mass**: The molar mass of water, essential for calculations involving mass and moles.
3. **Vapor Pressure Strategy**: The specific strategy used to calculate vapor pressure; for Water, we use the Buck equation.
4. **Condensability**: Indicates whether the species can condense under certain atmospheric conditions. For Water, this is typically true.
5. **Concentration**: The initial concentration of Water in the mixture, which could vary based on the scenario.

Here is how you can use the `GasSpeciesBuilder` to set up Water:


```python
# Configure the builder with the necessary properties
water_species = (
    par.gas.GasSpeciesBuilder()
    .set_name("Water")
    .set_molar_mass(18.01528, molar_mass_units="g/mol")
    .set_vapor_pressure_strategy(water_buck)
    .set_condensable(True)
    .set_concentration(1e2, concentration_units="ug/m^3")
    .build()
)


# molar mass in kg/mol, concentration in kg/m3

print(water_species)
print(
    f"Notice the units of the concentration are now in kg/m^3: {water_species.concentration}"
)
print(
    f"Also the units of the molar mass are now in kg/mol: {water_species.molar_mass}"
)
```

    Water
    Notice the units of the concentration are now in kg/m^3: 1.0000000000000001e-07
    Also the units of the molar mass are now in kg/mol: 0.01801528


### Building Gas Species for Organics

Following Water, you can apply a similar process to build gas species for Organics like Butanol and Styrene. Each will have its set of properties based on the chemical's nature and the desired simulation context.

When calling `.build()`, it checks that all required properties are set correctly, raising an error if any essential attribute is missing or improperly configured. This ensures that each `GasSpecies` instance is valid and ready usage.


```python
# Define molar masses for organic species (Butanol and Styrene) in kilograms per mole (kg/mol).
organic_molar_mass = np.array(
    [0.074121, 104.15e-3]
)  # Molar mass for Butanol and Styrene respectively.

# List of vapor pressure strategies assigned to each organic species.
organic_vapor_pressure = [
    butanol_antione,
    styrene_antione,
]  # Using Antoine's equation for both.

# Define concentrations for each organic species in the mixture, in kilograms per cubic meter (kg/m^3).
organic_concentration = np.array(
    [2e-6, 1e-9]
)  # Concentration values for Butanol and Styrene respectively.

# Names of the organic species.
organic_names = np.array(["butanol", "styrene"])

# Using GasSpeciesBuilder to construct a GasSpecies object for organics.
# Notice how we can directly use arrays to set properties for multiple species.
organic_species = (
    par.gas.GasSpeciesBuilder()
    .set_name(organic_names)
    .set_molar_mass(organic_molar_mass, "kg/mol")
    .set_vapor_pressure_strategy(organic_vapor_pressure)
    .set_condensable([True, True])
    .set_concentration(organic_concentration, "kg/m^3")
    .build()
)

# The `build()` method validates all the properties are set and returns the constructed GasSpecies object(s).
# Here, organic_species will contain the built GasSpecies instances for Butanol and Styrene.
print(organic_species)
```

    ['butanol' 'styrene']


## Pure Vapor Pressures

With the gas species defined, we can now calculate the pure vapor pressures of Butanol, Styrene, and Water using the respective strategies we assigned earlier. This will help us understand the vapor pressure behavior of each species individually, which is crucial for predicting their behavior in mixtures and under varying conditions.


```python
temperature_range = np.linspace(
    273.15, 373.15, 100
)  # Temperature range from 0 to 100 degrees Celsius.

organic_pure_vapor_pressure = organic_species.get_pure_vapor_pressure(
    temperature_range
)
water_pure_vapor_pressure = water_species.get_pure_vapor_pressure(
    temperature_range
)

# Plotting the vapor pressure curves for the organic species.
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(organic_names)):
    ax.plot(
        temperature_range,
        organic_pure_vapor_pressure[i],
        label=organic_names[i],
    )
ax.plot(temperature_range, water_pure_vapor_pressure, label="Water")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Vapor Pressure (Pa)")
ax.set_yscale("log")
ax.legend()
plt.show()
```


    
![png](output_9_0.png)
    


## Saturation Ratios

Now that we have established the concentration of each gas species within the mixture, we can proceed to calculate the saturation ratio for each species. The saturation ratio is an essential parameter in determining the condensation behavior of gas species within a mixture.

- **Above 1**: A saturation ratio greater than 1 indicates that the species is supersaturated and is likely to condense.
- **Below 1**: Conversely, a saturation ratio below 1 suggests that the species will likely remain in the gas phase.

### Future Exploration

In subsequent sections of this notebook series, we will delve deeper into how these saturation ratios reach equilibrium with a liquid phase, enhancing our understanding of the phase behavior under different conditions.


```python
# Saturation ratio calculation
organic_saturation_ratio = organic_species.get_saturation_ratio(
    temperature_range
)
water_saturation_ratio = water_species.get_saturation_ratio(temperature_range)

# Plotting the saturation ratio curves for the organic species.
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(organic_names)):
    ax.plot(
        temperature_range, organic_saturation_ratio[i], label=organic_names[i]
    )
ax.plot(temperature_range, water_saturation_ratio, label="Water")
ax.set_ylim(0, 5)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Saturation Ratio")
ax.legend()
plt.show()
```


    
![png](output_11_0.png)
    


# Summary

The `GasSpecies` module, along with the `GasSpeciesBuilder`, provides a robust framework for defining and managing gas species within a mixture. By assigning specific vapor pressure strategies and other essential properties, we can accurately model the behavior of individual species and their interactions in various scenarios. This module serves as a foundational component for more advanced simulations and analyses involving gas mixtures, condensation, and phase equilibrium.

The next section is one more layer of abstraction, where we will define the `GasMixture` class to manage multiple gas species within a single mixture. This class will enable us to handle complex gas mixtures effectively and efficiently, paving the way particle to gas interactions.


---
# Particle_Representation_Tutorial.md

# Particle Representation

With the different aspects of particles laid out in the previous sections, we can now focus on how to represent them in a simulation. The representation of particles is crucial for having a unified way to handle particles in a system. This section will discuss building a particle representation that can be used in simulations and analyses.





```python
import numpy as np
import matplotlib.pyplot as plt

import particula as par
```

## Builder: Particle Representation

In this section, we will demonstrate how to create a particle distribution using specific particle properties such as radius, density, and concentration. This example will help illustrate the practical application of object-oriented programming in simulating and analyzing particle systems in scientific research. We'll utilize a builder pattern to construct an instance of a Particle class, allowing for flexible configuration of particle characteristics and behaviors.

Key Components:

- Radius and Concentration: Define the size and number of particles in nanometers and their concentration per cubic centimeter, respectively.
Density and Surface Tension: Specify the material's density and the surface tension for the particles, which are critical for calculating various physical and chemical properties.
- Builder Pattern: Use a builder pattern for creating a particle representation, which facilitates the step-by-step configuration of different strategies for distribution, activity, and surface approximations.

The code snippet below sets up a particle distribution with defined properties using multiple factory methods to specify behavior strategies for distribution, activity, and surface interactions. The use of a builder pattern enhances readability and maintainability of the code by separating the construction of a complex object from its representation.


```python
# Creating particle distribution examples
radius = np.array([100, 200, 300], dtype=np.float64)  # in nm
density = 2.5
concentration = np.array([1e2, 1e3, 1e4], dtype=np.float64)

# parameters
sufrace_tension_strategy = (
    par.particles.SurfaceStrategyMassBuilder()
    .set_surface_tension(0.072, "N/m")
    .set_density(2.5, "g/cm^3")
)

# Create a Particle instance using the RadiusParticleRepresentationBuilder
particle_rep_mass = (
    par.particles.ParticleRadiusRepresentationBuilder()
    .set_distribution_strategy(par.particles.RadiiBasedMovingBin())
    .set_activity_strategy(par.particles.ActivityIdealMass())
    .set_surface_strategy(sufrace_tension_strategy)
    .set_concentration(
        concentration=concentration, concentration_units="1/cm^3"
    )
    .set_density(density=density, density_units="g/cm^3")
    .set_radius(radius=radius, radius_units="nm")
    .set_charge(charge=0)
    .build()
)

# Accessing calculated properties
print("Mass of particles:", particle_rep_mass.get_mass())
print("Radius of particles:", particle_rep_mass.get_radius())
print(
    "Total mass of the particle distribution:",
    particle_rep_mass.get_mass_concentration(),
)
```

    Mass of particles: [1.04719755e-17 8.37758041e-17 2.82743339e-16]
    Radius of particles: [1.e-07 2.e-07 3.e-07]
    Total mass of the particle distribution: 2.912256389877738e-06


## Factory: Particle Representation Implementation

The factory pattern plays a crucial role in the flexibility and extensibility of particle property computations, such as mass, radius, and total mass, within different computational models. It allows for dynamic selection of computational strategies based on the scenario, facilitating accurate and tailored simulations of real-world conditions.

In this section, we'll demonstrate how to use a factory to construct speciated particles characterized by varied properties, enhancing our ability to simulate diverse environmental scenarios. Initially, it's beneficial to directly manipulate builders to familiarize yourself with various strategies. Subsequently, parameters can be saved in JSON format. In future iterations, these saved configurations can be rapidly deployed through the factory, streamlining particle creation and modification.



```python
# Generating random properties for speciated particles
mass_distribution = np.random.rand(500, 3).astype(np.float64)
concentration = np.random.rand(500, 1).astype(np.float64) * 1e3

# Defining surface tension parameters
surface_tension_parameter = {
    "surface_tension": 0.072,  # in N/m
    "surface_tension_units": "N/m",
    "density": 2.5,  # Density in g/cm^3
    "density_units": "g/cm^3",
}
surface_strategy = par.particles.SurfaceFactory().get_strategy(
    "mass", surface_tension_parameter
)
distribution_strategy = par.particles.DistributionFactory().get_strategy(
    "speciated_mass_moving_bin"
)
activity_strategy = par.particles.ActivityFactory().get_strategy("mass_ideal")

# Setting up parameters for the particle representation factory
parameters = {
    "distribution_strategy": distribution_strategy,
    "activity_strategy": activity_strategy,
    "surface_strategy": surface_strategy,
    "density": 2.5,
    "density_units": "g/cm^3",
    "concentration": concentration,
    "concentration_units": "1/cm^3",
    "mass": mass_distribution,
    "mass_units": "pg",  # picograms
    "charge": 0,
}

# Using the factory to create a speciated particle representation
speciated_mass_rep = (
    par.particles.ParticleRepresentationFactory().get_strategy(
        "mass", parameters
    )
)

# Outputting the total mass of the particle distribution
print(
    f"Total mass of the particle distribution: {speciated_mass_rep.get_mass_concentration()}"
)

# Plot histogram of the mass distribution and number distribution vs radius
radius = speciated_mass_rep.get_radius()
masses = speciated_mass_rep.get_mass()
concentration = speciated_mass_rep.get_concentration(clone=True)


fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(masses * 1e6, bins=20, color="blue", alpha=0.7)
ax.set_xlabel("Mass (ug)")
ax.set_ylabel("Number of Particles (Droplets)")
ax.set_title("Mass Distribution Particles")
plt.show()

fig, ax2 = plt.subplots(figsize=(8, 6))
ax2.hist(radius * 1e9, bins=20, color="red", alpha=0.7)
ax2.set_ylabel("Concentration (1/m^3)")
ax2.set_xlabel("Radius (nm)")
ax2.set_title("Number Distribution Particles")
plt.show()
```

    Total mass of the particle distribution: 0.1952955162888737



    
![png](output_5_1.png)
    



    
![png](output_5_2.png)
    


## Limited Representation Builder

The last representation, is a modification of the radii builder, where we can specify a lognomal distribution parameters. This is useful when we want a to start a simulation quick and are not trying to explicitly reproduce a specific system.


```python
lognormal_rep = (
    par.particles.PresetParticleRadiusBuilder()
    .set_mode(np.array([100, 2000]), "nm")
    .set_geometric_standard_deviation(np.array([1.5, 2.0]))
    .set_number_concentration(np.array([1e4, 1e4]), "1/cm^3")
    .set_distribution_type("pmf")
    .build()
)

# plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogx(
    lognormal_rep.get_radius(),
    lognormal_rep.get_concentration(),
    label="Number Distribution",
    color="blue",
)
ax.set_xlabel("Radius (m)")
ax.set_ylabel("Concentration (1/m^3)")
ax.set_title("Number Distribution of Particles")
plt.legend()
plt.show()
```


    
![png](output_7_0.png)
    


## Summary

In this notebook, we have discussed the importance of particle representation in simulations and analyses. We have demonstrated how to create a particle distribution using specific particle properties such as radius, density, and concentration. We have also shown how to use a builder pattern to construct an instance of a Particle class, allowing for flexible configuration of particle characteristics and behaviors. Finally, we have discussed the factory pattern and how it can be used to construct speciated particles characterized by varied properties.


---
# Particle_Surface_Tutorial.md

# Particle Surface Tutorial

Understanding how particle surfaces are represented is crucial in the study of condensation processes involving water and organic molecules. This is primarily influenced by the Kelvin curvature effect, which describes how the saturation vapor pressure of a liquid droplet varies with its size. This tutorial will introduce the fundamental approaches to modeling the particle surface in aerosol particle simulations.

[Kelvin Curvature Effect](https://en.wikipedia.org/wiki/Kelvin_equation)

## Strategies for Surface Representation

To accurately simulate particle surfaces, one must adhere to the `SurfaceStrategy` abstract base class. This class outlines common methods required for all surface representation strategies:

- `kelvin_radius`: Calculates the particle radius that corresponds to the Kelvin curvature effect.
- `kelvin_term`: Computes the Kelvin term, defined as exp(kelvin_radius / particle_radius).

Specifically, the strategies differ in how they calculate:

- `effective_surface_tension`: Determines the effective surface tension of species based on their concentration.
- `effective_density`: Computes the effective density of species based on their concentration.

While each strategy may require additional parameters, defining surface tension is essential for all. The primary strategies include:

- `SurfaceStrategyMolar`: Uses mole fraction weighted values to determine surface tension and density.
- `SurfaceStrategyMass`: Uses mass fraction weighted values to determine surface tension and density.
- `SurfaceStrategyVolume`: Uses volume fraction weighted values to determine surface tension and density.

Each strategy is interchangeable and suitable for use in aerosol particle simulations. The choice of strategy should be guided by the available data and the level of detail required for the simulation.

In this tutorial, we will demonstrate how to create and use these strategies, employing builders and factories to instantiate them and calculate both the Kelvin radius and term.


```python
import numpy as np
import matplotlib.pyplot as plt

import particula as par
```

## Direct Strategy Creation

The following code demonstrates how directly create and instance of a molar surface strategy and calculate the Kelvin radius and term.

Note this approach assumes base SI units, if you want conversions on the inputs and data validation checks then use the subsequent builder and factory methods.



```python
glycerol_molar_mass = 0.092093  # kg/mol
glycerol_density = 1261  # kg/m^3
glycerol_surface_tension = 0.063  # N/m

# Create a surface strategy
glycerol_surface_strategy = par.particles.SurfaceStrategyMolar(
    glycerol_molar_mass, glycerol_density, glycerol_surface_tension
)

glycerol_kelvin_radius = glycerol_surface_strategy.kelvin_radius(
    molar_mass=glycerol_molar_mass,
    mass_concentration=0.1,  # not this doesn't not matter for a single species mixture
    temperature=300,
)

print(f"kelvin radius of glycerol {glycerol_kelvin_radius} m")
```

    kelvin radius of glycerol 5.392780089118282e-09 m


### Kelvin term

To see the Kelvin term in action, we will calculate the Kelvin term for an array of glycerol particles. The Kelvin term is a dimensionless quantity that describes the effect of the Kelvin curvature on the saturation vapor pressure of a liquid droplet. It is defined as exp(kelvin_radius / particle_radius).

So values of the Kelvin term greater than 1 indicate that the saturation vapor pressure required to maintain the particle's size is higher than the saturation vapor pressure of the bulk liquid (flat surface). This is due to the increased in curvature of the particle surface.



```python
radii = np.logspace(-10, -6, 100)

glycerol_kevlin_term = glycerol_surface_strategy.kelvin_term(
    radius=radii,
    molar_mass=glycerol_molar_mass,
    mass_concentration=0.1,
    temperature=300,
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(radii, glycerol_kevlin_term)
ax.plot(
    [glycerol_kelvin_radius, glycerol_kelvin_radius],
    [min(glycerol_kevlin_term), max(glycerol_kevlin_term)],
    color="red",
    linestyle="--",
)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_title("Kelvin term of glycerol")
ax.set_xlabel("radius (m)")
ax.set_ylabel("Kelvin term")
ax.legend(["Kelvin term", "Kelvin radius"])
plt.show()
```


    
![png](output_5_0.png)
    


## Builder for Squalane

The following code demonstrates how to use the builder to create a surface strategy for squalane particles. The builder allows for the specification of the surface tension and density of the species, as well as the concentration of the species in the particle.

Squalane is a larger molecule with a lower surface tension than glycerol, so the Kelvin term will be lower for the same particle size.


```python
squalane_surface = (
    par.particles.SurfaceStrategyMassBuilder()
    .set_density(0.81, density_units="g/cm^3")  # call with parameter name
    .set_surface_tension(28, "mN/m")  # call without parameter name
    .build()
)

# create plot
squalane_kelvin_radius = squalane_surface.kelvin_radius(
    molar_mass=0.422, mass_concentration=0.1, temperature=300
)
squalane_kelvin_term = squalane_surface.kelvin_term(
    radius=radii, molar_mass=0.422, mass_concentration=0.1, temperature=300
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(radii, squalane_kelvin_term)
ax.plot(
    [squalane_kelvin_radius, squalane_kelvin_radius],
    [min(squalane_kelvin_term), max(squalane_kelvin_term)],
    color="red",
    linestyle="--",
)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_title("Kelvin term of squalane")
ax.set_xlabel("radius (m)")
ax.set_ylabel("Kelvin term")
ax.legend(["Kelvin term", "Kelvin radius"])
plt.show()
```


    
![png](output_7_0.png)
    


## Factory Approach

Next example is the creation using a factory method, which is more flexible and allows for the use of different units and conversions. The factory method also performs data validation checks to ensure the input values are within the expected range. As the factory is just a wrapper around the builder, the same parameters can/must be used.

Here we'll create a mixture of glycerol and squalane particles, and calculate the Kelvin term for a range of particle sizes for a volume fraction of 0.5 for each species. And assume glycerol is the condensing species.

[double check the graph output, if the mixture should be lower than the pure or not]


```python
parameters = {  # glycerol, squalane
    "density": np.array([1261, 810]),  # kg/m^3
    "density_units": "kg/m^3",
    "surface_tension": np.array([0.063, 0.028]),  # N/m
    "surface_tension_units": "N/m",
}

mixture_surface = par.particles.SurfaceFactory().get_strategy(
    strategy_type="volume",
    parameters=parameters,
)

mixture_kelvin_radius = mixture_surface.kelvin_radius(
    molar_mass=0.092093,
    mass_concentration=np.array([0.1, 0.1]),
    temperature=300,
)
print(f"kelvin radius of mixture {mixture_kelvin_radius} m")

mixture_kelvin_term = mixture_surface.kelvin_term(
    radius=radii,
    molar_mass=0.092093,
    mass_concentration=np.array([0.1, 0.1]),
    temperature=300,
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(radii, mixture_kelvin_term, label="Mixure")
ax.plot(
    [mixture_kelvin_radius, mixture_kelvin_radius],
    [min(mixture_kelvin_term), max(mixture_kelvin_term)],
    color="red",
    linestyle="--",
)
ax.plot(radii, glycerol_kevlin_term, label="Glycerol")
ax.plot(
    [glycerol_kelvin_radius, glycerol_kelvin_radius],
    [min(glycerol_kevlin_term), max(glycerol_kevlin_term)],
    color="red",
    linestyle="--",
)
ax.plot(radii, squalane_kelvin_term, label="Squalane")
ax.plot(
    [squalane_kelvin_radius, squalane_kelvin_radius],
    [min(squalane_kelvin_term), max(squalane_kelvin_term)],
    color="red",
    linestyle="--",
)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_title("Kelvin term of mixture")
ax.set_xlabel("radius (m)")
ax.set_ylabel("Kelvin term")
ax.legend()
plt.show()
```

    kelvin radius of mixture 3.1208511801897936e-09 m



    
![png](output_9_1.png)
    


# Summary

This tutorial has demonstrated the fundamental approaches to modeling particle surfaces in aerosol particle simulations. By using the `SurfaceStrategy` abstract base class, we can create and use different strategies to calculate the Kelvin radius and term. The choice of strategy should be guided by the available data and the level of detail required for the simulation.

The `SurfaceStrategyMolar`, `SurfaceStrategyMass`, and `SurfaceStrategyVolume` strategies provide flexibility in determining the effective surface tension and density of species based on their concentration. By using builders and factories, we can create surface strategies with the necessary parameters and perform data validation checks to ensure the input values are within the expected range.


---
# Vapor_Pressure.md

# Vapor Pressure Tutorial

Vapor pressure is defined as the pressure exerted by a vapor in equilibrium with its liquid or solid phase. It is a crucial measure for understanding the tendency of molecules to transition from the liquid phase to the gas phase. This property is particularly important in systems where an aerosol (gas phase + particle phase) is in equilibrium with both phases.

The vapor pressure varies with temperature, and this variation can manifest in several forms. Understanding these changes is key to predicting how substances will behave under different temperature conditions.

In this notebook, we will explore the strategies for calculating vapor pressure as implemented in the `vapor_pressure` module. These strategies are essential for accurately modeling and understanding the behavior of aerosols in equilibrium with a liquid phase.

Wikipedia: [Vapor Pressure](https://en.wikipedia.org/wiki/Vapor_pressure)

## Units
All measurements and calculations in this module adhere to the base SI units:
- Molar mass is in kilograms per mole (kg/mol).
- Concentration is in kilograms per cubic meter (kg/m^3).
- Temperature is in Kelvin (K).
- Pressure is in Pascals (Pa).


```python
import numpy as np
import matplotlib.pyplot as plt

import particula as par
```

## Strategies for Vapor Pressure Calculations

In our framework, all strategies for calculating vapor pressure are encapsulated within classes that inherit from the `VaporPressureStrategy` abstract base class. This design ensures that each strategy conforms to a standardized interface, making them interchangeable and simplifying integration with other components of our modular framework.

### Core Functions
We define two primary functions that form the backbone of our vapor pressure calculations:

- `calculate_partial_pressure`: This function computes the partial pressure of a gas given its concentration, molar mass, and temperature. It applies the ideal gas law to derive the partial pressure in Pascals (Pa).

- `calculate_concentration`: This function inversely calculates the concentration of a gas from its partial pressure, molar mass, and temperature, also using the ideal gas law.

These functions can be reused for different strategies.

### Abstract Base Class
The `VaporPressureStrategy` class serves as an abstract base class that outlines the necessary methods for vapor pressure calculations:

- `partial_pressure`: Calculates the partial pressure of a gas based on its concentration, molar mass, and temperature.

- `concentration`: Calculates the concentration of a gas based on its partial pressure, temperature, and molar mass.

- `saturation_ratio`: Computes the ratio of the current vapor pressure to the saturation vapor pressure, which indicates how "saturated" the gas is with respect to a given temperature.

- `saturation_concentration`: Determines the maximum concentration of a gas at saturation at a given temperature.

- `pure_vapor_pressure`: This abstract method must be implemented by each subclass to calculate the pure (saturation) vapor pressure of a gas at specific temperatures.

By structuring our vapor pressure strategies around this abstract base class, we maintain high flexibility and robustness in our approach. Each subclass can implement specific behaviors for different gases or conditions, while relying on a common set of tools and interfaces provided by the base class.

# Example: Antoine Equation Vapor Pressure Strategy

The Antoine equation is a widely used empirical formula for estimating the vapor pressure of a substance over a range of temperatures. It takes the form:

$$
\log_{10}(P) = A - \frac{B}{T - C}
$$

where:
- $P$ is the vapor pressure in mmHg,
- $T$ is the temperature in Kelvin,
- $A$, $B$, and $C$ are substance-specific constants.
  - These constants are typically determined experimentally and can vary for different substances.
  - The Antoine equation is often used for organic compounds and provides a good approximation of vapor pressure behavior, over a limited temperature range.

We will implement this for the following substances, using constants from [link](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1):

- n-Butanol: "Formula": "C4H10O", "A": 7.838, "B": 1558.190, "C": 196.881

- Styrene: "Formula": "C8H8", "A": 6.92409, "B": 1420, "C": 226

- Water: "Formula": "H2O", "A": 7.94917, "B": 1657.462, "C": 227.02

## Direct Strategy, Builder, and Factory Patterns

We will demonstrate the use of the direct, builder, and factory patterns to create instances of the `AntoineVaporPressure` strategy. These patterns provide different levels of abstraction and flexibility in object creation, catering to various use cases and design requirements.

- **Direct Strategy**: This involves directly creating instances of the `AntoineVaporPressure` class with the required parameters. It is straightforward but may be less flexible when dealing with complex object creation or configuration.
- **Builder Pattern**: The builder pattern separates the construction of a complex object from its representation, allowing for more flexible and readable object creation. We will use a `VaporPressureBuilder` class to construct instances of the `AntoineVaporPressure` strategy with different parameters. The parameters can be set in any order, and the builder provides a clear and intuitive way to create objects.
- **Factory Pattern**: The factory pattern provides an interface for creating objects without specifying the exact class of the object to be created. We will use a `VaporPressureFactory` class to create instances of the `AntoineVaporPressure` strategy based on the substance name. This pattern allows for dynamic object creation based on input parameters, enhancing flexibility and extensibility.


```python
# Direct instantiation of an AntoineVaporPressureStrategy for butanol.
# This approach directly sets the coefficients 'a', 'b', and 'c' specific
# to butanol for calculating its vapor pressure.
butanol_antione = par.gas.AntoineVaporPressureStrategy(
    a=7.838, b=1558.19, c=196.881
)

# Use the Builder pattern to create a vapor pressure strategy for styrene.
# The Builder pattern allows for more flexible object creation by setting properties step-by-step.
# This approach, also validates the input parameters and ensures the object is fully defined.
# Here, coefficients are set individually using setter methods provided by
# the AntoineBuilder.
styrene_coefficients = {"a": 6.924, "b": 1420, "c": 226}
styrene_antione = (
    par.gas.AntoineBuilder()
    .set_a(styrene_coefficients["a"])
    .set_b(styrene_coefficients["b"])
    .set_c(styrene_coefficients["c"])
    .build()
)

# Initialize a vapor pressure strategy for water using the factory method.
# The factory method abstracts the creation logic of the builder and can instantiate different builder strategies based on the input strategy.
# This approach ensures that object creation is centralized and consistent across the application.
# Note: The strategy name provided to the factory method is case-insensitive.
water_coefficients = {"a": 7.949017, "b": 1657.462, "c": 227.02}
water_antione = par.gas.VaporPressureFactory().get_strategy(
    strategy_type="Antoine", parameters=water_coefficients
)


# Calculate and print the vapor pressures at 300 Kelvin for each substance using the initialized strategies.
# The function 'pure_vapor_pressure' is used here, which calculates the
# vapor pressure based on the provided temperature.

print(
    f"Butanol Antoine vapor pressure at 300 K: {butanol_antione.pure_vapor_pressure(300)} Pa"
)
print(
    f"Styrene Antoine vapor pressure at 300 K: {styrene_antione.pure_vapor_pressure(300)} Pa"
)
print(
    f"Water Antoine vapor pressure at 300 K: {water_antione.pure_vapor_pressure(300)} Pa"
)
```

    Butanol Antoine vapor pressure at 300 K: 7.1170940952359955e-06 Pa
    Styrene Antoine vapor pressure at 300 K: 7.239588688753633e-11 Pa
    Water Antoine vapor pressure at 300 K: 2.305360971329159e-13 Pa


## Builder Validation

Here we call the `AntoineBuilder` pattern, with incomplete parameters, to demonstrate the error handling mechanism.


```python
# failed build due to missing parameters
try:
    styrene_fail = (
        par.gas.AntoineBuilder().set_a(styrene_coefficients["a"]).build()
    )
except ValueError as e:
    print(e)  # prints error message
```

    [ERROR|abc_builder|L135]: Required parameter(s) not set: b, c


    Required parameter(s) not set: b, c


## Temperature Variation

With the vapor pressure strategy implemented, we can now explore how the vapor pressure of these substances varies with temperature. We will plot the vapor pressure curves for n-Butanol, Styrene, and Water over a range of temperatures to observe their behavior.


```python
# create a range of temperatures from 200 to 400 Kelvin
temperatures = np.linspace(300, 500, 100)

# Calculate the vapor pressures for each substance at the range of temperatures.
butanol_vapor_pressure = butanol_antione.pure_vapor_pressure(temperatures)
styrene_vapor_pressure = styrene_antione.pure_vapor_pressure(temperatures)
water_vapor_pressure = water_antione.pure_vapor_pressure(temperatures)

# Plot the vapor pressures for each substance.
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(temperatures, butanol_vapor_pressure, label="Butanol")
ax.plot(temperatures, styrene_vapor_pressure, label="Styrene", linestyle="--")
ax.plot(temperatures, water_vapor_pressure, label="Water", linestyle="-.")
ax.set_yscale("log")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Pure Vapor Pressure (Pa)")
ax.set_title("Pure Vapor Pressure vs Temperature")
ax.legend()
plt.show()
```


    
![png](output_7_0.png)
    


## Saturation Concentration

We will also calculate the concentration of these substances at different temperatures. The saturation concentration represents the maximum amount of a substance that can be in a gas at a given temperature. By examining how the saturation concentration changes with temperature, we can gain insights into the solubility and volatility of these substances.

$$
C = \frac{P_{pure}M}{RT}
$$

where:
- $C$ is the concentration in kg/m^3,
- $P_{pure}$ is the pure vapor pressure in Pa, (also known as $P_{sat}$, or $P_{0}$),
- $M$ is the molar mass in kg/mol,
- $R$ is the ideal gas constant (8.314 J/(mol K)),
- $T$ is the temperature in Kelvin.


In the case of water, the saturation ratio can be used to determine the relative humidity of the air, as it is a key factor in weather and climate models. 

We can do this calculation from the directly from the vapor pressure strategy, as it is a common in the abstract base class. So even if we change how the vapor pressure is calculated, we can still use the same method to calculate the saturation concentration.


```python
# Define the molar mass of each substance in kg/mol
butanol_molar_mass = 74.12e-3
styrene_molar_mass = 104.15e-3
water_molar_mass = 18.015e-3

# calculate the concentration pressure vs temperature
butanol_saturation_concentration = butanol_antione.saturation_concentration(
    molar_mass=butanol_molar_mass, temperature=temperatures
)
styrene_saturation_concentration = styrene_antione.saturation_concentration(
    molar_mass=styrene_molar_mass, temperature=temperatures
)
water_saturation_concentration = water_antione.saturation_concentration(
    molar_mass=water_molar_mass, temperature=temperatures
)

# Plot the saturation concentrations for each substance.
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(temperatures, butanol_saturation_concentration, label="Butanol")
ax.plot(
    temperatures,
    styrene_saturation_concentration,
    label="Styrene",
    linestyle="--",
)
ax.plot(
    temperatures, water_saturation_concentration, label="Water", linestyle="-."
)
ax.set_yscale("log")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Saturation Concentration (kg/m^3)")
ax.set_title("Saturation Concentration vs Temperature")
ax.legend()
plt.show()
```


    
![png](output_9_0.png)
    


# Partial Pressure

The partial pressure of a gas is the pressure that the gas would exert if it occupied the entire volume alone. It is a key concept in understanding gas mixtures and the behavior of gases in equilibrium. The partial pressure of a gas is proportional to its concentration and temperature, as described by the ideal gas law.

$$
P_{partial} = \frac{C R T}{M}
$$

where:
- $P_{partial}$ is the partial pressure in Pascals (Pa),
- $C$ is the concentration of the gas in kg/m^3,
- $R$ is the ideal gas constant (8.314 J/(mol K)),
- $T$ is the temperature in Kelvin,
- $M$ is the molar mass of the gas in kg/mol.

We can use the `calculate_partial_pressure` method from the vapor pressure strategy to calculate the partial pressure of a gas given its concentration, molar mass, and temperature. This calculation is essential for understanding the behavior of gas mixtures and the distribution of gases in a system.

We will use the partial pressure at 300 K and calculate how it changes with temperature for the three substances.



```python
# saturation concentration at 300 K
butanol_300K_concentration = butanol_saturation_concentration[0]
styrene_300K_concentration = styrene_saturation_concentration[0]
water_300K_concentration = water_saturation_concentration[0]

# caculate the partial pressure of each substance at 300 K
butanol_partial_pressure = butanol_antione.partial_pressure(
    concentration=butanol_300K_concentration,
    molar_mass=butanol_molar_mass,
    temperature=temperatures,
)
styrene_partial_pressure = styrene_antione.partial_pressure(
    concentration=styrene_300K_concentration,
    molar_mass=styrene_molar_mass,
    temperature=temperatures,
)
water_partial_pressure = water_antione.partial_pressure(
    concentration=water_300K_concentration,
    molar_mass=water_molar_mass,
    temperature=temperatures,
)

# Plot the partial pressures for each substance.
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(temperatures, butanol_partial_pressure, label="Butanol")
ax.plot(
    temperatures, styrene_partial_pressure, label="Styrene", linestyle="--"
)
ax.plot(temperatures, water_partial_pressure, label="Water", linestyle="-.")
ax.set_yscale("log")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Partial Pressure (Pa)")
ax.set_title("Partial Pressure vs Temperature")
ax.legend()
plt.show()
```


    
![png](output_11_0.png)
    


## Saturation Ratio

The saturation ratio is the ratio of a gas's current vapor pressure to its saturation vapor pressure at a specific temperature. This ratio helps determine how "saturated" the gas is with respect to the substance it is in equilibrium with. A saturation ratio of 1 implies that the gas is at equilibrium with the liquid phase. Values less than 1 indicate that the gas is sub-saturated (less than equilibrium), and values greater than 1 indicate supersaturation (more than equilibrium).

$$
SR = \frac{P}{P_{sat}}
$$

where:
- $SR$ is the saturation ratio,
- $P$ is the partial pressure of the gas,
- $P_{sat}$ is the saturation vapor pressure of the gas at the given temperature.

To calculate the saturation ratio, we use the concentration of the gas and compare it to the saturation concentration. We calculate the partial pressure from the concentration and the saturation concentration, and then calculate the saturation ratio. 

We will start with the gas's initial concentration at 300K and calculate the saturation ratio at various temperatures while keeping the concentration constant.

This method simulates the behavior of a gas that is initially at equilibrium with its liquid phase at 300K. If the temperature changes but the concentration remains constant, the saturation ratio will begin at 1 and typically decrease as the temperature increases. This decrease reflects the gas moving from a state of equilibrium to a state of sub-saturation as it becomes less capable of remaining in the liquid phase at higher temperatures.


```python
# caculate the saturation ratio
butanol_saturation_ratio = butanol_antione.saturation_ratio(
    concentration=butanol_300K_concentration,
    molar_mass=butanol_molar_mass,
    temperature=temperatures,
)
styrene_saturation_ratio = styrene_antione.saturation_ratio(
    concentration=styrene_300K_concentration,
    molar_mass=styrene_molar_mass,
    temperature=temperatures,
)
water_saturation_ratio = water_antione.saturation_ratio(
    concentration=water_300K_concentration,
    molar_mass=water_molar_mass,
    temperature=temperatures,
)

# Plot the saturation ratios for each substance.
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(temperatures, butanol_saturation_ratio, label="Butanol")
ax.plot(
    temperatures, styrene_saturation_ratio, label="Styrene", linestyle="--"
)
ax.plot(temperatures, water_saturation_ratio, label="Water", linestyle="-.")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Saturation Ratio")
ax.set_title("Saturation Ratio vs Temperature")
ax.legend()
plt.show()
```


    
![png](output_13_0.png)
    


## Other Strategies

In addition to the common methods shared across all vapor pressure strategies, we have several specialized strategies that can be employed to calculate vapor pressure based on different principles:

- **Constant**: This strategy applies a fixed value for the vapor pressure, regardless of external conditions.
- **Antoine**: Utilizes the Antoine equation to determine the vapor pressure of a substance, adjusting based on temperature changes.
- **Clausius_Clapeyron**: Employs the Clausius-Clapeyron equation to estimate changes in vapor pressure in response to temperature variations.
- **Water_Buck**: Specifically designed for water, this strategy uses the Buck equation to calculate vapor pressure accurately.

We will apply these different strategies to calculate the pure vapor pressure of water and observe how the values vary with temperature.

### Consistency Across Methods
Despite using different calculation strategies, the method calls remain consistent. This uniformity allows for straightforward substitutions between methods without altering the structure of the code.


```python
# Setting a constant vapor pressure at 300 K for water
water_pure_at_300K = {"vapor_pressure": 1234.56}  # in Pascals (Pa)
water_constant_strategy = par.gas.VaporPressureFactory().get_strategy(
    strategy_type="constant", parameters=water_pure_at_300K
)

# Setting parameters for the Clausius-Clapeyron equation for water
water_clausius_clapeyron_parameters = {
    "latent_heat": 40.7e3,  # specific latent heat J/mol
    "temperature_initial": 300,  # Initial temperature in Kelvin
    "pressure_initial": 1234.56,  # Initial pressure in Pascals
}
water_clausius_clapeyron_strategy = (
    par.gas.VaporPressureFactory().get_strategy(
        strategy_type="clausius_clapeyron",
        parameters=water_clausius_clapeyron_parameters,
    )
)

# Using the Water Buck strategy, no additional parameters needed
water_buck_strategy = par.gas.VaporPressureFactory().get_strategy(
    strategy_type="water_buck"
)

# Define a range of temperatures for which to calculate vapor pressures
temperatures = range(250, 500)  # From 280 K to 320 K

# Calculate the pure vapor pressure at different temperatures using
# various strategies
water_pure_constant = [
    water_constant_strategy.pure_vapor_pressure(temp) for temp in temperatures
]
water_pure_antione = [
    water_antione.pure_vapor_pressure(temp) for temp in temperatures
]
water_pure_clausius_clapeyron = [
    water_clausius_clapeyron_strategy.pure_vapor_pressure(temp)
    for temp in temperatures
]
water_pure_buck = [
    water_buck_strategy.pure_vapor_pressure(temp) for temp in temperatures
]

# Plotting the results using Matplotlib
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(temperatures, water_pure_constant, label="Constant", linestyle="-")
ax.plot(temperatures, water_pure_antione, label="Antoine", linestyle="--")
ax.plot(
    temperatures,
    water_pure_clausius_clapeyron,
    label="Clausius-Clapeyron",
    linestyle="-.",
)
ax.plot(temperatures, water_pure_buck, label="Buck", linestyle=":")
ax.set_yscale("log")
ax.set_ylim(bottom=1e-10)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Pure Vapor Pressure (Pa)")
ax.set_title("Comparison of Water Vapor Pressure Calculations")
ax.legend(loc="lower right")
plt.show()
```


    
![png](output_15_0.png)
    


## Summary

In this notebook, we covered how the strategies for vapor pressure calculations are implemented in our system. By using an abstract base class and common core functions, we ensure that each strategy adheres to a standardized interface and can be easily integrated into our framework. We explored the Antoine equation vapor pressure strategy for n-Butanol, Styrene, and Water, examining how their vapor pressures and saturation concentrations change with temperature. We calculated the partial pressure, saturation ratio, and saturation concentration for these substances, providing insights into their behavior in gas-phase systems. Finally, we demonstrated the consistency and flexibility of our approach by applying different vapor pressure strategies to calculate the vapor pressure of water and observing how the values vary with temperature.

This modular and extensible design allows us to incorporate various vapor pressure calculation methods while maintaining a consistent interface and ensuring robustness and flexibility in our system.
