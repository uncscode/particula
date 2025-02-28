

---
# Chamber_Forward_Simulation.md

# Chamber Forward Simulation

Comprehending particle dynamics within controlled environments is fundamental for the precise interpretation of experimental measurements. An aerosol chamber forward simulation is an approach employed to analyze and predict the behavior of particles under laboratory conditions. This method enables us to construct a virtual representation of the chamber dynamics, providing a platform to systematically examine the influence of different physical and chemical processes on aerosol populations. Specifically, we focus on three key processes: chamber aerosol dilution, particle coagulation, and wall loss (deposition). Each of these plays a pivotal role in shaping the size distribution of particles:

- **Chamber Aerosol Dilution**: Dilution refers to the reduction in particle concentration due to the introduction of clean air into the chamber. This process can lead to a decrease in the overall number of particles without altering the size distribution significantly. However, it can indirectly influence the dynamics of coagulation and deposition by changing the particle concentration.
- **Particle Coagulation**: Coagulation is the process where particles collide and stick together, forming larger particles. This leads to a shift in the size distribution towards larger sizes, reducing the number of smaller particles and increasing the average size of particles in the chamber. Coagulation is particularly significant for smaller particles due to their higher Brownian motion and likelihood of interaction.
- **Wall Loss** (Deposition): Wall loss occurs when particles deposit onto the walls of the chamber, removing them from the airborne population. This process preferentially affects larger particles due to their greater settling velocity and can lead to a decrease in the overall number of particles and a subtle shift in the size distribution towards smaller sizes.

We'll be running a simulation of a chamber experiment, and turn on/off each of these processes to see how they affect the size distribution of particles. We'll also be able to see how the size distribution changes over time as the experiment progresses.

The initial `particula` imports are next.


```python
import numpy as np
from matplotlib import pyplot as plt

import particula as par
```

## Initial Distribution

In this section, we define the initial conditions and parameters for our chamber simulation. The `simple_dic_kwargs` dictionary contains all the necessary parameters to initialize our particle distribution within the chamber. Here's a breakdown of each parameter:

- mode: The median diameter of the particles.
- geometric_standard_deviation: The geometric standard deviation of the particle size distribution.
- number_in_mode: The number of particles in the mode.

We define the radius bins, logarithmically, the we can get the particle concentration in a **Probability Mass Function** (PMF) representation. Or more commonly called `dN`.


```python
# Define initial simulation parameters
mode = np.array([100e-9, 500e-9])  # Median diameter of the particles in meters
geometric_standard_deviation = np.array(
    [1.3, 1.5]
)  # Geometric standard deviation of particle size distribution
number_in_mode = (
    np.array([5e4, 5e3]) * 1e6
)  # Number of particles in each mode  1/m^3


# define the radius bins for the simulation
radius_bins = np.logspace(-8, -5, 250)


# Create particle distribution using the defined parameters

concentraiton_pmf = par.particles.get_lognormal_pmf_distribution(
    x_values=radius_bins,
    mode=mode,
    geometric_standard_deviation=geometric_standard_deviation,
    number_of_particles=number_in_mode,
)


# plot the initial particle distribution
fig, ax = plt.subplots()
ax.plot(
    radius_bins, concentraiton_pmf, label="Initial distribution", marker="."
)
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel(r"Particle Concentration (dN), $\dfrac{1}{m^{3}}$")
plt.show()
```


    
![png](output_3_0.png)
    


## Rates

With the initial concentration setup we can now get the rates of change for the distribution of particles. These come from the `dynamics` module, which contains the functions to calculate the rates of change for each process. The `dynamics` module contains the following functions:

- `dilution_rate`: Calculates the rate of change due to dilution.
- `coagulation_rate`: Calculates the rate of change due to coagulation.
- `wall_loss_rate`: Calculates the rate of change due to wall loss.


```python
# coagulation rate

mass_particle = (
    4 / 3 * np.pi * radius_bins**3 * 1000
)  # mass of the particles in kg

kernel = par.dynamics.get_brownian_kernel_via_system_state(
    particle_radius=radius_bins,
    mass_particle=mass_particle,
    temperature=293.15,
    pressure=101325,
    alpha_collision_efficiency=1,
)
coagulation_loss = par.dynamics.get_coagulation_loss_rate_discrete(
    concentration=concentraiton_pmf,
    kernel=kernel,
)
coagulation_gain = par.dynamics.get_coagulation_gain_rate_discrete(
    radius=radius_bins,
    concentration=concentraiton_pmf,
    kernel=kernel,
)
coagulation_net = coagulation_gain - coagulation_loss

# dilution rate
dilution_coefficent = par.dynamics.get_volume_dilution_coefficient(
    volume=1,  # m^3
    input_flow_rate=2 * 1e-6,  # m^3/s
)
dilution_loss = par.dynamics.get_dilution_rate(
    coefficient=dilution_coefficent,
    concentration=concentraiton_pmf,
)

# wall loss rate
chamber_wall_loss_rate = par.dynamics.get_rectangle_wall_loss_rate(
    wall_eddy_diffusivity=0.1,
    particle_radius=radius_bins,
    particle_density=1000,
    particle_concentration=concentraiton_pmf,
    temperature=293.15,
    pressure=101325,
    chamber_dimensions=(1, 1, 1),  # m
)

# plot rates
fig, ax = plt.subplots()
ax.plot(
    radius_bins,
    coagulation_net,
    label="Coagulation Net",
)
ax.plot(
    radius_bins,
    dilution_loss,
    label="Dilution Loss",
)
ax.plot(
    radius_bins,
    chamber_wall_loss_rate,
    label="Chamber Wall Loss",
)
ax.plot(
    radius_bins,
    coagulation_net + dilution_loss + chamber_wall_loss_rate,
    label="Net Rate",
    linestyle="--",
)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel(r"Rate $\dfrac{1}{m^{3} s^{1}}$")
ax.grid()
plt.legend()
plt.show()
```


    
![png](output_5_0.png)
    


## For-loop Simulation

With the an example of how to calculate the rates of change for each process, we can now simulate the chamber experiment. We'll iterate over a series of time steps and calculate the change in particle concentration due to each process. This is an iterative process where we update the particle distribution at each time step based on the rates of change calculated for dilution, coagulation, and wall loss. The rates are also updated at each time step to account for the changing particle concentration within the chamber.


```python
# time steps
time_array = np.linspace(start=0, stop=3600, num=1000)
dt = time_array[1] - time_array[0]

# create a matrix to store the particle distribution at each time step
concentration_matrix = np.zeros((len(time_array), len(radius_bins)))
coagulation_net_matrix = np.zeros((len(time_array), len(radius_bins)))
dilution_loss_matrix = np.zeros((len(time_array), len(radius_bins)))
chamber_wall_loss_rate_matrix = np.zeros((len(time_array), len(radius_bins)))

# set the initial concentration
concentration_matrix[0, :] = concentraiton_pmf

kernel = par.dynamics.get_brownian_kernel_via_system_state(
    particle_radius=radius_bins,
    mass_particle=mass_particle,
    temperature=293.15,
    pressure=101325,
    alpha_collision_efficiency=1,
)
# iterate over the time steps
for i, time in enumerate(time_array[1:], start=1):

    # calculate the coagulation rate
    coagulation_loss = par.dynamics.get_coagulation_loss_rate_discrete(
        concentration=concentration_matrix[i - 1, :],
        kernel=kernel,
    )
    coagulation_gain = par.dynamics.get_coagulation_gain_rate_discrete(
        radius=radius_bins,
        concentration=concentration_matrix[i - 1, :],
        kernel=kernel,
    )
    coagulation_net = coagulation_gain - coagulation_loss

    # calculate the dilution rate
    dilution_coefficent = par.dynamics.get_volume_dilution_coefficient(
        volume=1,  # m^3
        input_flow_rate=2 * 1e-6,  # m^3/s
    )
    dilution_loss = par.dynamics.get_dilution_rate(
        coefficient=dilution_coefficent,
        concentration=concentration_matrix[i - 1, :],
    )

    # calculate the wall loss rate
    chamber_wall_loss_rate = par.dynamics.get_rectangle_wall_loss_rate(
        wall_eddy_diffusivity=0.1,
        particle_radius=radius_bins,
        particle_density=1000,
        particle_concentration=concentration_matrix[i - 1, :],
        temperature=293.15,
        pressure=101325,
        chamber_dimensions=(1, 1, 1),  # m
    )

    # update the concentration matrix
    concentration_matrix[i, :] = (
        concentration_matrix[i - 1, :]
        + (coagulation_net + dilution_loss + chamber_wall_loss_rate) * dt
    )

    # update the rate matrices
    coagulation_net_matrix[i, :] = coagulation_net
    dilution_loss_matrix[i, :] = dilution_loss
    chamber_wall_loss_rate_matrix[i, :] = chamber_wall_loss_rate

print("Done")
```

    Done


## Visualization of Particle Size Distribution Over Time

In our chamber simulation, the output solution is a matrix representing the evolution of particle size distribution over time. Specifically, the solution is a 500x100 matrix where each row corresponds to a specific particle size (500 size bins in total), and each column represents the particle distribution at a given time point (100 time steps in total).

The semi-logarithmic plot visualizes how the particle size distribution changes over the course of the simulation. We are focusing on three specific time points to illustrate these dynamics:

- **Initial Distribution**: This is the distribution at the beginning of the simulation (t=0). It sets the baseline for how particles are initially distributed across different sizes.
- **Mid-Time Distribution**: Represents the distribution at a midpoint in time (here, at the 50th time step out of 100). This snapshot provides insight into the evolution of the distribution as particles undergo processes like coagulation, dilution, and wall loss.
- **Final Distribution**: Shows the distribution at the end of the simulation (at the 100th time step). It indicates the final state of the particle sizes after all the simulated processes have taken place over the full time course.

By comparing these three distributions, we can observe and analyze how the particle sizes have coalesced, dispersed, or shifted due to the underlying aerosol dynamics within the chamber.


```python
# Plotting the simulation results
# Adjusting the figure size for better clarity
fig, ax = plt.subplots(1, 1, figsize=[8, 6])

# plot the initial particle distribution
ax.plot(
    radius_bins,
    concentration_matrix[0, :],
    label="Initial distribution",
)
# plot the final particle distribution
ax.plot(
    radius_bins,
    concentration_matrix[-1, :],
    label="Final distribution",
)
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel(r"Particle Concentration (dN), $\dfrac{1}{m^{3}}$")
plt.legend()
plt.show()
```


    
![png](output_9_0.png)
    



```python
# plot the Initial and Final rates
fig, ax = plt.subplots()
ax.plot(
    radius_bins,
    coagulation_net_matrix[1, :],
    label="Initial Coagulation Net",
)
ax.plot(
    radius_bins,
    dilution_loss_matrix[1, :],
    label="Initial Dilution Loss",
)
ax.plot(
    radius_bins,
    chamber_wall_loss_rate_matrix[1, :],
    label="Initial Chamber Wall Loss",
)
ax.plot(
    radius_bins,
    coagulation_net_matrix[-1, :],
    label="Final Coagulation Net",
    linestyle="--",
)
ax.plot(
    radius_bins,
    dilution_loss_matrix[-1, :],
    label="Final Dilution Loss",
    linestyle="--",
)
ax.plot(
    radius_bins,
    chamber_wall_loss_rate_matrix[-1, :],
    label="Final Chamber Wall Loss",
    linestyle="--",
)
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel(r"Rate $\dfrac{1}{m^{3} s^{1}}$")
ax.grid()
plt.legend()
plt.show()
```


    
![png](output_10_0.png)
    


## Takeaways

In this notebook, we conducted a series of simulations to study the behavior of aerosol particles within a controlled chamber environment. Our objective was to understand how different processes — namely coagulation, dilution, and wall loss — individually and collectively influence the size distribution of particles over time.

Our simulations revealed several key findings:

- **Coagulation Alone**: When only coagulation was considered, the particle size distribution shifted towards larger particles as expected, since smaller particles tend to combine. However, this view was incomplete as it did not account for other loss mechanisms.
- **Importance of Wall Loss**: The inclusion of wall loss in the simulations proved to be significant. Wall loss, or deposition, especially affected the larger particles due to their higher probability of contact with the chamber walls. This process led to a noticeable reduction in the number concentration of particles, altering the peak and width of the size distribution.
- **Combined Processes**: By simulating a combination of processes, we observed a more complex and realistic representation of particle dynamics. The coagulation plus dilution scenario showed a lower overall concentration across all sizes, while adding wall loss further decreased the number concentration and altered the distribution shape, underscoring the importance of including wall loss in chamber simulations.

The comparison between the different scenarios highlighted that coagulation alone could not fully explain the experimental observations. The absence of wall loss from the simulation would lead to discrepancies when comparing with empirical data, as wall loss is a critical process in chamber dynamics.


---
# Custom_Nucleation_Single_Species.md

# Custom Nucleation: Single Species

In this *How-to Guide*, we will demonstrate how to create a custom nucleation model for a single-species aerosol system. We will use fixed nucleation rates for demonstration purposes. This approach highlights the flexibility of adding new processes to your aerosol simulation before full integration into the main codebase.

This guide is based on the **Dynamics Customization** tutorial.

**Imports**


```python
import numpy as np
import matplotlib.pyplot as plt

# particula
import particula as par
```

## Aerosol Setup

We will begin by setting up ammonium sulfate vapor alongside a few pre-existing particles. The vapor phase will include a constant vapor pressure for ammonium sulfate, and a lognormal distribution will be used to represent the initial particle population.

The pre-existing particles are also necessary as, the zero particle case is not supported in the current version of the model.


```python
# Build the sulfate gas species

# Ammonium sulfate properties
molar_mass_ammonium_sulfate = 132.14e-3  # kg/mol
density_ammonium_sulfate = 1.77e3  # kg/m^3

# Define vapor pressure parameters for ammonium sulfate
parameters_vapor = {
    "vapor_pressure": 4e-12,  # pascal
    "vapor_pressure_units": "atm",  # units
}

# Create a constant vapor pressure strategy using the VaporPressureFactory
vapor_pressure_sulfate = par.gas.VaporPressureFactory().get_strategy(
    "constant", parameters_vapor
)

# Calculate the saturation concentration at a given temperature
sulfate_saturation = vapor_pressure_sulfate.saturation_concentration(
    molar_mass=molar_mass_ammonium_sulfate,
    temperature=298.15,  # Kelvin
)

# Set initial sulfate concentration as a fraction of saturation concentration
initial_sulfate_concentration = 0.5 * sulfate_saturation  # kg/m^3

# Build the sulfate gas species using the GasSpeciesBuilder
gas_sulfate = (
    par.gas.GasSpeciesBuilder()
    .set_name("sulfate")
    .set_molar_mass(molar_mass_ammonium_sulfate, "kg/mol")
    .set_condensable(True)
    .set_vapor_pressure_strategy(vapor_pressure_sulfate)
    .set_concentration(initial_sulfate_concentration, "kg/m^3")
    .build()
)

# Build the atmosphere with the sulfate species and environmental conditions
atmosphere = (
    par.gas.AtmosphereBuilder()
    .add_species(gas_sulfate)  # Add sulfate to the atmosphere
    .set_temperature(25, temperature_units="degC")  # Set temperature to 25°C
    .set_pressure(1, pressure_units="atm")  # Set pressure to 1 atm
    .build()
)

# Generate a lognormal particle size distribution
particle_sample = par.particles.get_lognormal_sample_distribution(
    mode=np.array([400e-9]),  # Mean particle diameter of 400 nm
    geometric_standard_deviation=np.array([1.4]),  # GSD of 1.4
    number_of_particles=np.array(
        [1e4]
    ),  # Number of particles in the distribution
    number_of_samples=100,  # Number of particle samples
)

# Calculate the mass of each particle based on its size and ammonium sulfate density
particle_mass_sample = (
    4 / 3 * np.pi * particle_sample**3 * density_ammonium_sulfate  # kg
)

volume_sim = 0.1 * par.util.get_unit_conversion("cm^3", "m^3")  # m^3
# Build the resolved particle mass representation for the aerosol particles
resolved_masses = (
    par.particles.ResolvedParticleMassRepresentationBuilder()
    .set_distribution_strategy(par.particles.ParticleResolvedSpeciatedMass())
    .set_activity_strategy(par.particles.ActivityIdealMass())
    .set_surface_strategy(par.particles.SurfaceStrategyVolume())
    .set_mass(particle_mass_sample, "kg")
    .set_density(density_ammonium_sulfate, "kg/m^3")
    .set_charge(0)
    .set_volume(volume_sim, "m^3")
    .build()
)

# Create the aerosol object with the atmosphere and particles
aerosol = par.Aerosol(atmosphere=atmosphere, particles=resolved_masses)

# Print the properties of the created aerosol system
print(aerosol)
```

    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['sulfate']
    [0]: Particle Representation:
    	Strategy: ParticleResolvedSpeciatedMass
    	Activity: ActivityIdealMass
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 8.240e-07 [kg/m^3]
    	Number Concentration: 1.000e+09 [#/m^3]



```python
# Build the sulfate gas species

# Ammonium sulfate and water vapor pressure
molar_mass_ammonium_sulfate = 132.14e-3  # kg/mol
density_ammonium_sulfate = 1.77e3  # kg/m^3
parameters_vapor = {
    "vapor_pressure": 4e-12,  # pascal
    "vapor_pressure_units": "atm",
}
vapor_pressure_sulfate = par.gas.VaporPressureFactory().get_strategy(
    "constant", parameters_vapor
)

# get initial vapor concentration
sulfate_saturation = vapor_pressure_sulfate.saturation_concentration(
    molar_mass=molar_mass_ammonium_sulfate,
    temperature=298.15,
)

initial_sulfate_concentration = 0.5 * sulfate_saturation


# Create the gas species
gas_sulfate = (
    par.gas.GasSpeciesBuilder()
    .set_name("sulfate")
    .set_molar_mass(molar_mass_ammonium_sulfate, "kg/mol")
    .set_condensable(True)
    .set_vapor_pressure_strategy(vapor_pressure_sulfate)
    .set_concentration(initial_sulfate_concentration, "kg/m^3")
    .build()
)

# AtmosphereBuilder constructs the atmosphere with predefined species
atmosphere = (
    par.gas.AtmosphereBuilder()
    .add_species(gas_sulfate)  # Add the sulfate gas species to the atmosphere
    .set_temperature(25, temperature_units="degC")  # Set temperature to 25°C
    .set_pressure(1, pressure_units="atm")  # Set pressure to 1 atmosphere
    .build()  # Finalize the atmosphere object
)

# Generate a particle distribution using a lognormal sample distribution
# This distribution has a mean particle diameter (mode) and geometric standard deviation (GSD)
particle_sample = par.particles.get_lognormal_sample_distribution(
    mode=np.array([400e-9]),  # Mean particle diameter of 100 nm
    geometric_standard_deviation=np.array([1.4]),  # GSD of 1.3
    number_of_particles=np.array([1e4]),  # Total number of particles
    number_of_samples=100,  # Number of samples for particle distribution
)


# Calculate the mass of each particle in the sample
particle_mass_sample = (
    4 / 3 * np.pi * particle_sample**3 * density_ammonium_sulfate
)  # Particle mass in kg


# Build a resolved mass representation for each particle
# This defines how particle mass, activity, and surface are represented
resolved_masses = (
    par.particles.ResolvedParticleMassRepresentationBuilder()
    .set_distribution_strategy(par.particles.ParticleResolvedSpeciatedMass())
    .set_activity_strategy(par.particles.ActivityIdealMass())
    .set_surface_strategy(par.particles.SurfaceStrategyVolume())
    .set_mass(particle_mass_sample, "kg")
    .set_density(density_ammonium_sulfate, "kg/m^3")
    .set_charge(0)
    .set_volume(0.1, "cm^3")
    .build()
)

# # Create an aerosol object with the defined atmosphere and resolved particles
aerosol = par.Aerosol(atmosphere=atmosphere, particles=resolved_masses)


# Print the properties of the atmosphere
print(aerosol)
```

    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['sulfate']
    [0]: Particle Representation:
    	Strategy: ParticleResolvedSpeciatedMass
    	Activity: ActivityIdealMass
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 6.873e-07 [kg/m^3]
    	Number Concentration: 1.000e+09 [#/m^3]


## Simulation

This section performs a step in the simulation using a manual stepping method. The steps include:

1. Adding more vapors to the gas phase.
2. Calculating the new saturation ratio.
3. Calculating the nucleation rate based on the saturation difference.
4. Determining the number of new particles nucleated.
5. Determining the number of resolved particles to be added to the aerosol.
6. Creating and adding the new particles to the aerosol.
7. Performing a condensation step to account for gas-phase condensation onto existing particles.
8. Performing a coagulation step to account for particle-particle interactions.

And before we start, we also need to initialize the condensation and coagulation runnables.


```python
# setup dynamics for condensation
condensation_method = par.dynamics.CondensationIsothermal(
    molar_mass=molar_mass_ammonium_sulfate,
    accommodation_coefficient=1,
    diffusion_coefficient=2e-5,
)
condensation_runnable = par.dynamics.MassCondensation(
    condensation_strategy=condensation_method
)
# setup dynamics for coagulation
coagulation_strategy = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("particle_resolved")
    .build()
)
coagulation_runnable = par.dynamics.Coagulation(
    coagulation_strategy=coagulation_strategy
)

step_count = 0
```

You can repeatedly run the next cell to see the evolution of the aerosol system.


```python
# Initialize or increment step counter
step_count += 1
print(f"Step {step_count}")

# Define key parameters
vapor_production = (
    sulfate_saturation * 0.2
)  # Adding 20% of saturation concentration per second
base_nucleation_rate = (
    1e-8 * density_ammonium_sulfate
)  # Base nucleation rate in kg/m^3/s
mass_nucleated_particle = (
    4 / 3 * np.pi * (2e-9) ** 3 * density_ammonium_sulfate
)  # Mass of a 10 nm particle in kg
exponent_nucleation = 2  # Nucleation rate exponent (empirical)
time_step = 1  # Time step in seconds

# 1. Add more vapor to the gas phase (e.g., by external sources)
print(
    "Current sulfate concentration: ",
    aerosol.atmosphere.species[0].get_concentration(),
)
aerosol.atmosphere.species[0].add_concentration(vapor_production * time_step)
print(
    "New sulfate concentration: ",
    aerosol.atmosphere.species[0].get_concentration(),
)

# 2. Calculate the new saturation ratio for sulfate in the atmosphere
saturation_ratio = aerosol.atmosphere.species[0].get_saturation_ratio(
    temperature=298.15
)
print(f"Saturation ratio: {saturation_ratio}")

# 3. Calculate the nucleation rate based on the saturation ratio
# Ensure the saturation ratio is above 1, nucleation only occurs above saturation
saturation_difference = np.maximum(
    saturation_ratio - 1, 0
)  # No nucleation if S ≤ 1
# Calculate the nucleation rate using the exponential form (custom)
# note this is mass based, if you have a volume based nucleation rate, you need to convert it
# to mass, as the resolved particles are mass based
nucleation_rate = (
    base_nucleation_rate * (saturation_difference / 500) ** exponent_nucleation
)
print(
    f"Nucleation rate [mass concentration per sec, kg/m^3/s]: {nucleation_rate}"
)

# 4. Calculate the number of new particles nucleated
# Floor division ensures we only get whole particles
number_of_new_particles = (
    time_step * nucleation_rate // mass_nucleated_particle
)
print(f"Number of new particles nucleated: {number_of_new_particles}")

# 5. Determine the number of resolved particles to create (based on simulation volume)
single_resolved_particle = aerosol.particles[0].get_concentration().max()
number_of_new_resolved_particles = int(
    number_of_new_particles // single_resolved_particle
)
print(
    f"Number of new resolved particles to be created: {number_of_new_resolved_particles}"
)

# 6. If new particles are nucleated, proceed to add them to the aerosol
if number_of_new_resolved_particles > 0:
    # Remove nucleated mass from the gas phase to conserve mass
    aerosol.atmosphere.species[0].add_concentration(
        -number_of_new_resolved_particles * mass_nucleated_particle
    )

    # Create arrays to store the properties of the newly resolved particles
    new_resolved_particle_masses = np.full(
        number_of_new_resolved_particles, mass_nucleated_particle
    )
    new_resolved_particle_concentrations = np.ones_like(
        new_resolved_particle_masses
    )  # Concentration of 1 per particle

    # Add the new resolved particles to the aerosol
    aerosol.particles[0].add_concentration(
        added_concentration=new_resolved_particle_concentrations,
        added_distribution=new_resolved_particle_masses,
    )

# Print the total particle concentration before dynamics (for monitoring)
total_particle_concentration = aerosol.particles[0].get_total_concentration()
print(
    f"Total particle concentration before dynamics [#/m^3]: {total_particle_concentration}"
)

# 7. Perform the condensation step
condensation_runnable.execute(aerosol, time_step)

# 8. Perform the coagulation step
coagulation_runnable.execute(aerosol, time_step)

# Print the total particle concentration and mass after running the dynamics
total_particle_concentration_after_process = aerosol.particles[
    0
].get_total_concentration()
print(
    f"Total particle concentration after dynamics [#/m^3]: {total_particle_concentration_after_process}"
)

total_particle_mass_after_process = aerosol.particles[
    0
].get_mass_concentration()
print(
    f"Total particle mass after dynamics [kg/m^3]: {total_particle_mass_after_process}"
)

# Retrieve and print the total number of resolved particles simulated
total_resolved_particles_in_simulation = aerosol.particles[
    0
].concentration.sum()
print(
    f"Total resolved particles in simulation: {total_resolved_particles_in_simulation}"
)
```

    Step 1
    Current sulfate concentration:  1.0802192486690696e-11
    New sulfate concentration:  1.5123069481366975e-11
    Saturation ratio: 0.7
    Nucleation rate [mass concentration per sec, kg/m^3/s]: 0.0
    Number of new particles nucleated: 0.0
    Number of new resolved particles to be created: 0
    Total particle concentration before dynamics [#/m^3]: 999999999.9999999
    Total particle concentration after dynamics [#/m^3]: 999999999.9999999
    Total particle mass after dynamics [kg/m^3]: 6.872785667890301e-07
    Total resolved particles in simulation: 100.0


## Time Loop

Now that we see the simulation is working, we can put that into a loop and save out the distribution of particles at each time step.

We'll first reset the aerosol system to its initial state, create a output matrix, then run the previous simulation in a for loop.


```python
# Build the sulfate gas species using the GasSpeciesBuilder
gas_sulfate = (
    par.gas.GasSpeciesBuilder()
    .set_name("sulfate")
    .set_molar_mass(molar_mass_ammonium_sulfate, "kg/mol")
    .set_condensable(True)
    .set_vapor_pressure_strategy(vapor_pressure_sulfate)
    .set_concentration(initial_sulfate_concentration, "kg/m^3")
    .build()
)

# AtmosphereBuilder constructs the atmosphere with predefined species
atmosphere = (
    par.gas.AtmosphereBuilder()
    .add_species(gas_sulfate)  # Add the sulfate gas species to the atmosphere
    .set_temperature(25, temperature_units="degC")  # Set temperature to 25°C
    .set_pressure(1, pressure_units="atm")  # Set pressure to 1 atmosphere
    .build()  # Finalize the atmosphere object
)
# Build the resolved particle mass representation for the aerosol particles
resolved_masses = (
    par.particles.ResolvedParticleMassRepresentationBuilder()
    .set_distribution_strategy(par.particles.ParticleResolvedSpeciatedMass())
    .set_activity_strategy(par.particles.ActivityIdealMass())
    .set_surface_strategy(par.particles.SurfaceStrategyVolume())
    .set_mass(particle_mass_sample, "kg")
    .set_density(density_ammonium_sulfate, "kg/m^3")
    .set_charge(0)
    .set_volume(0.1, "cm^3")
    .build()
)

# Create the aerosol object with the atmosphere and particles
aerosol = par.Aerosol(atmosphere=atmosphere, particles=resolved_masses)

# Print the properties of the created aerosol system
print(aerosol)


# Set up time and sub-steps for the coagulation process
total_time = 200
time_step = 1
sub_steps = 2

# bins
bins_lognormal = np.logspace(-9, -7, 200)

# output arrays
time = np.arange(0, total_time, time_step)
total_mass_resolved = np.ones_like(time, dtype=np.float64)
number_distribution_binned = np.zeros((len(time), len(bins_lognormal) - 1))
total_number_resolved = np.ones_like(time, dtype=np.float64)
saturation_ratio_output = np.ones_like(time, dtype=np.float64)

print(f"Total iterations to do: {len(time)*sub_steps}")
```

    Gas mixture at 298.15 K and 101325.0 Pa consisting of ['sulfate']
    [0]: Particle Representation:
    	Strategy: ParticleResolvedSpeciatedMass
    	Activity: ActivityIdealMass
    	Surface: SurfaceStrategyVolume
    	Mass Concentration: 6.873e-07 [kg/m^3]
    	Number Concentration: 1.000e+09 [#/m^3]
    Total iterations to do: 400



```python
# Define key parameters
vapor_production = (
    sulfate_saturation * 0.2
)  # Adding 20% of saturation concentration per second
base_nucleation_rate = (
    1e-8 * density_ammonium_sulfate
)  # Base nucleation rate in kg/m^3/s
mass_nucleated_particle = (
    4 / 3 * np.pi * (2e-9) ** 3 * density_ammonium_sulfate
)  # Mass of a 10 nm particle in kg
exponent_nucleation = 2  # Nucleation rate exponent (empirical)


for i, t in enumerate(time):
    if i > 0:
        # 1. Add more vapor to the gas phase (e.g., by external sources)
        aerosol.atmosphere.species[0].add_concentration(
            vapor_production * time_step
        )

        # 2. Calculate the new saturation ratio for sulfate in the atmosphere
        saturation_ratio = aerosol.atmosphere.species[0].get_saturation_ratio(
            temperature=298.15
        )

        # 3. Calculate the nucleation rate based on the saturation ratio
        # Ensure the saturation ratio is above 1, nucleation only occurs above saturation
        saturation_difference = np.maximum(
            saturation_ratio - 1, 0
        )  # No nucleation if S ≤ 1
        # Calculate the nucleation rate using the exponential form (custom)
        # note this is mass based, if you have a volume based nucleation rate, you need to convert it
        # to mass, as the resolved particles are mass based
        nucleation_rate = (
            base_nucleation_rate
            * (saturation_difference / 500) ** exponent_nucleation
        )

        # 4. Calculate the number of new particles nucleated
        # Floor division ensures we only get whole particles
        number_of_new_particles = (
            time_step * nucleation_rate // mass_nucleated_particle
        )

        # 5. Determine the number of resolved particles to create (based on simulation volume)
        single_resolved_particle = (
            aerosol.particles[0].get_concentration().max()
        )
        number_of_new_resolved_particles = int(
            number_of_new_particles // single_resolved_particle
        )

        # 6. If new particles are nucleated, proceed to add them to the aerosol
        if number_of_new_resolved_particles > 0:
            # Remove nucleated mass from the gas phase to conserve mass
            aerosol.atmosphere.species[0].add_concentration(
                -number_of_new_resolved_particles * mass_nucleated_particle
            )

            # Create arrays to store the properties of the newly resolved particles
            new_resolved_particle_masses = np.full(
                number_of_new_resolved_particles, mass_nucleated_particle
            )
            new_resolved_particle_concentrations = np.ones_like(
                new_resolved_particle_masses
            )  # Concentration of 1 per particle

            # Add the new resolved particles to the aerosol
            aerosol.particles[0].add_concentration(
                added_concentration=new_resolved_particle_concentrations,
                added_distribution=new_resolved_particle_masses,
            )

        # 7. Perform the condensation step
        condensation_runnable.execute(aerosol, time_step, sub_steps)
        # 8. Perform the coagulation step
        coagulation_runnable.execute(aerosol, time_step, sub_steps)

    total_mass_resolved[i] = aerosol.particles[0].get_mass_concentration()
    number_distribution = aerosol.particles[0].get_radius(clone=True)
    number_distribution_binned[i, :], edges = np.histogram(
        number_distribution, bins=bins_lognormal
    )
    total_number_resolved[i] = np.sum(number_distribution[i] > 0)
    saturation_ratio_output[i] = aerosol.atmosphere.species[
        0
    ].get_saturation_ratio(temperature=298.15)

    if i % 20 == 0:
        # Retrieve and print the total number of resolved particles simulated
        total_resolved_particles_in_simulation = aerosol.particles[
            0
        ].concentration.sum()
        print(
            f"Index {i}: Total resolved particles in simulation: {total_resolved_particles_in_simulation}"
        )


number_distribution_binned = number_distribution_binned / volume_sim
```

    Index 0: Total resolved particles in simulation: 100.0
    Index 20: Total resolved particles in simulation: 333234.0
    Index 40: Total resolved particles in simulation: 364642.0
    Index 60: Total resolved particles in simulation: 345367.0
    Index 80: Total resolved particles in simulation: 326109.0
    Index 100: Total resolved particles in simulation: 308844.0
    Index 120: Total resolved particles in simulation: 293211.0
    Index 140: Total resolved particles in simulation: 279059.0
    Index 160: Total resolved particles in simulation: 266035.0
    Index 180: Total resolved particles in simulation: 254240.0


## Graphing

In this section, we will visualize the nucleation events over time. The initial particles will be displayed, followed by their coagulated pairs. As the simulation progresses, particle growth results from both coagulation and condensation processes.


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

ax.set_ylim([1e-9, 1e-7])  # Set limits for y-axis

# Set axis labels
ax.set_yscale("log")  # Log scale for particle radius on y-axis
ax.set_xlabel("Time (s)")
ax.set_ylabel("Particle radius (m)")
fig.tight_layout()
plt.show()
```


    
![png](output_13_0.png)
    



```python
# plot the total mass and water saturation on twin y-axis
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(time, total_mass_resolved, label="Total mass", color="blue")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Total Particle mass (kg/m^3)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(time, saturation_ratio_output, label="Satruation Ratio", color="red")
ax2.set_ylabel("Saturation Ratio", color="red")
ax2.tick_params(axis="y", labelcolor="red")

fig.tight_layout()
plt.show()
```


    
![png](output_14_0.png)
    


## Conclusion

In this guide, we demonstrated how to integrate custom nucleation processes into the aerosol simulation. This shows the flexibility of the aerosol model, allowing for the addition of new processes before they are fully integrated into the core framework.

*Note*: Custom nucleation, particularly at high rates, can significantly increase the number of particles simulated, potentially slowing down the computation. A rescaling mechanism to adjust the simulation volume and control the number of resolved particles is planned for future enhancements to address this issue.


---
# activity_part1.md

# Activity Example

This notebook demonstrates the Binary Activity Theory (BAT) model application, crucial for calculating the activity of water and organic compounds in mixtures and understanding phase separation. This model, as detailed in Gorkowski, K., Preston, T. C., & Zuend, A. (2019), provides critical insights into aerosol particle behavior, essential in environmental and climate change research.

 Reference: Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
 Relative-humidity-dependent organic aerosol thermodynamics Via an efficient
 reduced-complexity model. Atmospheric Chemistry and Physics
 https://doi.org/10.5194/acp-19-13383-2019


```python
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs

# Specific functions from the particula package for activity calculations
from particula.activity import (
    water_activity,
    phase_separation,
    species_density,
    activity_coefficients,
)
```

## Activity Calculation

Define the parameters required by the activity module to calculate the activity of water and organic compounds in a mixture, as well as phase separation. These parameters include organic mole fraction, density, molecular weight ratio [water/organic], and the density of the organic compound. Using these parameters helps in accurately modeling the behavior of aerosol particles in various environmental conditions.



```python
# Define a range of organic mole fractions for the calculation
organic_mole_fraction = np.linspace(0.001, 1, 1000)

# Define other necessary parameters
oxygen2carbon = 0.225  # Oxygen to carbon ratio
molar_mass_ratio = 18.016 / 100  # Water to organic molecular weight ratio
density = species_density.organic_density_estimate(
    18.016 / molar_mass_ratio, oxygen2carbon
)  # Estimate of organic compound density

# Calculate activity coefficients using the binary_activity function
(
    activity_water,
    activity_organic,
    mass_water,
    mass_organic,
    gamma_water,
    gamma_organic,
) = activity_coefficients.bat_activity_coefficients(
    molar_mass_ratio,
    organic_mole_fraction,
    oxygen2carbon,
    density,
    functional_group=None,
)
```

## Plotting the Activity and Phase Separation

Here we plot the activity of water and the organic compound as a function of the organic mole fraction. Visualizing these activities helps in identifying phase separation or miscibility gaps, crucial for understanding the behavior of aerosols under different environmental conditions. Phase separation is indicated by activities greater than 1.0 or non-monotonic behavior in the activity curve, as shown below.


```python
fig, ax = plt.subplots()
ax.plot(
    1 - organic_mole_fraction,
    activity_water,
    label="water",
    linestyle="dashed",
)
ax.plot(
    1 - organic_mole_fraction,
    activity_organic,
    label="organic",
)
ax.set_ylim()
ax.set_xlabel("water mole fraction")
ax.set_ylabel("activity")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(
    1 - organic_mole_fraction, gamma_water, label="water", linestyle="dashed"
)
ax.plot(
    1 - organic_mole_fraction,
    gamma_organic,
    label="organic",
)
ax.set_ylim()
ax.set_xlabel("water mole fraction")
ax.set_ylabel("activity coefficient")
ax.legend()
plt.show()
```


    
![png](output_5_0.png)
    



    
![png](output_5_1.png)
    


## $ q^\alpha $

The $q^\alpha$ parameter signifies the transition from an organic-rich phase to a water-rich phase. This transition is crucial for understanding the phase behavior of aerosol particles. It can be calculated using the `particula.activity.phase_separation` function. The plot below illustrates $q^\alpha$ based on the activity calculations performed earlier.



```python
# Finding phase separation points and calculating q_alpha
phase_sep_aw = phase_separation.find_phase_separation(
    activity_water, activity_organic
)
q_alpha = phase_separation.q_alpha(
    seperation_activity=phase_sep_aw["upper_seperation"],
    activities=activity_water,
)

# Plotting q_alpha
fig, ax = plt.subplots()
plt.plot(activity_water, q_alpha)
plt.xlabel("Water Activity")
plt.ylabel("$q^{\\alpha}$ [Organic Rich to Water Rich]")
plt.show()
```


    
![png](output_7_0.png)
    


## Water Activity Focus

In atmospheric aerosol modeling, water activity is often a more critical parameter than mole fraction. This is because water activity is typically a controllable or known variable in atmospheric conditions, unlike the exact mole fractions in a solution. To correlate water activity with the mole fraction required to achieve it, we utilize functions from the `particula.activity` module.


```python
# select the water activity desired
water_activity_desired = np.linspace(0.5, 1, 100)
oxygen2carbon = 0.25

# calculate the mass fraction of water in the alpha and beta phases
# for each water activity
alpha, beta, q_alpha = water_activity.fixed_water_activity(
    water_activity=water_activity_desired,
    molar_mass_ratio=molar_mass_ratio,
    oxygen2carbon=oxygen2carbon,
    density=density,
)

# plot the results vs water activity
fig, ax = plt.subplots()
ax.plot(
    water_activity_desired,
    alpha[2],
    label="alpha phase mass fraction water",
)
ax.plot(
    water_activity_desired,
    q_alpha,
    label="q_alpha",
)
if beta is not None:
    ax.plot(
        water_activity_desired,
        beta[2],
        label="beta phase mass fraction water",
    )
ax.set_ylim()
ax.set_xlabel("water activity (Relative Humidity/100)")
ax.set_ylabel("mass fraction of water")
plt.legend()
plt.show()
```


    
![png](output_9_0.png)
    


## Higher Oxygen to Carbon Ratios

Higher oxygen to carbon ratios in the mixture tend to inhibit phase separation. The following analysis demonstrates this effect. This observation is crucial in predicting the behavior of aerosol particles under varying chemical compositions (more or less 'aged').



```python
# select the water activity desired
water_activity_desired = np.linspace(0.5, 1, 100)
# select the oxygen to carbon ratio
oxygen2carbon = 0.6

# calculate the mass fraction of water in the alpha and beta phases
# for each water activity
alpha, beta, q_alpha = water_activity.fixed_water_activity(
    water_activity=water_activity_desired,
    molar_mass_ratio=molar_mass_ratio,
    oxygen2carbon=oxygen2carbon,
    density=density,
)

# plot the results vs water activity
fig, ax = plt.subplots()
ax.plot(
    water_activity_desired,
    alpha[2],
    label="alpha phase mass fraction water",
)
ax.plot(
    water_activity_desired,
    q_alpha,
    label="q_alpha",
)
if beta is not None:
    ax.plot(
        water_activity_desired,
        beta[2],
        label="beta phase mass fraction water",
    )
ax.set_ylim()
ax.set_xlabel("water activity (Relative Humidity/100)")
ax.set_ylabel("mass fraction of water")
plt.legend()
plt.show()
```


    
![png](output_11_0.png)
    


## Summary

This notebook demonstrates how to use the activity module for calculating the activity of water and organic compounds in a mixture and assessing phase separation. The insights gained are vital for applications in aerosol thermodynamics, cloud condensation nuclei, and cloud microphysics.

This is an implementation of the Binary Activity Theory (BAT) model
developed in Gorkowski, K., Preston, T. C., & Zuend, A. (2019).


---
# equilibria_part1.md

# Liquid Vapor Equilibrium

This notebook explores the calculation of equilibrium composition in liquid-vapor mixtures, a crucial concept in aerosol science and environmental studies. We utilize an activity coefficient model to understand how different volatile organic compounds distribute between the liquid and vapor phases. This analysis is particularly important for predicting aerosol behavior and understanding atmospheric processes.


```python
# Importing necessary libraries
import matplotlib.pyplot as plt  # For creating plots and visualizations
import numpy as np  # For numerical operations
from particula.activity import (
    species_density,
)  # For calculating species density

# For partitioning calculations in liquid-vapor equilibrium
from particula.equilibria import partitioning
```

## Setup the System

To simulate the liquid-vapor equilibrium, we define several key parameters:
- `c_star_j_dry`: Represents the volatility distribution of organic compounds in dry air, calculable from vapor pressure.
- `concentration_organic_matter`: The combined concentration of vapor and liquid organic matter in the system.
- `oxygen2carbon`: The ratio of oxygen to carbon in the organic compounds, crucial for characterizing their chemical nature.
- `molar_mass`: The molar mass of the organic compounds.

These parameters help us determine the density of organics in the system, a vital step in understanding their distribution between phases.



```python
# Defining system parameters
c_star_j_dry = [1e-6, 1e-4, 1e-1, 1e2, 1e4]  # Volatility distribution in ug/m3
# Total concentration in ug/m3
concentration_organic_matter = [1, 5, 10, 15, 10]
oxygen2carbon = np.array([0.2, 0.3, 0.5, 0.4, 0.4])  # Oxygen to carbon ratios

molar_mass = np.array([200, 200, 200, 200, 200])  # Molar mass in g/mol
water_activity_desired = np.array([0.8])  # Desired water activity
molar_mass_ratio = 18.015 / np.array(molar_mass)  # Molar mass ratio

# Calculate the density of organic compounds
density = species_density.organic_array(
    molar_mass, oxygen2carbon, hydrogen2carbon=None, nitrogen2carbon=None
)
```

## Calculate the Activity Coefficients

The next step involves calculating the activity coefficients, which are pivotal in determining how the organic compounds distribute between the liquid and vapor phases. We use the `partitioning.get_properties_for_liquid_vapor_equilibrium` function, a specialized tool that simplifies the process by returning only the essential properties: activity coefficients, mass fractions, and the two-phase *q* values for the alpha-beta equilibrium.



```python
# Calculate the properties needed for liquid-vapor partitioning
gamma_organic_ab, mass_fraction_water_ab, q_ab = (
    partitioning.get_properties_for_liquid_vapor_partitioning(
        water_activity_desired=water_activity_desired,
        molar_mass=molar_mass,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )
)

# The optimization the partition coefficients, i.e. the partitioning calculation
alpha_opt, beta_opt, system_opt, fit_result = (
    partitioning.liquid_vapor_partitioning(
        c_star_j_dry=c_star_j_dry,
        concentration_organic_matter=concentration_organic_matter,
        molar_mass=molar_mass,
        gamma_organic_ab=gamma_organic_ab,
        mass_fraction_water_ab=mass_fraction_water_ab,
        q_ab=q_ab,
        partition_coefficient_guess=None,
    )
)

print(f"mass in organic aerosol [ug/m3]: {system_opt[0]}")
print(f"mass in water [ug/3]: {system_opt[1]}")
```

    mass in organic aerosol [ug/m3]: 23.96066277089358
    mass in water [ug/3]: 1.76895852033799e+16


    C:\GitHub\particula\particula\activity\gibbs_mixing.py:85: RuntimeWarning: invalid value encountered in divide
      phi2 / organic_mole_fraction


## Activity Coefficients as a Function of Relative Humidity (f(RH))

The binary activity model's key feature is its interaction with water, particularly through relative humidity (RH). Here, we will calculate how the activity coefficients vary as a function of RH. This is done by iterating over a range of RH values and computing the corresponding activity coefficients, providing insights into how atmospheric humidity influences the equilibrium behavior of the system.


```python
# Calculating activity coefficients across a range of RH values
# Range of water activity (RH/100)
water_activity_curve = np.linspace(0.01, 0.99, 50)
total_oa_concentration = np.empty([len(water_activity_curve), 1], dtype=float)
water_concentration = np.empty([len(water_activity_curve), 1], dtype=float)

for i, water_activity in enumerate(water_activity_curve):
    # Get properties for liquid-vapor partitioning at each RH value
    gamma_organic_ab, mass_fraction_water_ab, q_ab = (
        partitioning.get_properties_for_liquid_vapor_partitioning(
            water_activity_desired=water_activity,
            molar_mass=molar_mass,
            oxygen2carbon=oxygen2carbon,
            density=density,
        )
    )

    # Optimize the partition coefficients for each RH value
    alpha_opt, beta_opt, system_opt, fit_result = (
        partitioning.liquid_vapor_partitioning(
            c_star_j_dry=c_star_j_dry,
            concentration_organic_matter=concentration_organic_matter,
            molar_mass=molar_mass,
            gamma_organic_ab=gamma_organic_ab,
            mass_fraction_water_ab=mass_fraction_water_ab,
            q_ab=q_ab,
            partition_coefficient_guess=None,
        )
    )

    # Record the total organic and water concentration
    total_oa_concentration[i] = system_opt[0]
    water_concentration[i] = system_opt[1]

print("Calculation complete")
```

    Calculation complete


## Plotting the Equilibrium Composition vs. Relative Humidity

(To be updated, current bug in q-alpha transfer)

Now that we have calculated the equilibrium composition for a range of RH values, we will visualize these results. The plot will show how the total organic aerosol concentration and the water concentration in the aerosol vary with changing RH. This visualization is crucial for understanding the dynamic behavior of aerosols in different atmospheric humidity conditions.


```python
fig, ax = plt.subplots()
ax.plot(
    water_activity_curve,
    total_oa_concentration,
    label="total organic concentration",
    color="green",
)
aw = ax.twinx()
aw.plot(
    water_activity_curve,
    water_concentration,
    label="water concentration",
    color="blue",
)
plt.legend(loc="upper left")
ax.set_xlabel("water activity (a_w is RH/100)")

handles, labels = ax.get_legend_handles_labels()
aw_handles, aw_labels = aw.get_legend_handles_labels()
handles.extend(aw_handles)
labels.extend(aw_labels)
plt.legend(handles, labels, loc="upper left")

ax.set_ylabel("organic aerosol concentration [ug/m3]")
aw.set_ylabel("aerosol water concentration [ug/m3]")
plt.show()
```


    
![png](output_9_0.png)
    


## Summary

In this notebook, we have journeyed through the process of defining a liquid-vapor equilibrium system and employing the binary activity model to calculate activity coefficients as a function of relative humidity (RH). We then used these coefficients to determine the equilibrium composition of the liquid and vapor phases. Finally, the results were visualized to demonstrate the impact of RH on aerosol behavior, which is essential for understanding atmospheric aerosol dynamics and their environmental implications.

