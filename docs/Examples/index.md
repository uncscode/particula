# Examples

Welcome to the Particula Examples Gallery! Here you’ll find a curated collection of notebooks and step‑by‑step tutorials designed to help you explore, learn, and extend the core functionality of Particula. Whether you’re looking for a full end‑to‑end simulation, a focused how‑to guide, or a deep dive into individual components, each card below links to a ready‑to‑run example with contextual explanations and code snippets.

Use the **End‑to‑End Simulations** to see complete workflows in action, then explore the **How‑To Guides** for targeted recipes that tackle specific processes (e.g., chamber wall losses, thermodynamic equilibria, or nucleation events). Finally, the **Tutorials** section breaks down the building blocks of Particula’s architecture—Aerosol objects, Dynamics modules, Gas Phase definitions, and Particle Phase representations—so you can customize and combine them in your own research.

Jump in by selecting any card below and follow along in your browser or local environment. Happy modeling!


## End-to-End Simulations

<div class="grid cards" markdown>

-  __[Aerosol-Cloud Interactions](Simulations/Notebooks/Biomass_Burning_Cloud_Interactions.ipynb)__

    ---

    Biomass burning aerosols that activate into cloud droplets, simulating the interactions between aerosols and clouds.

    [:octicons-arrow-right-24: Simulation](Simulations/Notebooks/Biomass_Burning_Cloud_Interactions.ipynb)

- __[Organic Partitioning and Coagulation](Simulations/Notebooks/Organic_Partitioning_and_Coagulation.ipynb)__

    ---

    Simulation of organic partitioning on to seed particles and coagulation.

    [:octicons-arrow-right-24: Simulation](Simulations/Notebooks/Organic_Partitioning_and_Coagulation.ipynb)

- __[Cough Droplets Partitioning](Simulations/Notebooks/Cough_Droplets_Partitioning.ipynb)__

    ---

    Simulates the evaporation of cough droplets, tracking size distribution and composition changes over time.

    [:octicons-arrow-right-24: Simulation](Simulations/Notebooks/Cough_Droplets_Partitioning.ipynb)

- __[Soot Formation in Flames](Simulations/Notebooks/Soot_Formation_in_Flames.ipynb)__
 
    ---

    Simulates soot formation in a cooling combustion plume, tracking particle growth and chemical speciation.

    [:octicons-arrow-right-24: Simulation](Simulations/Notebooks/Soot_Formation_in_Flames.ipynb)

</div>


## How-To Guides

<div class="grid cards" markdown>

-   __[Setup Particula](Setup_Particula/index.md)__

    ---

    How to setup python and install `Particula` via pip.

    [:octicons-arrow-right-24: Tutorial](Setup_Particula/index.md)

-   __[Chamber Wall Loss](Chamber_Wall_Loss/index.md)__

    ---

    How to simulate experiments for the loss of particles to the chamber walls,
    including the strategy-based wall loss API
    (`WallLossStrategy`, `SphericalWallLossStrategy`, and `RectangularWallLossStrategy`).

    [:octicons-arrow-right-24: Tutorial](Chamber_Wall_Loss/index.md)

-   __[Equilibria](Equilibria/index.md)__

    ---

    How to simulate aerosol thermodynamic equilibria using the Binary Activity Thermodynamic `BAT` Model. Useful for water uptake and cloud droplet activation.

    [:octicons-arrow-right-24: Tutorial](Equilibria/index.md)

-   __[Nucleation](Nucleation/index.md)__

    ---

    How to simulate aerosol nucleation by adding particles during simulations. Showing how to add a nucleation event.

    [:octicons-arrow-right-24: Tutorial](Nucleation/index.md)

</div>


## Tutorials

<div class="grid cards" markdown>

-   __[Aerosol](Aerosol/index.md)__

    ---

    Learn what goes into the [Aerosol](Aerosol/index.md) object and why it is used.

    [:octicons-arrow-right-24: Tutorial](Aerosol/index.md)

-   __[Dynamics](Dynamics/index.md)__

    ---

    [Dynamics](Dynamics/index.md) is a collection of classes that processes `Aerosol` objects, to perform coagulation, condensation, and other processes.

    [:octicons-arrow-right-24: Tutorial](Dynamics/index.md)

-   __[Gas Phase](Gas_Phase/index.md)__

    ---

    Learn how to represent the [Gas Phase](Gas_Phase/index.md), including vapor pressures and atmospheric properties.

    [:octicons-arrow-right-24: Tutorial](Gas_Phase/index.md)

-   __[Particle Phase](Particle_Phase/index.md)__

    ---

    Learn about how to represent the [Particle Phase](Particle_Phase/index.md), including different particle representations; radius bins, speciated mass bins, and particle resolved.

    [:octicons-arrow-right-24: Tutorial](Particle_Phase/index.md)

</div>