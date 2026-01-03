# Dynamics

Here we collect tutorials on the dynamic processes that can affect aerosol populations, 
including condensation, coagulation, and special customizations.


## Condensation

These notebooks demonstrate bin-based and fully resolved approaches to modeling condensation.

- [Condensation 1: Bins](Condensation/Condensation_1_Bin.ipynb)
- [Condensation 2: Masses Binned](Condensation/Condensation_2_MassBin.ipynb)
- [Condensation 3: Masses Resolved](Condensation/Condensation_3_MassResolved.ipynb)
- [Staggered Condensation](Condensation/Staggered_Condensation_Example.ipynb) – Demonstrates staggered ODE stepping for improved stability and mass conservation.

## Coagulation

- [Coagulation 1: PMF Pattern](Coagulation/Coagulation_1_PMF_Pattern.ipynb) – Shows probability mass function approach.
- [Coagulation 3: Particle Resolved](Coagulation/Coagulation_3_Particle_Resolved_Pattern.ipynb) – Demonstrates a particle-resolved approach.
- [Coagulation 4: Methods Compared](Coagulation/Coagulation_4_Compared.ipynb) – Compares multiple coagulation strategies.

### Functional

These illustrate functional approaches to coagulation, comparing PMF- and PDF-based methods against particle-resolved methods.

- [Coagulation 1: Probability Mass Function](Coagulation/Functional/Coagulation_Basic_1_PMF.ipynb)
- [Coagulation Tutorial: Basic 2-PDF](Coagulation/Functional/Coagulation_Basic_2_PDF.ipynb)
- [Coagulation Tutorial: Basic 3-Compared](Coagulation/Functional/Coagulation_Basic_3_compared.ipynb)

### Charge

Here we show how to include charge effects in the coagulation kernel:

- [Coagulation Charges via functions](Coagulation/Charge/Coagulation_with_Charge_functional.ipynb)
- [Coagulation Charges via objects](Coagulation/Charge/Coagulation_with_Charge_objects.ipynb)

## Customization

- [Adding Particles During Simulation](Customization/Adding_Particles_During_Simulation.ipynb) – 
  Demonstrates customizing the simulation by injecting new particles.
