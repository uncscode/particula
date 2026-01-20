# Simulations

This directory hosts end‑to‑end aerosol simulation notebooks built with Particula. Each notebook walks you through a complete modeling workflow—from setup to visualization or analysis.

## Available Simulations

- **[Biomass Burning Cloud Interactions](Notebooks/Biomass_Burning_Cloud_Interactions.ipynb)**  
  End‑to‑end biomass burning aerosol‑cloud simulation using Particula.

- **[Organic Partitioning and Coagulation](Notebooks/Organic_Partitioning_and_Coagulation.ipynb)**
  End‑to‑end organic aerosol partitioning and coagulation simulation using Particula.

- **[Cough Droplets Partitioning](Notebooks/Cough_Droplets_Partitioning.ipynb)**  
  Simulates the evaporation of cough droplets in a well-mixed air environment, tracking size distribution and composition changes over time.

- **[Soot Formation in Flames](Notebooks/Soot_Formation_in_Flames.ipynb)**  
  Simulates soot formation in a cooling combustion plume, tracking particle growth and chemical speciation.

- **[Cloud Chamber Cycles](Notebooks/Cloud_Chamber_Cycles.ipynb)**  
  Multi-cycle cloud chamber simulation with 4 activation-deactivation cycles and three seed composition scenarios:
  - **Scenario A**: Ammonium Sulfate seeds (κ=0.61, high hygroscopicity)
  - **Scenario B**: Sucrose seeds (κ=0.10, lower hygroscopicity)
  - **Scenario C**: Mixed AS + sucrose population (competition for water vapor)
  
  Features helper functions for repeated cycles (`run_cycle()`, `run_multi_cycle()`), particle-resolved dilution during dry phases, and comprehensive visualizations including particle size trajectories, activated fraction vs dry diameter, water mass fraction evolution, mass accumulation analysis, and comparison overlay plots.