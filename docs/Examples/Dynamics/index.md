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

These coagulation tutorials are maintained as Jupytext percent pairs (`.py` ⇄
`.ipynb`). Edit the `.py` file, then sync and execute to refresh outputs.

- [Coagulation 1: PMF Pattern](Coagulation/Coagulation_1_PMF_Pattern.ipynb)
  – Probability mass function (discrete) approach.
- [Coagulation 2: PDF Pattern](Coagulation/Coagulation_2_PDF_Pattern.ipynb)
  – Probability density function (continuous) approach.
- [Coagulation 3: Particle Resolved](Coagulation/Coagulation_3_Particle_Resolved_Pattern.ipynb)
  – Particle-resolved approach with individual particle tracking.
- [Coagulation 4: Methods Compared](Coagulation/Coagulation_4_Compared.ipynb)
  – Side-by-side strategy comparison (PMF vs PDF) using the PMF↔PDF helper.
- [Coagulation 5: Time Evolution](Coagulation/Coagulation_5_Time_Evolution.ipynb)
  – Compare number concentration decay over time for PMF, PDF, and particle-resolved.

### Functional

These illustrate functional approaches to coagulation, comparing PMF- and PDF-based methods against particle-resolved methods. All coagulation tutorials are Jupytext-paired: edit the `.py`, run `python3 .opencode/tool/validate_notebook.py <ipynb> --sync`, then `python3 .opencode/tool/run_notebook.py <ipynb>`.

- [Coagulation 1: Probability Mass Function](Coagulation/Functional/Coagulation_Basic_1_PMF.ipynb)
- [Coagulation Tutorial: Basic 2-PDF](Coagulation/Functional/Coagulation_Basic_2_PDF.ipynb)
- [Coagulation Tutorial: Basic 3-Compared](Coagulation/Functional/Coagulation_Basic_3_compared.ipynb)

### Charge

Charged coagulation tutorials are Jupytext-paired (`.py:percent` ⇄ `.ipynb`).
Edit the `.py`, then sync and execute to refresh outputs for the site.

- [Coagulation with Charge (functional)](Coagulation/Charge/Coagulation_with_Charge_functional.ipynb)
  – Edit [the paired `.py`](Coagulation/Charge/Coagulation_with_Charge_functional.py);
    executed outputs are present in the notebook.
- [Coagulation with Charge (objects)](Coagulation/Charge/Coagulation_with_Charge_objects.ipynb)
  – Edit [the paired `.py`](Coagulation/Charge/Coagulation_with_Charge_objects.py);
    executed outputs are present in the notebook.

## Customization

- [Adding Particles During Simulation](Customization/Adding_Particles_During_Simulation.ipynb) – 
  Demonstrates customizing the simulation by injecting new particles.
