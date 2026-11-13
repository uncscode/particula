# Dynamics

Here we collect tutorials on the dynamic processes that can affect aerosol populations, 
including condensation, coagulation, and special customizations.


## Condensation

These examples demonstrate bin-based and fully resolved approaches to
modeling condensation, plus a CPU-only latent-heat diagnostic/reference
walkthrough.

- [Condensation 1: Bins](condensation/condensation_1_bin.ipynb)
- [Condensation 2: Masses Binned](condensation/condensation_2_massbin.ipynb)
- [Condensation 3: Masses Resolved](condensation/condensation_3_massresolved.ipynb)
- [Condensation: Latent Heat Bookkeeping](condensation/condensation_latent_heat.ipynb)
  – Published CPU-only latent-heat bookkeeping walkthrough using
  `MassCondensation` with public factory/builder APIs. The reported
  energy density is diagnostic only and does not feed back into
  temperature. The executable CPU integration baseline remains
  `particula/integration_tests/condensation_latent_heat_conservation_test.py`.
- [Staggered Condensation](condensation/staggered_condensation_example.ipynb) – Demonstrates staggered ODE stepping for improved stability and mass conservation.

## Coagulation

These coagulation tutorials are maintained as Jupytext percent pairs (`.py` ⇄
`.ipynb`). Edit the `.py` file, then sync and execute to refresh outputs.

- [Coagulation 1: PMF Pattern](coagulation/coagulation_1_pmf_pattern.ipynb)
  – Probability mass function (discrete) approach.
- [Coagulation 2: PDF Pattern](coagulation/coagulation_2_pdf_pattern.ipynb)
  – Probability density function (continuous) approach.
- [Coagulation 3: Particle Resolved](coagulation/coagulation_3_particle_resolved_pattern.ipynb)
  – Particle-resolved approach with individual particle tracking.
- [Coagulation 4: Methods Compared](coagulation/coagulation_4_compared.ipynb)
  – Side-by-side strategy comparison (PMF vs PDF) using the PMF↔PDF helper.
- [Coagulation 5: Time Evolution](coagulation/coagulation_5_time_evolution.ipynb)
  – Compare number concentration decay over time for PMF, PDF, and particle-resolved.

### Functional

These illustrate functional approaches to coagulation, comparing PMF- and PDF-based methods against particle-resolved methods. All coagulation tutorials are Jupytext-paired: edit the `.py`, run `python3 .opencode/tools/validate_notebook.py <ipynb> --sync`, then `python3 .opencode/tools/run_notebook.py <ipynb>`.

- [Coagulation 1: Probability Mass Function](coagulation/functional/coagulation_basic_1_pmf.ipynb)
- [Coagulation Tutorial: Basic 2-PDF](coagulation/functional/coagulation_basic_2_pdf.ipynb)
- [Coagulation Tutorial: Basic 3-Compared](coagulation/functional/coagulation_basic_3_compared.ipynb)

### Charge

Charged coagulation tutorials are Jupytext-paired (`.py:percent` ⇄ `.ipynb`).
Edit the `.py`, then sync and execute to refresh outputs for the site.

- [Coagulation with Charge (functional)](coagulation/charge/coagulation_with_charge_functional.ipynb)
  – Edit [the paired `.py`](https://github.com/uncscode/particula/blob/main/docs/examples/processes/coagulation/charge/coagulation_with_charge_functional.py);
    executed outputs are present in the notebook.
- [Coagulation with Charge (objects)](coagulation/charge/coagulation_with_charge_objects.ipynb)
  – Edit [the paired `.py`](https://github.com/uncscode/particula/blob/main/docs/examples/processes/coagulation/charge/coagulation_with_charge_objects.py);
    executed outputs are present in the notebook.

## Customization

- [Adding Particles During Simulation](customization/adding_particles_during_simulation.ipynb) – 
  Demonstrates customizing the simulation by injecting new particles.
