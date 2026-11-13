# Dynamics

Here we collect tutorials on the dynamic processes that can affect aerosol populations,
including condensation, coagulation, and special customizations.


## Condensation

These examples demonstrate bin-based and fully resolved approaches to
modeling condensation, plus a CPU-only latent-heat diagnostic/reference
walkthrough.

- [Condensation 1: Bins](condensation/condensation_1_bin.py)
- [Condensation 2: Masses Binned](condensation/condensation_2_massbin.py)
- [Condensation 3: Masses Resolved](condensation/condensation_3_massresolved.py)
- [Condensation: Latent Heat Bookkeeping](condensation/condensation_latent_heat.py)
  – Published CPU-only latent-heat bookkeeping walkthrough using
  `MassCondensation` with public factory/builder APIs. The reported
  energy density is diagnostic only and does not feed back into
  temperature. The executable CPU integration baseline remains
  `particula/integration_tests/condensation_latent_heat_conservation_test.py`.
- [Staggered Condensation](condensation/staggered_condensation_example.py) – Demonstrates staggered ODE stepping for improved stability and mass conservation.

## Coagulation

These coagulation tutorials are maintained as percent-cell Python sources with
saved execution results. Edit the `.py` source and refresh the registered example
using the [authoring workflow](../../../website/AUTHORING.md).

- [Coagulation 1: PMF Pattern](coagulation/coagulation_1_pmf_pattern.py)
  – Probability mass function (discrete) approach.
- [Coagulation 2: PDF Pattern](coagulation/coagulation_2_pdf_pattern.py)
  – Probability density function (continuous) approach.
- [Coagulation 3: Particle Resolved](coagulation/coagulation_3_particle_resolved_pattern.py)
  – Particle-resolved approach with individual particle tracking.
- [Coagulation 4: Methods Compared](coagulation/coagulation_4_compared.py)
  – Side-by-side strategy comparison (PMF vs PDF) using the PMF↔PDF helper.
- [Coagulation 5: Time Evolution](coagulation/coagulation_5_time_evolution.py)
  – Compare number concentration decay over time for PMF, PDF, and particle-resolved.

### Functional

These illustrate functional approaches to coagulation, comparing PMF- and PDF-based methods against particle-resolved methods. Refresh changed scientific workloads with `python scripts/docs_examples.py refresh EXAMPLE_KEY`, inspect the results, and commit the source with its saved output bundle.

- [Coagulation 1: Probability Mass Function](coagulation/functional/coagulation_basic_1_pmf.py)
- [Coagulation Tutorial: Basic 2-PDF](coagulation/functional/coagulation_basic_2_pdf.py)
- [Coagulation Tutorial: Basic 3-Compared](coagulation/functional/coagulation_basic_3_compared.py)

### Charge

Charged coagulation tutorials are Jupytext-paired (`.py:percent` ⇄ `.ipynb`).
Edit the `.py`, then sync and execute to refresh outputs for the site.

- [Coagulation with Charge (functional)](coagulation/charge/coagulation_with_charge_functional.py)
  – Edit [the paired `.py`](https://github.com/uncscode/particula/blob/main/docs/examples/processes/coagulation/charge/coagulation_with_charge_functional.py);
    executed outputs are present in the notebook.
- [Coagulation with Charge (objects)](coagulation/charge/coagulation_with_charge_objects.py)
  – Edit [the paired `.py`](https://github.com/uncscode/particula/blob/main/docs/examples/processes/coagulation/charge/coagulation_with_charge_objects.py);
    executed outputs are present in the notebook.

## Customization

- [Adding Particles During Simulation](customization/adding_particles_during_simulation.py) –
  Demonstrates customizing the simulation by injecting new particles.
