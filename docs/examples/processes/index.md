# Processes: connect a mechanism to an observable change

Choose the change you need to explain, then follow its number and material
budgets. Condensation transfers material between phases; coagulation merges
particles; nucleation and injection create a population through different
sources; dilution and walls remove it. Similar-looking spectra can arise from
different combinations.

## Growth and evaporation

Start with [the initial condensation rate](condensation/condensation_1_bin.py)
to connect activity, curvature, and vapor pressure to mass transfer. Then choose
the reservoir appropriate to your question:

- [Maintained-vapor growth](condensation/condensation_2_massbin.py): particles draw from a prescribed reservoir.
- [Finite-reservoir competition](condensation/condensation_3_massresolved.py): gas depletion couples the particles.
- [Latent-energy accounting](condensation/condensation_latent_heat.py): diagnose transferred energy without assuming temperature feedback.
- [Staggered stepping](condensation/staggered_condensation_example.py): distinguish a small conservation residual from trajectory accuracy.

Compare [activity models](../thermodynamics/activity/index.md) and
[phase equilibrium](../thermodynamics/equilibria/index.md) when the thermodynamic
assumptions, rather than the transfer algorithm, are your main uncertainty.

## Collisions and population representation

For a small first calculation, choose [PMF](coagulation/functional/coagulation_basic_1_pmf.py),
[PDF](coagulation/functional/coagulation_basic_2_pdf.py), or
[particle-resolved](coagulation/functional/coagulation_basic_4_particleresolved.py)
storage. The [compact unit comparison](coagulation/functional/coagulation_basic_3_compared.py)
shows why curves need a common measure before comparison.

For longer evolution and more visible redistribution, follow:

1. [Bin totals](coagulation/coagulation_1_pmf_pattern.py) and [radius densities](coagulation/coagulation_2_pdf_pattern.py).
2. [Individual particles](coagulation/coagulation_3_particle_resolved_pattern.py), with a material-inventory check.
3. [Matched PMF/PDF inputs](coagulation/coagulation_4_compared.py), converted to common units.
4. [Total-number evolution](coagulation/coagulation_5_time_evolution.py), with the limits of agreement made explicit.

For electrical interactions, the [builder](coagulation/charge/coagulation_with_charge_objects.py)
and [factory](coagulation/charge/coagulation_with_charge_functional.py) examples
construct the same charged-kernel model through different interfaces. They map
pair coefficients for prescribed charges; they do not evolve a charge population.

## Sources and losses

Use [nucleation](nucleation/index.md) to connect a formation law with gas debit,
particle size, and fixed capacity. Use [external particle insertion](customization/adding_particles_during_simulation.py)
when the material enters from outside the modeled gas reservoir.

The [dilution reference](dilution/cpu_dilution.py) checks uniform exponential
removal. The [wall-loss sequence](wall-loss/index.md) adds size-dependent removal,
chamber geometry, and coupled-process comparisons.

## Add research-specific behavior

[Timed CPU chemistry](customization/timed_box_chemistry.py) combines a scheduled
pulse, reaction, and dilution with analytic species and inventory checks.
[Registered tracers](customization/particle_tracers.py) track additive markers
through mergers, removal, resampling, and births.

Before coupling a custom process, state what it reads and changes and how its
inventory crosses the system boundary. Then test a controlled limit and a
specific observable—not just whether the full simulation runs.
