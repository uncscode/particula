# Phase equilibria

<a id="from-phase-properties-to-equilibrium-aerosol-mass"></a>

A prescribed humidity does not uniquely specify aerosol composition without a
thermodynamic model and an inventory of available material. These examples show
the two parts of that calculation: how water and an organic component mix, and
how a volatility distribution partitions between gas and condensed phases.

Begin with phase properties and activities, then use those properties to calculate how much organic material resides in the particle phase. An equilibrium endpoint alone gives no growth timescale. The [equilibrium derivation](../../../../theory/thermodynamics/equilibria-theory.md) gives the assumptions behind the mass balance.

<!-- STUDY_LIST_START -->
- [Follow water–organic mixing from composition to water activity](activity_part1.py)
- [How does water uptake change organic aerosol partitioning?](equilibria_part1.py)
- [How long does an evaporating droplet take to reach two-liquid equilibrium?](aiomfac_evaporating_droplet.py)
- [Separate water uptake from organic partitioning](equilibria_example.py)
- [When does drying split an alcohol–salt droplet into two liquids?](aiomfac_humidity_sweep.py)
- [Which of 1,000 drying droplets first separate into two liquids?](aiomfac_mixing_state_ramp.py)
<!-- STUDY_LIST_END -->
