# A humidity ramp across 1,000 individually resolved droplets

All 1,000 droplets began with one liquid and developed a second liquid as humidity fell. At six consecutive observations the counts of newly split droplets were 14, 191, 324, 254, 192 and 25. Their first two-liquid observations spanned **85.75% to 84.50% RH**. Each onset lies between that droplet's preceding one-liquid observation and its first two-liquid observation, about 0.25 RH percentage points apart.

## Kinetic experiment

A 10 cm³ box at 298.15 K and 101325 Pa contains 1,000 native particle-resolved slots, each holding one physical particle. The generator with seed 2012 draws a distinct dry composition and size for each ID. Dry salt mass fraction spans 0.19 to 0.36, the KCl share of dry salt spans 0.15 to 0.70, and dry mass spans 7 × 10⁻¹⁵ to 1.3 × 10⁻¹⁴ kg. The remaining dry material is 2-propanol. Water exchanges with the gas, while the 2-propanol, NaCl and KCl inventories stay in each particle. We consider this uptake and evaporation timescale with two possible liquids, instantaneous internal redistribution and dissolved salts. A separate CPU calculation checks whether halite or sylvite would saturate at the onset of liquid separation.

The initial water inventory of each particle equilibrates with 90% RH, including curvature. An independent stable-phase evaluation verifies every initial droplet is homogeneous. The program lowers the target gas RH by 0.25 percentage points at the start of each 6 ms plateau, reaching 82% after 32 plateaus. Native condensation on the GPU conserves gas-plus-particle water during each plateau. Replacing the vapor concentration at the next plateau is an external water exchange, which is recorded separately. All four material densities enter particle volume and radius. The Kelvin calculation uses 1000 kg m⁻³ for water and a fixed surface tension of 0.072 N m⁻¹.

The solver selects AIOMFAC through Particula's activity factory in stable mode. It evaluates direct activities during transport and repeats the stability calculation at the accepted particle masses to record the number of liquids. The maximum absolute water-accounting error over the original run was 1.89 × 10⁻²² kg, or 9.10 × 10⁻¹⁶ of the initial system water mass. The dry masses were unchanged to the stored precision. All 1,000 onsets were bracketed, with no phase reversals. The original run took 326.72 s on an RTX 5060 with concurrent GPU qualification jobs.

## Independent equilibrium comparison

The predicted equilibrium cloud points span 85.1304% to 86.2115% RH. The independent CPU cloud-point calculation agrees with all 33,000 sampled phase counts when evaluated at the actual particle water fractions. Each particle's entire kinetic onset bracket lies below its corresponding equilibrium RH. Across the ensemble the median lower and upper bounds on this displacement are 0.305 and 0.555 RH percentage points. Both mineral saturation indices, log₁₀(IAP/Ksp), remain negative for all 1,000 cloud-point calculations. The largest indices are −0.828 for halite and −0.424 for sylvite. This saturation check applies specifically at the onset.

## Transport timestep refinement

Sixteen original IDs, evenly spaced from 0 to 999, were rerun at one, two and four transport substeps per 6 ms plateau. The forcing changes only at the original observation boundaries. The subset has one physical particle per ID in a 0.16 cm³ volume, preserving the original particle number concentration. Its one-substep run reproduced each selected particle’s onset bracket from the full experiment.

| Transport substep | Maximum particle water mass difference from four substeps | Onset brackets differing from four substeps |
| --- | ---: | ---: |
| 6 ms | 0.390% | 2 of 16 |
| 3 ms | 0.129% | 0 of 16 |
| 1.5 ms | Reference | Reference |

Both changed onsets moved one sampling interval later after refinement. The largest realized gas-RH difference between coarse and refined runs was 3.63 × 10⁻⁸ as a fraction. The largest absolute water-accounting error across all three runs was 3.65 × 10⁻²⁴ kg. A single particle's first observation therefore has timestep sensitivity of up to one 0.25-percentage-point interval in this control. The independent equilibrium comparison also includes physical water-transfer lag and the finite observation spacing.

## Reproduce

Run in the repository environment with Particula, its optional AIOMFAC package and Warp. The first two commands require CUDA hardware. Commands here use repository-relative paths.

```sh
export PYTHONPATH=packages/particula-aiomfac/src:.
python studies/aiomfac-mixing-state/ramp.py --output studies/aiomfac-mixing-state/ramp.json
python studies/aiomfac-mixing-state/ramp.py --refinement --output studies/aiomfac-mixing-state/timestep-refinement.json
OPENBLAS_NUM_THREADS=1 python studies/aiomfac-mixing-state/equilibrium.py
```

The kinetic archive contains 33 observations for each persistent particle ID, with water amount, phase count, first onset bracket, censoring status, net water exchange and maximum dry-mass error. It also contains the dry species inventory for each ID, imposed and realized gas RH, and the water budget. The independent CPU analysis calculates the cloud-point composition and matched Kelvin-corrected equilibrium RH for each of the same 1,000 dry compositions. It checks each sampled phase count against the corresponding cloud-point water fraction. These comparisons assess internal thermodynamic and numerical consistency for a synthetic experiment.

The two-salt configuration was selected after a pilot with 2-propanol, 1,4-butanediol and ammonium sulfate exposed a discrepancy between a solvent transfer chemical potential and the finite-difference Gibbs derivative. That pilot is reproducible with `equilibrium.py --screen-mixture`. The two-salt run retains every ID drawn from its stated input distribution.
