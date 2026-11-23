# particula-aiomfac

Version 0.1.4 evaluates liquid-phase activities with bundled AIOMFAC-web 3.15
parameters. NumPy and SciPy provide CPU evaluation. The optional Warp backend
evaluates the same chemistry in float64 on NVIDIA CUDA devices. It runs independently of Particula.
The PyPI installation commands for the standalone backend and the Particula integration are

```sh
python -m pip install particula-aiomfac
python -m pip install "particula[aiomfac]"
```

The combined install includes the backend. The first combined release targets Particula 0.3.5.
[CPU and CUDA accuracy, timing and supported table domains](https://github.com/uncscode/particula/blob/main/docs/studies/validation/condensation/aiomfac-sparse/index.md).

Each component is a neutral molecule described by explicit subgroup counts or a
salt described by its ion IDs and analytical formula-unit stoichiometry.

```python
from particula_aiomfac import AIOMFACModel, MixtureSpec

mixture = MixtureSpec({
    "water": {16: 1},
    "vanillic_acid": {9: 3, 10: 2, 17: 1, 24: 1, 137: 1},
    "butanediol": {142: 2, 150: 2, 153: 2},
    "NaCl": {202: 1, 242: 1},
})
prepared = AIOMFACModel.prepare(mixture)
result = prepared.evaluate([0.977, 0.001, 0.002, 0.020], 298.0, diagnostics=True)
water = result.neutral_activities[result.neutral_ids.index("water")]
mean_nacl = result.mean_electrolyte_coefficients[result.electrolyte_ids.index((202, 242))]
```

At 298 K the reference water activity is 0.9878567852129456 and the mean NaCl molal coefficient is 0.7037048661941817.

Use any common mass unit in mixture declaration order on the last axis.
Temperature in kelvin broadcasts over the leading axes. All outputs retain those
axes. Volatile neutral targets use the pure-liquid mole-fraction convention.
Aqueous CO₂ uses the upstream ion-specific reference correction and is excluded
from generic vapor transfer. Individual
ionic coefficients use infinite dilution in water and molality per kg of
electrolyte-free solvent. Salt outputs cover each cation–anion pairing with its
charge-balanced stoichiometry. Undefined log coefficients and mean coefficients
carry NaN plus a definition mask. Diagnostics report dimensionless SR/MR/LR
terms in neutral then ion order, ionic strength in mol/kg and the subtracted
log molality conversion. Warning codes 10 and 11 flag temperature extrapolation
for ionic and neutral systems respectively. Code 16 denotes activity-only
evaluation. The pinned model identity is available in each result.

Declare the dissolved analytical composition of one liquid phase. Preparation
resolves shared ions and implicit species. Sulfate alone uses direct evaluation.
Bisulfate, sulfate with a proton source, and closed bicarbonate systems activate
equilibrium as specified by the pinned preparation rules. The carbonate solver
updates water and aqueous CO₂ at every trial. When calcium is present in a
coupled system, the upstream rule first removes stoichiometric CaSO₄.
Missing interactions raise during preparation.

The last axis of `species_moles_per_kg_input` follows `neutral_ids` then
`ion_ids`. Amounts refer to one kilogram of the original analytical mixture.
`solver_status` is 0 for direct or degenerate rows and 1 for verified
equilibrium. `diagnostics=True` adds evaluation and retry counts,
`precipitated_caso4_mol_per_kg_input` and four `equilibrium_residuals`
in order of maximum absolute log mass-action error, normalized charge, carbon,
and sulfur imbalance. A failed active row raises an indexed `ConvergenceError`.

## Direct CUDA evaluation

Install the standalone wheel with the `cuda` extra or use the combined
Particula installation. Select the device during preparation.

```python
gpu = AIOMFACModel.prepare(mixture, execution="cuda", device="cuda:0")
result = gpu.evaluate([0.977, 0.001, 0.002, 0.020], 298.0)
```

For device-resident callers, `evaluate_into(masses, temperature, out, status,
workspace, stream=stream)` enqueues the calculation on the supplied Warp stream.
Supply contiguous arrays on the prepared device. Masses have shape
`(rows, analytical_components)`, temperatures have shape `(rows,)` in kelvin,
and the output and scratch arrays have shapes `(rows, gpu.output_width)` and
`(rows, gpu.work_width)`. All four arrays use float64. Use a separate int32
`status` array of length `rows`. Allocate disjoint writable buffers per
concurrent launch and keep them alive until the stream finishes.
`gpu.output_slices` maps quantity names to columns. Read outputs for rows with
nonnegative status. Negative codes identify invalid masses (−1), temperature
(−2), missing solvent or water (−3), unrepresentable coefficients (−4), or
failed equilibrium (−5). The host-facing API handles chunking and status checks.

## Particula integration

Build both local distributions and run all three installed-artifact checks from
the repository root. Use the resulting base wheel to select this local build
even when the same Particula version is available on a package index.

```sh
python packages/particula-aiomfac/tools/qualify_packages.py --dist-dir .artifacts/aiomfac-cpu-dist
python -m pip install ".artifacts/aiomfac-cpu-dist/base/particula-0.3.5-py3-none-any.whl[aiomfac]" --find-links .artifacts/aiomfac-cpu-dist/backend
python packages/particula-aiomfac/examples/closed_water.py
```

The [closed-water example](examples/closed_water.py) selects the provider with
`ActivityFactory().get_strategy("aiomfac", ...)`. Its material mapping identifies
each dissolved component once and lists inert lanes separately. The gas map
chooses the exchanging neutral species. Publish the compatible backend before
releasing the public Particula extra.

Backend 0.1.3 also binds the full dissolved composition to Particula’s
fixed-step and adaptive CUDA solvers. The two-vapor box exchanges water and
1,4-butanediol. The rising parcel adds pressure, temperature, a shared surface
heat balance and a vapor reservoir for each species. Both examples have CPU
references and restartable CUDA RK23 and ROS3 runs. Run the installed examples
with the GPU package check from the Particula repository root.

```sh
python packages/particula-aiomfac/tools/qualify_packages.py --cuda
```

The [Fortran reference guide](reference/README.md) describes the pinned
source, table generation, and numerical qualification. The backend wheel includes
the generated tables and GPL notices. The source archive includes the tools,
examples, reference corpus and tests.

## Precompute and use neutral activity tables

Install the backend, or select it through Particula's `aiomfac` extra. The
standalone `cuda` extra adds Warp. Generate a portable artifact using either
native calculator in an environment with the matching device.

```sh
particula-aiomfac-table build mixed-water --device cpu --output mixed-cpu.npz
particula-aiomfac-table build mixed-water --device cuda:0 --output mixed-gpu.npz
```

```python
from particula_aiomfac.sparse import ActivityTable, build_table

table = ActivityTable.load("mixed-water")  # or ActivityTable.load("mixed-gpu.npz")
model = table.prepare(device="cpu")
activities, log_error, reason = model.evaluate([[0.977, 0.001, 0.002, 0.020]], 298)
```

Each row contains mass amounts in the recipe's analytical order. Temperature
is in kelvin. Output columns follow `table.recipe.targets`. An occupied row
uses lookup (0) or same-device direct evaluation for absent components (1),
incompatible dry ratios (2), out-of-domain coordinates (3), or unqualified
cells (4). Vacant rows have reason −1. Direct failures raise an indexed error.
The empirical error expresses absolute uncertainty in natural log activity.
Fortran fixtures provide independent historical comparisons.

The installed recipes are `mixed-water`, `mixed-two-vapor`,
`mixed-variable-acid`, `neutral`, `shared-ion`, `sulfate` and `carbonate`.
Inspect one with `from particula_aiomfac.sparse import load_recipe`. To adapt
it, serialize `dataclasses.asdict(load_recipe("mixed-water"))` to JSON, edit
the mixture, targets, coordinates and tolerances, then pass the JSON filename
to the builder. The first two coordinates are natural log water/dry mass and
temperature. Subsequent coordinates are each named component's fraction of
the remaining dry mass. Run independent accuracy checks for each new domain.

For Particula condensation, add `"evaluation": "hybrid"` and
`"table": "mixed-two-vapor"` (or an artifact path) to the existing
`ActivityFactory().get_strategy("aiomfac", parameters)` configuration. Supply
the full analytical mixture and material mapping. Fixed stepping, RK23 and ROS3
guard the predicted flux direction. ROS3 uses the table in stage fluxes and
direct same-device chemistry for the physical Jacobian. This applies to CPU
and CUDA closed boxes and resident CUDA multi-vapor parcels. Ionic outputs
and speciation diagnostics use direct evaluation. Checkpoints bind the table digest.

### Solid–liquid equilibrium

Use a Particula AIOMFAC activity provider and anhydrous salt dissolution
constants to partition condensed material among the stable liquids and
crystals. In this example one cubic metre contains 1 µg of water and 1 µg
of total NaCl. At 298.15 K about 0.640 µg of halite crystallizes. Dilution
with an additional 9 µg of water dissolves all of the salt.

```python
from particula.particles.activity_factories import ActivityFactory
from particula_aiomfac import MixtureSpec, solve_solid_liquid

mixture = MixtureSpec({"water": {16: 1}, "NaCl": {202: 1, 242: 1}})
activity = ActivityFactory().get_strategy("aiomfac", {
    "mixture": mixture, "material_mapping": {0: "water", 1: "NaCl"},
    "phase_mode": "stable",
})
for water in (1.0, 10.0, 0.0):
    result = solve_solid_liquid(activity, [water, 1.0], 298.15, {"NaCl": "halite"})
    print(result.solid_masses, result.ion_masses, result.residual_ion_masses)
```

The two rows of `neutral_masses` and `ion_masses` refer to alpha and beta.
Sum these physical masses with `solid_masses` and `residual_ion_masses` for
the condensed inventory in µg/m³. The separate `dissolved_masses` vector
expresses those same liquids in analytical components. For feeds specified
as independent ions use `solve_ionic_solid_liquid`, supplying neutral names
and AIOMFAC ion IDs in mol/m³. Each liquid is electroneutral. On complete
dry-out, available ions form the specified crystals and unmatched ions retain
their identities. Wet reactive restarts accept physical dissolved species,
including bicarbonate, carbonate, aqueous CO₂, hydroxide, bisulfate and
sulfate when the corresponding chemistry is enabled. The crystal transfer
conserves total carbon, sulfur, solvent equivalents, spectator ions and charge,
so bicarbonate can supply a carbonate crystal. Source salts can supply
different crystals through shared ions. The bulk equilibrium calculation assumes immediate crystallization and
redissolution. Nucleation delays and metastable efflorescence require kinetics.

For each present crystal `log_saturation` is log₁₀(IAP/Ksp) and equals zero
at equilibrium. Absent crystals may be undersaturated. The quantity is
undefined for a completely dry sample. The supplied pure halite and sylvite
constants are log₁₀(Ksp) = 1.57 and 0.90 at 298.15 K on the unit-molality
scale. A caller can specify `SolidSalt(log10_ksp, temperature_min_k,
temperature_max_k)` for a declared anhydrous cation–anion formula, with a
constant value over its stated temperature interval. Hydrates and mixtures
triggering the upstream prescribed CaSO₄ removal have no explicit solid
model here. [USGS PHREEQC Pitzer database](https://github.com/phreeqc-dev/phreeqc3/blob/master/database/pitzer.dat)

At 298.15 K the saturated NaCl calculation gives a flat-surface RH of 75.53%.
The measured value is 75.29 ± 0.12%, a model difference of +0.24 percentage
points. The crystal saturation residual in this calculation is below 10⁻⁹
in log₁₀(IAP/Ksp). [Greenspan (1977), Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC5295834/)

The CUDA two-liquid search can be costly in dilute reactive carbonate
solutions. At 298.15 K, one stress case with water and CaCO₃ in a 1 to
0.00002 mass ratio and a synthetic crystal log₁₀Ksp of −4 reached an
unresolved device stability/solid status. The CPU solve gave one unsaturated
liquid with water activity 0.999990797. Native kinetic trials propagate
the numerical failure, allowing adaptive rejection or termination if
the controller exhausts its retries.
