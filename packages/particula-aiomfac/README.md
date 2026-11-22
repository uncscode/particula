# particula-aiomfac

Version 0.1.3 evaluates liquid-phase activities with bundled AIOMFAC-web 3.15
parameters. NumPy and SciPy provide CPU evaluation. The optional Warp backend
evaluates the same chemistry in float64 on NVIDIA CUDA devices. It runs independently of Particula.
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
python -m pip install ".artifacts/aiomfac-cpu-dist/base/particula-0.3.4-py3-none-any.whl[aiomfac]" --find-links .artifacts/aiomfac-cpu-dist/backend
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
