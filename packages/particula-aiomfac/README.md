# particula-aiomfac

Version 0.1.1 evaluates liquid-phase activities directly on the CPU using NumPy
and the bundled AIOMFAC-web 3.15 parameters. It runs independently of Particula.
Each component is a neutral molecule described by explicit subgroup counts or a
fully dissociated salt described by its ion IDs and formula-unit stoichiometry.

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
axes. Neutral activities use the pure-liquid mole-fraction convention. Individual
ionic coefficients use infinite dilution in water and molality per kg of
electrolyte-free solvent. Salt outputs cover each cation–anion pairing with its
charge-balanced stoichiometry. Undefined log coefficients and mean coefficients
carry NaN plus a definition mask. Diagnostics report dimensionless SR/MR/LR
terms in neutral then ion order, ionic strength in mol/kg and the subtracted
log molality conversion. Warning codes 10 and 11 flag temperature extrapolation
for ionic and neutral systems respectively. Code 16 denotes activity-only
evaluation. The pinned model identity is available in each result.

Preparation rejects missing parameters and raises `SpeciationRequiredError` for
declared systems that activate sulfate/bisulfate or carbonate equilibria. The
prepared model identifies upstream analog estimates in `estimated_pairs`.
The user supplies the dissolved composition of one liquid phase.

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

The [Fortran reference guide](reference/README.md) describes the pinned
source, table generation, and numerical qualification. The backend wheel includes
the generated tables and GPL notices. The source archive includes the tools,
examples, reference corpus and tests.
