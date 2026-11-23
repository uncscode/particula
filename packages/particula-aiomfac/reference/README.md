# Reproducing the activity reference

Use Python 3.12+, Git, GNU make and GNU Fortran on Linux. Run the following
from the Particula repository root. The workflow also works in the unpacked
backend source archive using `tools/reference.py`. Set `FC` to the path of a
specific GNU Fortran compiler if needed. Only the first fetch needs network access.

```sh
ref=packages/particula-aiomfac/tools/reference.py
python "$ref" fetch
python "$ref" build
python "$ref" run --case mixed
python "$ref" verify
python "$ref" fetch --offline
python "$ref" verify
```

The checkout, generated driver, build products, compiler log, and candidate
JSON are under `.artifacts/aiomfac-reference` in the calling directory.
`AIOMFAC_CACHE` can point to a different dedicated cache. Fetch checks the Git
revision, cleanliness and SHA-256 of every build input. Build copies verified
files into a temporary directory, builds the standard executable, instruments
that copy and rebuilds the activity executable. Generation metadata binds the
executables to the sources, generator, compiler and flags. The input CSV used
by the standard driver is copied to its temporary run directory. Cases use
explicit subgroups and require no external SMILES or viscosity service.

`verify` writes fresh candidates, checks physical identities and compares
against tracked JSON. Inspect numerical differences by case, species and field.
Use `update` to deliberately replace expectations after reviewing the candidates.
Review the resulting Git diff, upstream changes, and provenance before accepting
new fixtures. The core writes list-directed double precision before CLI formatting.
The independent mixed-case check uses the stock executable on row 1 of the
**original** example and checks every species' mass fraction, mole fraction,
molality, coefficient and activity within half of the last printed decimal place.
Both drivers share the upstream equations, so this comparison tests the harness.
The two additional independent checks use the neutral activity identity and the
geometric-mean relation for a 1:1 electrolyte.

## Reading case and result files

The components registry maps stable names to upstream integer subgroup IDs and
counts, including salt stoichiometry. Each case states the component order,
temperature in K, source, independent-component mass or mole fractions and
expected status. The upstream text format implies the first component by complement. The
harness checks the complete composition first. The carbonate example deliberately
uses the original analytical mole fractions, including the trace bicarbonate. A second case omits the zero CO₂ input and checks that upstream’s implicitly added aqueous CO₂ gives the same named results.

- `species` contains raw reported mass fractions, dissociated mole fractions,
  molalities in mol per kg of electrolyte-free solvent, coefficients and
  activities. Ion keys are upstream subgroup IDs. Ionic activities use molality
  relative to the standard molality of 1 mol/kg. The upstream carbonate model
  treats aqueous CO₂ specially. Preserve its raw outputs and use the pinned
  carbonate convention before interpreting them.
- `neutrals` and `ions` record natural log coefficients and their dimensionless
  SR, MR and LR contributions after the final equilibrium calculation.
  `molality_conversion` is the upstream Tmolal correction for ionic coefficients.
  Fields are sampled at the end of AIOMFAC_calc. Absent species have
  `defined=false` and a null log coefficient. The raw CLI convention of zero
  for an absent component is preserved separately in `species`.
- `salts` gives ion IDs, stoichiometric counts, mean molal coefficient and
  molal ion activity product. Undefined means are null. Charge balance is the
  signed sum of charge times molality divided by the total absolute ionic charge
  on the same basis. Ionic strength uses one half the sum of molality times
  charge squared.
- `equilibrium.sulfate` samples the final signed log-K residual and initial and
  final sulfur molality in the sulfate routine. The closed carbonate vector
  samples the final DiffK_carb_sulf call, including MINPACK's return status.
  Its first three entries are log-equilibrium residuals. Coupled cases add
  the bisulfate log-equilibrium residual as the fourth entry. Its last two are
  upstream-scaled mole balances for hydrogen and carbon. The independent
  normalized carbon balance uses the solver's initial carbon and summed final
  species amounts. Coupled cases also record the final dissolved sulfur against
  the initial dissolved inventory after any upstream CaSO₄ removal. Each relative balance uses its conserved initial amount.
- `status`, `errors` and `warning` distinguish input rejected by Python,
  rejected upstream subgroup definitions (code 13), temperature extrapolation
  (code 10) and successful activity evaluation. Upstream warning 16 records
  deliberately disabled viscosity. An input error has no scientific outputs.

Every canonical point runs in a fresh process. Verification also compares
reordered organic species and carbonate sequences A–B–A and B–A–B within
one process against the cold-start results. Ordinary log-term disagreement
must be ≤ 1e-8. Other numeric fields use absolute 1e-10 and relative 1e-8.
The equilibrium residual gate is 1e-8 and normalized balance gates are 1e-10.
A discrepant upstream result requires case-specific review and recording the
discrepancy before any acceptance change.

The trace-bound sulfate–carbonate stress case stops with upstream error 17 and
MINPACK status 4. The GNU 13.3 Ubuntu build stops at a slightly different
unsuccessful iterate from the canonical GNU 16.2 build. The largest observed
difference in its residual vector was 3.324e-9. Both builds leave a carbon
imbalance of about 7.26%. The [compiler observation](compiler-observations/coupled-trace-bound.json)
records the four affected diagnostics, exact compiler identity and canonical
fixture digest. Verification uses the same field tolerances and still checks
the complete failure status, remaining species and balances. Converged reference
cases continue to use one fixture across compilers.
[Original GNU 13 comparison](https://github.com/uncscode/particula/actions/runs/45093663412/job/137428595837)

## Package qualification

From the repository root, run
`python packages/particula-aiomfac/tools/qualify_packages.py` with the Python
`build` frontend installed. The script builds both wheels **from their source
archives**, inspects payloads and metadata, and installs base-only,
backend-only, and combined into three clean environments. Pip needs access to
ordinary Particula dependencies via an index or wheel cache. Use
`--no-isolation` when flit_core and build are already available. Run the cheap
fixture tests with `python -m unittest discover -s packages/particula-aiomfac/tests`
in an environment with the backend installed. Computational reference tests use
the commands above. The base version is controlled independently by Particula.

## Reproducing the installed parameter tables

From the repository root, fetch the pinned sources as above. With NumPy in the
developer environment, run the following commands. The exporter builds an
isolated Fortran driver, runs SRdata and MRdata, reads Fortran-order arrays, and
compares their inventory, shapes, dtypes, values and SHA-256 hashes. Add
`--update` to deliberately replace the runtime bundle.

```sh
python packages/particula-aiomfac/tools/export_parameters.py
python -m unittest discover -s packages/particula-aiomfac/tests
OPENBLAS_NUM_THREADS=1 python packages/particula-aiomfac/tools/benchmark_cpu.py
```

The runtime [manifest](../src/particula_aiomfac/data/manifest.json) gives the
offset to subtract from upstream IDs along each axis, the physical units,
the active dataset, and explicitly identified analog salt pairs. Hashes cover
little-endian C-order bytes after restoring the original array shape.
The source manifest identifies the Fortran modules. The short-range dataset
uses only the A interaction matrix. Mixture-specific PEG overrides and the
water reference terms are applied during preparation and evaluation.


## CPU and CUDA equilibrium qualification

Run the backend corpus with `python -m unittest discover -s packages/particula-aiomfac/tests`.
For physical CUDA qualification install the backend CUDA extra and set
`AIOMFAC_TEST_CUDA=1` for the same command. An unavailable device fails this
requested run. Use `python packages/particula-aiomfac/tools/qualify_packages.py --cuda`
to exercise the installed standalone and combined CUDA wheels outside the checkout.
The coupled fixtures include simultaneous carbon and sulfur inventories and a
calcium-containing system. The reference driver starts each canonical state in a
fresh process. The production solvers cold-start each row.
