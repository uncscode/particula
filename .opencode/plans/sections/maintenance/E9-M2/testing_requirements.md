# Testing Requirements

## Coverage and co-located policy

1. Test coverage thresholds must NEVER be lowered.
2. Each phase must include self-contained tests.
3. Tests are committed in the same PR as the implementation.
4. Test files use `*_test.py` suffix in module-level `tests/` directories.
5. Coverage comes only from the full applicable suite using repository-configured
   full-package coverage and its normal threshold through the untargeted
   `.opencode/tools/run_pytest.py` runner.

The generic template's fixed 80% example is not a new local gate or an accurate
reason to override active configuration. Focused fix checks use coverage-disabled
assertion evidence (`--no-cov`). Focused-target coverage is invalid evidence,
not a fix failure. After focused assertions pass, run the untargeted runner;
do not pass coverage source/threshold overrides or raw comprehensive `--cov`
controls. No local `-Werror`; retain the configured warning policy.

## Required behavioral matrix

| Area | Assertions | Owner |
|---|---|---|
| Flat construction | Exact three held identities, candidate nonmutation, no facade invocation, current direct writes visible | P1 |
| Individual replacements | All three accepted cases, unchanged peers, same-object validation, wrong type/box/species cases | P2 |
| Coordinated replacement | New compatible layout accepted by identity; each preflight failure leaves entire old/candidate triple untouched | P2 |
| Alignment | Distinct species values, interleaved `[True, False, True]`, all-false flags, wrong declared mapping/order and dimensions | P1–P3/P5 |
| Builders | Required inputs, documented allocation/copy defaults, fluent use, constructor-consistent validation | P3 |
| Radius-bin presets | PDF/PMF distinctions, mass/radius conversion, units, invalid distribution, non-unit volume | P4 |
| Resolved presets | Controlled sampling, mass/density/charge shapes, count and volume, rejection, no implicit normalization | P5 |
| Data-only integration | Initialization → construction → access/replacement; unchanged quantities except explicitly supplied replacement state | P6 |
| Protected boundaries | Existing runnable behavior pending M4, scientific behavior pending M3, retained transfer/export contracts and no automatic GPU rebind | All affected phases/P6 |

Use M1's approved distribution convention with V=0.25 and 4 m^3. Independently
calculate physical number density (w/V for counts, c for density storage),
particle species inventory `V * sum_i(m_i,s * c_i)` and gas inventory
`V * gas_concentration_s` using the approved map. Keep per-particle total mass
distinct. Construction/wrapping must not silently change these quantities;
replacement selects caller-provided state, not a conservation operation that
rescales it. Assert component outputs, not only an all-species sum. Preserve
existing tolerances; use explicit scale-justified float64 comparisons and exact
identity/array/metadata snapshots for nonmutation checks.

Use declared metadata for order validation; do not promise detection of an
unnamed same-shape permutation. Direct mutation does not bypass later process
validation. Preserve single-box CPU rejection tests; batched construction
does not authorize multi-box execution. Migrate enduring facade-fixture
assertions with their owning construction phase, leaving later-track tests
operational through the approved temporary transition rather than deleting them.

## Future implementation validation sequence

Refine focused paths against the actual diff and include any new adjacent
initialization modules and P6 data-only integration test. Representative commands:

```bash
pytest particula/tests/aerosol_test.py particula/tests/aerosol_builder_test.py -q --no-cov
pytest particula/particles/tests/particle_data_builder_test.py particula/particles/tests/representation_builders_test.py particula/gas/tests/gas_data_builder_test.py -q --no-cov
pytest particula/dynamics/tests/dilution_test.py particula/dynamics/tests/nucleation_runnable_test.py -q --no-cov
pytest particula/dynamics/tests/wall_loss_strategies_test.py -q --no-cov
pytest particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q --no-cov
pytest particula/gpu/tests/conversion_test.py particula/gpu/tests/kernel_exports_test.py -q --no-cov
.opencode/tools/run_pytest.py
.opencode/tools/run_linters.py
mypy particula/ --ignore-missing-imports
python3 .opencode/tools/build_mkdocs.py --validate-only --strict
mkdocs build --strict
```

Both wall-loss suites are explicit guards; the adjacent concrete suite is
excluded from normal recursive collection. Warp CPU is required when installed;
CUDA is optional pass-or-clean-skip, never fallback. Add an affected prepared
state identity regression if needed to demonstrate no rebind, without adding
production hooks or changing resident lifecycle. No performance claims required.
Log literal outputs and final revision; required unavailable/failing checks
block M3 handoff. None of these commands ran during this drafting task.
