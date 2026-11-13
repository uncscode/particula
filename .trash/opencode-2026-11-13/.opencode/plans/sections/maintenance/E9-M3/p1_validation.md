# P1 implementation evidence

Entry commit: `81ba14f02`, based on merged M2 `fd114270a`.

The CPU slice introduces concrete-module `CondensationSpeciesMap`, current
state admission and integrated density weights. All three native rate families
use explicit mapping with full gas-ordered transport/vapor parameters and full
particle composition for activity/surface calculations. Native output keeps
particle species order; skipped/unmapped/nonpartitioning lanes are zero.
Builders/factory carry the process configuration without constructing data.
Legacy facade and untagged conversion fixtures remain a temporary compatibility
path; their passing results are not claimed as tagged native evidence.

## CPU assertion ledger

New file: `particula/dynamics/condensation/tests/species_mapping_test.py`.

| Enduring contract / prior coverage | Native assertion |
|---|---|
| Native strategy setup and missing strategy errors in condensation_strategies_test.py | native_strategy; test_native_strategy_configuration_rejects_without_mutation, across all three families |
| Species skip/rate behavior | test_native_rate_density_and_disabled_lanes; test_full_gas_flags_and_skip_do_not_compact_storage |
| Independent rate physics, replacing facade parity as sole oracle | test_native_mapped_rates_against_independent_equations: permuted three gas/two particle lanes, explicit molecular transport/Kelvin equations at V=.25/1/4 |
| Single-box rejection | test_valid_multibox_state_is_rejected_before_scientific_work |
| Builder/factory construction | test_factory_and_fluent_builder_preserve_native_configuration across three builders, retained process identities and cleared optional reload state |
| Representation/current-state admission | mutated particle/gas snapshots, invalid map/skip/environment, fixed-volume PMF/PDF rejection |
| Population weighting | independent resolved extensive species inventory, nonuniform PDF product quadrature, unchanged PMF density |
| Unmapped particle material and inactivity | test_native_activity_retains_unmapped_particle_material; test_inactive_nodes_have_no_native_mass_rate |

## Checks so far

Repository `.venv`; commands below are assertion evidence unless identified as
the untargeted runner. No tolerance or coverage-policy changes.

- `pytest particula/dynamics/condensation/tests/species_mapping_test.py
  particula/dynamics/condensation/tests/condensation_factories_test.py
  particula/dynamics/condensation/condensation_builder/tests/
  particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py
  -q --no-cov --tb=short`: 142 passed.
- `pytest particula/dynamics/condensation/ -q --no-cov --tb=short`:
  404 passed in 5.04 s. Includes retained legacy strategy/stability/helper tests.
- `ruff check particula/dynamics/condensation/`: passed.
- `mypy particula/dynamics/condensation/ --ignore-missing-imports`: success,
  24 source files (existing untyped-test note is informational).
- `.opencode/tools/run_linters.py`: Ruff and mypy passed. Literal log under
  `.artifacts/e9-m3/p1-cpu/linters.log`.
- Untargeted `.opencode/tools/run_pytest.py`, with the repository environment
  on PATH and normal host cache/hardware access: 7,005 passed, 19 skipped in
  141.65 s; coverage 92% against the unchanged 80% threshold. Literal log:
  `.artifacts/e9-m3/p1-cpu/package-tests-host.log`. This validates the CPU
  source slice recorded here, before the next GPU changes.
- The initial restricted run failed 135 tests. A focused reproduction exposed
  a read-only Warp compilation cache, and CUDA was unavailable there. The
  successful host rerun above supersedes that environment-limited attempt;
  no assertions or tolerances were changed to obtain the passing result.

The first independent-rate run exposed vector constant-vapor-pressure scalar
temperature broadcasting; native vector evaluation now supplies a full-gas
float temperature vector and ordered per-species model sequences remain
supported. Factory tests use their existing explicit unit-key contract.

## GPU metadata and explicit mapping

The concrete design and bounded support matrix are recorded in
`implementation_record.md`. New co-located tests:

- `particula/gpu/tests/population_metadata_test.py`: raw discrete/resolved
  round trips, V=.25/1/4 independent species inventories, legacy retention,
  unsupported restore rejection before runtime probe, independent lane widths,
  full gas order, skipped lanes, stale bindings, and metadata-only validation.
- `particula/gpu/tests/population_boundary_test.py`: newly tagged consumers
  reject before payload access until their owning science phase (or M4).
- `particula/execution/tests/gpu_session_test.py`: native CPU setup rejects
  before transfer; resident revalidation rejects a later population-tag change.

Assertion migration: the old all-tag transfer rejection in
`particle_population_test.py` becomes unsupported-PDF rejection, retaining the
before-runtime boundary. The M2 aggregate construction integration test now
verifies that supported resolved metadata reaches the explicit runtime probe,
without requiring Warp for CPU construction evidence. Actual device round trips
and independent inventory checks are covered by `population_metadata_test.py`.
Its six old rejection assertions were the only failures in the first full GPU
slice run; no scientific tolerances or coverage policy were weakened.

Validation with the repository environment and normal cache/hardware access:

- `pytest particula/gpu/tests/population_metadata_test.py
  particula/gpu/tests/population_boundary_test.py
  particula/execution/tests/gpu_session_test.py -q --no-cov --tb=short`:
  223 passed in 39.85 s, including Warp CPU and CUDA. Two subsequent no-device-
  work parameter cases are included in the following focused/full runs.
- `pytest particula/integration_tests/native_aerosol_construction_test.py
  particula/gpu/tests/population_metadata_test.py
  particula/gpu/tests/population_boundary_test.py -q --no-cov --tb=short`:
  78 passed, including Warp CPU and CUDA. Log:
  `.artifacts/e9-m3/p1-gpu/focused-tests.log`.
- `.opencode/tools/run_linters.py`: Ruff and mypy passed. Log:
  `.artifacts/e9-m3/p1-gpu/linters.log`.
- Untargeted `.opencode/tools/run_pytest.py`: 7,078 passed, 19 skipped;
  coverage 92% against unchanged 80% threshold. Log:
  `.artifacts/e9-m3/p1-gpu/package-tests-host-final.log`.

P1 implementation and its validation checkpoint are complete on the M3 branch;
PR review/merge remain pending. P2 owns mapped native stepping, limiting,
conservation and latent energy, including replacement of interim GPU science
guards. The new rate and configuration paths alone do not establish correct
mapped steps. Later phases and M3 completion are not claimed by these results.
