# M6 implementation record

## Entry checkpoint — 2026-11-11

- Authorized objective: complete the approved fifteen-commit plan, clean/update
  documentation including full publication, and create one PR for #1607.
- Branch: `codex/e9-m6-legacy-removal`.
- Local base and freshly queried upstream main:
  `fb25078b6c0e8136ed06feeda59f0928b85c1b34` (M5 merge).
- Preserved the six existing planning-file modifications and reconciled the
  accepted plan: remove untagged execution/checkpoint admission; retain tagged
  v5 and explicit transfer functions; require both strict publication builds.
- Initial lexical inventory: `removal_inventory.json`, 95 matching files and
  740 test functions in matching Python files. Every disposition is pending;
  this is a search seed, not approval to delete tests or complete P1. Dynamic
  aliases, metadata conditionals and generated references still need expansion.
- Started the untargeted baseline with repository `.venv` on PATH:
  `python .opencode/tools/run_pytest.py --timeout 1200`.
  Output: `/tmp/m6-baseline-pytest.log`. Running session at handoff: 12235;
  verify that live handle before restarting. No passing result claimed yet.
- Planning diff whitespace check passed. No production changes or commits yet.

## Next work

Complete baseline collection and assertion-level inventory, starting with CPU
condensation native successors and surviving untagged CPU behavior. Map each
removed assertion before deleting its old fixture. Then execute serial commits
2–15 from `phase_details.md`. Preserve full required gates; do not shrink the
objective to facade imports alone. Readiness and PR creation remain pending.

## Condensation audit checkpoint

- Preserved 144 original strategy test nodes and 268 assertion/check expressions
  in `condensation_assertions.json`. This records the exact checks, not just
  function names; all successor/disposition reviews remain pending.
- Native baseline command (repository `.venv`):
  `pytest particula/dynamics/condensation/tests/native_step_test.py particula/dynamics/condensation/tests/native_step_admission_test.py particula/dynamics/condensation/tests/native_staggered_test.py particula/dynamics/condensation/tests/species_mapping_test.py -q --no-cov`
- Result: **331 passed in 0.65s**, exit 0. Log:
  `/tmp/m6-condensation-baseline.log`. Assertion evidence only, not coverage.
- Source inspection confirms native entry points already own mapped physics,
  staged inventory finalization and staggered private working state. Remaining
  legacy branches include unwrap/mixed-type guards, strategy fallback, raw-weight
  rate scaling, facade cache publication and separate old stepping paths.
- The untargeted baseline session 12235 remained active at the latest poll.

## Native execution guard preparation

Introduced shared `forbid_legacy_facades` test fixture. It inspects only
already-loaded facade classes to block cached constructors and blocks fresh
imports through `sys.modules`, without importing APIs slated for deletion.
Three native integration/adapter/full-loop suites now use that guard and retain
all original numerical assertions. Two guard regressions cover both already
loaded and absent legacy modules: **2 passed**. Scoped Ruff check/format passed.

Initial integration run: **59 passed, 4 skipped, 4 failed**. All failures were
Warp CPU cache writes to read-only `/home/kyle/.cache/warp`; rerun started with
`WARP_CACHE_PATH=/tmp/particula-m6-warp`, log
`/tmp/m6-native-guards-writable-cache.log`. No failed run is called passing.
The ongoing untargeted baseline lacks this cache override; inspect its outcome
and rerun with writable cache after it finishes. Test-guard files were edited
while that process was active, so do not label its outcome a pristine-base
revision result; final validation will run on a frozen executable revision.

## Native guard validation and publication preparation

- Writable-cache native integration/adapter/full-loop/guard run completed:
  **65 passed, 4 skipped in 44.24s**, exit 0. Log:
  `/tmp/m6-native-guards-writable-cache.log`.
- Repaired malformed Google section bullets and continuation indentation in
  106 production source files identified by M5's publication warnings. Each
  before/after AST was compared after stripping docstring nodes and was equal.
  No numerical code, signatures or runtime behavior changed in those files.
- Full package Ruff check passes and all 552 Python files pass format check.
- A full strict publication build is active (session 12505), output
  `/tmp/m6-publication-pass1.log`, site `/tmp/particula-m6-full-site`.
  Rendering still reports notebook Div warnings; publication is not yet clean.
- Documentation preparation was pulled forward while the original full-suite
  process runs. Keep it separate from API removal commits. Original baseline
  session 12235 remains live; do not restart until terminal state is observed.

## Publication parser audit

A direct Griffe traversal parsed **2710 production docstrings** without
rendering notebooks (test modules and aliases excluded). First repair audit:
147 warnings; second repair audit: **99 warnings**. Logs:
`/tmp/m6-griffe-audit.log`, `/tmp/m6-griffe-audit2.log`.
This is not equivalent to full publication success. Remaining issues include
stale parameters, missing types in unannotated APIs, and section indentation.

All **107** uncommitted package files were compared against HEAD after stripping
docstring AST nodes: executable ASTs are identical. Ruff check and format check
pass (552 files). Publication session 12505 and baseline session 12235 were
still live at last observation; preserve their handles. Current source changes
mean an in-flight publication run is diagnostic rather than final evidence.

Resume parser cleanup from `/tmp/m6-griffe-audit2.log`, retain scientific prose,
and then return to serial removal. Do not claim inventory P1 complete merely
because initial lexical and assertion snapshots exist. No facade production
API has been removed yet; this remains a required major outstanding deliverable.

## Parser cleanup completion (publication still pending)

Direct Griffe audit now parses **2710 docstrings with zero warnings**:
`/tmp/m6-griffe-audit4.log`. Fixed stale argument names/units against signatures,
missing documented types, continuation indentation and empty/incorrect sections.
This does not prove full publication rendering or visual correctness. Final AST
comparison and strict rendering remain required before documentation closeout.

Original untargeted run terminated exit 1: **7559 passed, 501 failed, 678 skipped**;
coverage gate passed. Cache configuration was defective as demonstrated by the
focused native-loop failures. Full rerun with `WARP_CACHE_PATH=/tmp/particula-m6-warp`
is active in session **67715**, log `/tmp/m6-baseline-writable-cache.log`.
Do not classify all original failures as cache-only without the corrected run.
Publication diagnostic session **12505** remains to be polled.

## First production consumer removal

- Committed docstring-only repair as `54a7d3257`: 113 files; final AST check
  against its parent confirmed executable AST identity. Direct Griffe parser
  was clean; this is not a final rendered-publication acceptance claim.
- First full publication diagnostic terminated exit 1 with 138 warnings. It
  began before later repairs; fresh build session **57066** uses repaired
  sources, log `/tmp/m6-publication-pass2.log` and site
  `/tmp/particula-m6-full-site-clean`.
- Removed Equilibria's legacy aggregate import/type union. Native result
  recognition remains; generic solver mappings/phase-result behavior unchanged.
- Strengthened adjacent return-identity test: separate real native input and
  output aggregates replace the old same-stub assertion, which could pass via
  the unknown-result fallback. No existing scientific assertion was deleted.
- Equilibria suites plus native CPU adapter suite: **69 passed in 0.10s**;
  log `/tmp/m6-equilibria-removal2.log`. Scoped mypy log:
  `/tmp/m6-equilibria-mypy.log`. Ruff and whitespace checks passed.

Scoped mypy did NOT pass: two override errors expose the still-native/legacy
`RunnableABC` contract in `runnable.py`. Do not suppress these errors or restore
the legacy equilibrium union. Next consumer cleanup must migrate the base
runnable and dependent annotations coherently; preserve existing run/sequence
behavior. Equilibrium assertions pass, but its type gate remains pending.

## Runnable base contract cleanup

Removed the legacy aggregate import/union from `runnable.py`; public native
nucleation annotations and dilution protocol now use native Aerosol. The still
pending LegacyNucleation deletion retains its original runtime behavior, with
an input-preserving generic return annotation so it cannot widen the native
base return contract. No type-error suppression was introduced.

Migrated runnable mock specification and the order-only nucleation sequence
fixture to native state, preserving dispatch-order/substep assertions. Updated
an adapter replacement sentinel cast to its declared native contract.

- Full package mypy: **clean across 552 files**; log
  `/tmp/m6-native-runnable-mypy2.log`.
- Runnable/execution/legacy-and-native nucleation/equilibria/native adapter
  group: **279 passed in 6.55s**; `/tmp/m6-runnable-native.log`.
- Scoped Ruff check/format passed.
- Full test session 67715 and full publication session 57066 remain pending.

## CPU adapter removal and completed diagnostic gates

Removed legacy aggregate unions from both CPU execution adapters. Migrated
ownership-only fixtures and subprocess imports to native Aerosol. Real CPU
integration fixtures now use tagged ParticleData, GasData and EnvironmentData;
condensation owns its physics and explicit species map on the process. Existing
kernel-oracle, dispatch, identity, uptake/evaporation/skip/inactive and
conservation assertions remain. Inactive native slots use canonical zero mass;
the obsolete all-zero-radius warning expectation is removed. An inert gas lane
is explicitly checked unchanged.

The coagulation mutation fixture now uses native RNG seed 0 instead of legacy
seed 41: the latter realizes no collision on this native path. The independent
Brownian kernel oracle, nonzero mutation, mass and charge conservation checks
are retained; no production numerics or tolerances changed. A temporary kernel
bin-count experiment was reverted.

- Adapter/integration group: **161 passed, 5 skipped**, log
  `/tmp/m6-adapter-removal3.log`; both earlier one-failure diagnostic runs are
  preserved in `/tmp/m6-adapter-removal.log` and `...removal2.log`.
- Full package mypy: **clean, 552 source files**,
  `/tmp/m6-adapter-mypy.log`. Scoped Ruff check/format passed.
- Prior writable-cache untargeted runner session 67715 completed exit 0:
  **8062 passed, 678 skipped, 92% coverage (unchanged 80% threshold)**,
  315.83 seconds, `/tmp/m6-baseline-writable-cache.log`. Source edits overlapped
  this diagnostic run, so final frozen-revision validation remains mandatory.
- Prior full strict publication session 57066 completed exit 0 in 237.45 seconds,
  `/tmp/m6-publication-pass2.log`. Pandoc unclosed-Div warnings remain in stderr;
  clean rendering is not claimed merely because MkDocs exited successfully.

Facade definitions and most CPU/GPU legacy branches still remain. Inventories
are pending disposition, final validation is pending, and no PR exists yet.

## CPU dilution legacy removal

Removed legacy aggregate admission, split gas setter machinery, facade snapshot
and rollback helpers, and the legacy whole-duration runnable shortcut. Native
per-substep behavior, preflight, raw-storage decay, normalized rate and rollback
remain unchanged. Explicit admission rejects non-native aggregate shapes before
reading fields. Updated public strategy/protocol/runnable annotations and the
AGENTS write-failure description.

Migrated the dilution helper and runnable fixtures to native containers with
non-unit volume and independent material/gas widths. `dilution_assertions.json`
preserves all **55 original test functions** and maps each to a current successor.
Numerical helper assertions remain unchanged. Facade setter and backing-object
replacement tests now map to native writer failure tests: before/partial failure
at both particle and full-gas writes, exact restoration, and held-array identity.
Native full gas has one writer, so the obsolete second gas-setter boundary is
not reproduced. No tolerance or coverage threshold changed. Broader source
admission matrices cover both no-work modes and every gas category.

Validation (repository .venv, writable Warp cache where applicable):

- Dilution helper/runnable/native runnable and native scientific/sequence
  integrations: **278 passed in 1.30s**, `/tmp/m6-dilution-removal4.log`.
- All `particula/dynamics/tests/`: **753 passed in 1.43s**,
  `/tmp/m6-dilution-broad2.log`.
- Dilution example and native process guides, `-Werror --no-cov`:
  **29 passed in 5.70s**, `/tmp/m6-dilution-examples.log`.
- Full-package mypy clean (552 files), `/tmp/m6-dilution-mypy2.log`;
  scoped Ruff check and format check passed.
- Earlier diagnostic runs had native-message expectation mismatches; two
  attempted groups used nonexistent test paths and ran no tests. Neither is
  counted as passing evidence. The corrected explicit paths above passed.

Remaining facade consumers, definition deletion, GPU untagged removal,
publication warning cleanup and final frozen-revision gates remain pending.

## Legacy nucleation runnable removal

Deleted `LegacyNucleation` (constructor-owned environment, facade topology and
cache synchronization, legacy source dispatch) from `particle_process.py`.
Native Nucleation's scientific implementation is unchanged. Updated commit
control units and the native scientific-process guide. Native source test guards
now use the common facade-free fixture and survive subsequent module deletion.

`nucleation_assertions.json` preserves and maps all **27 original runnable test
functions**. Migrated the fixture to explicitly tagged particles and unified
precursor/inert gas with a species map. Existing independent sequential transfer
math, live-gas depletion, held identities, inert preservation, capacity failure,
earlier-substep retention, zero admission/rate and invalid-input checks remain.
Native ownership replaces only obsolete compatibility assertions: held
environment validation instead of a constructor environment, per-substep state
validation instead of one facade lookup, shared immutable commit controls instead
of fresh legacy configs, and direct arrays/derived properties instead of caches.
Environment type rejection is also covered by `aerosol_test::test_wrong_container`.

Validation:
- Dynamics runnable and nucleation suites plus native scientific/sequence
  integrations: **1024 passed in 3.33s**, `/tmp/m6-nucleation-broad.log`.
- Nucleation and custom-source examples plus native aggregate tests with
  `-Werror --no-cov`: **16 passed in 2.73s**,
  `/tmp/m6-nucleation-examples.log`.
- Full package mypy: clean, 552 files, `/tmp/m6-nucleation-mypy.log`.
- Scoped Ruff passed; formatted one test after its expected-message adjustment.
- Initial focused diagnostic: 71 passed, one outdated native schema-message
  expectation; corrected in the passing broader run. No numerical changes,
  tolerance changes or lost scientific assertion were used to pass.

The common aggregate still has legacy condensation/coagulation/wall-loss paths;
those consumers and their strategies must migrate before facade deletion.

## Native-only wall-loss runnable

Removed WallLoss's legacy aggregate execution/rate and facade concentration
clipping branches. Native strategy invocation, equal durations, held-container
validation and in-place clamp remain unchanged; non-native aggregate objects
reject before strategy calls. Strategy-level facade branches are still pending.

Migrated all **11 original wall-loss runnable tests**, recorded with original
source and disposition in `wall_loss_runnable_assertions.json`. Explicit native
size lanes replace obsolete preset facades. Preserved reduction, no-op, rate
shape/sign, geometry/resolved smoke, equal substeps and composition assertions.
Native strategies already reject oversized steps before writing. The original
clamp assertion now uses successful custom-strategy output; a separate test
asserts oversized native rejection and exact unchanged storage. AGENTS and the
runnable docstring now distinguish those behaviors rather than promising native
clipping for arbitrary aggressive steps.

Validation:
- All dynamics runnable tests and native scientific/sequence integration:
  **807 passed in 2.08s**, `/tmp/m6-wall-runnable-broad.log`.
- Both required wall-loss strategy suites executed separately: **16 passed**
  and **43 passed**, `/tmp/m6-wall-top.log`, `/tmp/m6-wall-adjacent.log`.
- Full mypy clean (552 files), `/tmp/m6-wall-runnable-mypy.log`.
- Base strict documentation build passed in 1.59 seconds,
  `/tmp/m6-native-runnable-docs.log`. Full publication warning cleanup is still
  pending; this base build is not evidence for that separate gate.
- Scoped Ruff and whitespace checks passed. Initial focused run found the
  obsolete aggressive-step clipping expectation; its two native successor
  checks pass in the broader run. No physics or tolerances changed.

## Native-only wall-loss strategies

Removed facade unions/overloads/imports and legacy coefficient, rate and step
implementations from all CPU wall-loss strategies. Their existing native
implementations remain authoritative and now explicitly reject non-ParticleData
inputs before physical coefficient work. Removed the implicit charged-particle
cache reader: charge-free array calls assume zero charge; charged array calls
continue to take explicit charges through `compute_coefficient_from_arrays`.
The neutral/charged coefficient laws and standalone stochastic helper remain.

Both strategy suites had already migrated to native fixtures in M3 commit
`3ca096650`; their original 16 + 43 scientific tests are unchanged by this
removal. Added 16 population-boundary cases across four geometries/families and
four entry points, plus two independent charge-free helper checks. Existing
native independent coefficient, conservation/sink and stochastic checks remain.
Updated strategy/API prose and corrected the adjacent-suite collection note:
that suite is maintained native coverage and must still run explicitly.

Validation:
- Focused native + top-level strategy + runnable group: **437 passed**,
  `/tmp/m6-wall-strategies.log`.
- Separate adjacent wall-loss strategy suite: **43 passed**,
  `/tmp/m6-wall-strategies-adjacent.log`.
- All dynamics runnable tests plus native scientific/sequence integration:
  **825 passed**, `/tmp/m6-wall-strategies-broad.log`.
- Native process-guide examples with `-Werror --no-cov`: **23 passed**,
  `/tmp/m6-wall-strategies-examples.log`.
- Full-package mypy clean, 552 files, `/tmp/m6-wall-strategies-mypy.log`.
- Scoped Ruff/format and whitespace checks passed; base strict docs passed
  in 1.46 seconds, `/tmp/m6-wall-strategies-docs.log`.

CPU condensation/coagulation, facade definitions, untagged GPU/lifecycle
admission, complete inventories and final publication/readiness remain pending.

## Native coagulation mechanism fixtures

Migrated all six mechanism suites to explicit native discrete, radius-PDF,
and resolved initializers. Recorded all 43 original test functions and their
successors in `coagulation_mechanism_assertions.json`. Charged collision tests
now observe the real collision selector, retaining pre-merge charges and all
collision assertions without facade interception.

Seven former one-second update cases violate the existing native positivity
boundary. Their positive-update successors use 0.001 seconds; four additional
cases retain the original one-second inputs and verify rejection with exact
state preservation. No production algorithm or numerical tolerance changed.

Validation: six mechanism suites **47 passed**; full coagulation tree
**757 passed in 3.06s** (`/tmp/m6-coag-fixtures-broad.log`). Full-package mypy
passed for 552 files (`/tmp/m6-coag-fixtures-mypy.log`). These are fixture
migration checks, not evidence that shared legacy dispatch is removed.

## Native charged-coagulation integration scenarios

Replaced the legacy aggregate, atmosphere and resolved representation builder
in the three charged comparison scenarios with native resolved data, a held
explicit environment and an independent inert gas lane. All original merger,
mass and charge assertions remain, recorded in
`charged_integration_assertions.json`. Added the removal-safe facade guard,
per-material extensive conservation, held-object identity and protected
gas/environment assertions. The strategy RNG is explicitly seeded; original
physical inputs, timesteps and merger assertions are unchanged.

Validation: the three migrated cases pass; together with native scientific
integration and the full coagulation tree, **811 passed in 4.07s**
(`/tmp/m6-charged-integration-broad.log`). Focused mypy, Ruff, formatting and
whitespace checks passed. The initial collection attempt used an incorrect
EnvironmentData import; the corrected supported package import passed all
checks above. Shared coagulation dispatch and the remaining integration
fixtures still require migration/removal before facade definitions can go.

## Native-only coagulation runnable and general integration

Removed legacy aggregate dispatch from `Coagulation.execute` and `rate`.
Both require native Aerosol before strategy access; the established native
substep, current-environment, measure and held-container behavior is retained.
MassCondensation and the shared strategy compatibility paths remain pending.

Migrated all seven general coagulation integration tests to native initializers
and explicit environment/inert gas. `coagulation_integration_assertions.json`
records every original test body and its successor. Mass concentration is now
computed independently from native state: discrete sum, radius-PDF trapezoidal
quadrature, or resolved count-weighted sum divided by physical volume. Original
steps, sample counts, distribution parameters and tolerance assertions remain.

Replaced the native runnable suite's constructor-import guard with the shared
removal-safe fixture. Added six non-native admission cases and a replacement-
result rejection/held-identity regression. Updated the runnable docstring and
coagulation guide to state its native-only ownership contract accurately.

Validation at this implementation revision:
- General migrated integration suite: **7 passed in 0.78s**.
- All dynamics runnable tests, integration tests and coagulation tests:
  **1626 passed, 1 skipped in 6.24s**, `/tmp/m6-coag-runnable-final.log`.
- Full mypy: **552 source files passed**, `/tmp/m6-coag-runnable-mypy.log`.
- Scoped Ruff/format and whitespace checks passed.
- Base strict docs: passed in 1.86 seconds, `/tmp/m6-coag-runnable-docs.log`.
The full publication and final-revision readiness gates remain separate.

## Native-only shared coagulation step

Removed the legacy and untagged branches of `CoagulationStrategyABC.step`,
including their facade binning, getter mutation and duplicate collision paths.
The method now requires ParticleData with explicit distribution metadata and
calls the existing native step implementation unchanged. Zero-duration calls
also require metadata; no population interpretation is inferred. Removed the
now-unused legacy-step imports. Kernel/rate/helper compatibility is still
pending and is not claimed removed by this commit.

Nine original step-test bodies and native successors are preserved in
`coagulation_step_assertions.json`. Existing collision arrays, no-collision,
empty/zero-count, multibox rejection and additive-update assertions remain.
The old two-second discrete fixture expected negative concentration; it now
has a positive one-second update successor and a separate original-input
rejection test checking exact state preservation. Zero-edge rejection now
exercises the actual public controls rather than an injected legacy binning
result. Added explicit untagged and non-container admission checks. Updated
the stepping contract in source and user documentation.

Validation:
- Strategy suites after initial migration: **616 passed in 2.62s**.
- Full coagulation + dynamics runnable + integration + CPU coagulation adapter
  suites after boundary additions: **1718 passed, 3 skipped in 10.12s**,
  `/tmp/m6-step-removal-broad.log`.
- Entire source-example suite with `-Werror --no-cov`: **225 passed,
  2 skipped in 52.77s**, `/tmp/m6-step-removal-examples.log`.
- Full-package mypy passed for 552 files, `/tmp/m6-step-removal-mypy.log`.
- Scoped Ruff/format and whitespace checks passed; base strict documentation
  passed in 1.62 seconds, `/tmp/m6-step-removal-docs.log`.
These checks do not replace final full coverage, notebook execution or full
publication validation after all M6 removals.

## Complete native coagulation kernel/rate/helper boundary

Removed ParticleLike and all ParticleRepresentation imports, facade getters,
no-op unwrapping/type dispatch, untagged kernel/rate admission and duplicate
legacy kernel-binning implementations from the coagulation strategy tree.
Concrete kernels and gain/loss/net rates require explicitly tagged ParticleData.
Single-box material-property helpers retain raw internal material storage;
they neither infer a population measure nor provide population execution.
Shared step remains the previously validated native-only implementation.

Combined kernels now use a private material-law hook after outer population
and recursive control validation. Nested combinations retain zero-slot expansion
and physical laws without calling public kernels on untagged scratch. Custom
material laws retain keyword invocation, including keyword-only callables.
No numerical formulas or native tolerances changed.

`coagulation_helpers_assertions.json` records all original non-step helper/rate
test bodies and successors. Facade-only type/unwrap/None-charge contracts were
replaced with native admission, identity and explicit zero-charge checks.
The binning oracle uses the existing native count-weighted material average;
uncovered active particles reject with exact preservation rather than silently
vanishing. A separate empty-population test retains empty shapes and volume.
Mocked gain/rate fixtures now use four increasing nodes and a distinct PDF
coordinate as required by native admission. Added 80 concrete-family boundary
cases, three nested-combination cases, one keyword-only custom-law case and
one empty-bin case. Updated all obsolete strategy docstring examples and the
user-facing coagulation contract.

Validation:
- Initial run identified 19 obsolete fixture/helper failures; after migration,
  all **621** existing strategy cases passed before new boundary additions.
- Full coagulation, dynamics runnable and integration group: **1716 passed,
  1 skipped in 6.39s**, `/tmp/m6-coag-native-boundaries.log`.
- Entire source-example suite plus CPU coagulation adapter/integration suites,
  with `-Werror --no-cov`: **312 passed, 4 skipped in 45.86s**,
  `/tmp/m6-coag-native-boundaries-examples.log`.
- Full-package mypy passed, **552 files**,
  `/tmp/m6-coag-native-boundaries-mypy2.log`; the first run found an intentional
  wrong-type boundary call needing its explicit test-only type suppression.
- Scoped Ruff/format and whitespace checks passed. Base strict documentation
  passed in 1.60 seconds, `/tmp/m6-coag-native-boundaries-docs.log`.
- No remaining ParticleRepresentation, ParticleLike, is_native, or obsolete
  unwrapping/typeguard references in the coagulation tree. Remaining similarly
  named condensation helpers belong to the still-pending condensation removal.

## Native-only mass-condensation runnable and integration fixtures

Removed the final legacy aggregate dispatch from `MassCondensation.rate` and
`execute`. The whole particle_process module now has no legacy aggregate,
GasSpecies, ParticleRepresentation, atmosphere access or aggregate type union.
All five CPU runnables use the established native ownership path. Added
non-native admission and particle/gas replacement-result rejection coverage;
updated the public runnable contract and fixed its docstring import typo.

Migrated the isothermal resolved and latent-heat conservation integration
fixtures to native particle/full gas/environment data with explicit process
physics and species maps. Six original test functions (seven parametrized
cases) are recorded in `condensation_integration_assertions.json`; all original
scientific tolerances and finite/nonzero/conservation assertions remain. Gas
and particle inventories remain densities for comparison; native latent energy
is extensive joules, so its expectation multiplies the particle density gain
by the explicit physical parcel volume. Accepted scalar-material input shapes
are exercised through native initialization, and a real two-material container
replaces the removed getter monkeypatch for helper rejection.

The broader run found one additional latent-enabled/disabled runnable test
still building a legacy aerosol. Migrated both cases and recorded its native
successor in `condensation_assertions.json`; native type and exact held-object
identity assertions replace obsolete facade type assertions. Remaining shared
condensation strategy fixtures and dispatch still require migration/removal.
Two existing native integration guards now use the removal-safe shared fixture,
retaining every scientific/construction/replacement/transfer assertion.

Validation:
- Migrated condensation integration cases: **7 passed in 0.03s**.
- Condensation tree, all dynamics runnable/integration tests, CPU condensation
  adapter and integration suites: **1591 passed, 4 skipped in 12.67s**,
  `/tmp/m6-cond-runnable-broad2.log`. The initial run's single obsolete runnable
  fixture failure was resolved, without weakening numerical checks.
- Entire source-example suite with `-Werror --no-cov`: **225 passed,
  2 skipped in 42.98s**, `/tmp/m6-cond-runnable-examples.log`.
- Full-package mypy: **552 files passed**, `/tmp/m6-cond-runnable-mypy2.log`.
- Scoped Ruff/format and whitespace checks passed; base strict documentation
  passed in 1.52 seconds, `/tmp/m6-cond-runnable-docs.log`.
Full final coverage, notebook execution and full publication remain pending.

## Native staggered conservation, performance and stability fixtures

Migrated all three ancillary staggered suites to tagged resolved ParticleData,
GasDataBuilder and explicit process-owned activity/surface/vapor-pressure laws
and water species maps. Removal-safe guards prohibit compatibility constructors.
Original particle counts, seeds, timesteps, theta/batch matrices, numerical
thresholds and timing bounds are retained. Separate staggered assertion ledgers
preserve the original test bodies and identify native successors.

The conservation helper now explicitly sums count-weighted particle inventory
and gas concentration times representative volume. Original volume-one cases
retain their numeric meaning. Fixed an ineffective mass-change assertion that
previously compared aliases after in-place mutation: it now uses a pre-step
owned snapshot and the unchanged 1e-15 bound. Renamed two conservation cases
whose gas loadings did not establish the supersaturation/equilibrium claimed
by their names; preserved fixture values and scientific assertions.

Updated test prose and AGENTS.md to describe native vectorized batches with
sequential gas coupling. Removed unsupported claims of per-particle Python
loops, universal overhead/complexity and a tenfold stable-timestep advantage.
No production scientific code or algorithm was changed in this commit.

Validation:
- Condensation tree plus integration suites: **729 passed, 1 skipped in 9.29s**,
  `/tmp/m6-staggered-migrations-broad.log`.
- Final three migrated suites after test-name/prose corrections: **54 passed
  in 5.91s**, `/tmp/m6-staggered-final.log` (32 conservation, 9 performance,
  13 stability cases; no marker exclusions).
- Full-package mypy: **552 files passed**,
  `/tmp/m6-staggered-migrations-mypy.log`.
- Scoped Ruff/format and whitespace checks passed after formatting.
Shared condensation strategy legacy dispatch and its main test suite remain
pending; these focused passes are not final M6 coverage or readiness evidence.

## Preserve staggered controls independently of legacy fixtures

Moved 32 existing constructor, theta/RNG, batch/shuffle and clipping-log tests,
plus their logging helper, into `staggered_controls_test.py`. Their executable
method ASTs are identical to the original methods. The new class has no legacy
fixture setup and uses the shared removal-safe facade guard. Updated each old
node's native successor in `condensation_assertions.json`. This prevents the
pending main condensation fixture removal from discarding control/RNG coverage.

Validation: original condensation strategy suite plus extracted controls:
**144 passed in 0.16s**, `/tmp/m6-staggered-controls.log`; scoped Ruff/format and
whitespace checks passed. No production behavior changed. Main shared
condensation legacy execution removal remains pending.


## Native binned latent/isothermal condensation regressions

Migrated twelve binned regression methods and their fixture from the shared
legacy suite into `native_binned_condensation_test.py`, under the removal-safe
facade guard. Removed their old representation/species construction and
conversion calls. The native fixture retains radius nodes, lognormal inputs,
material densities/fractions, activity/surface/vapor-pressure laws, transport
parameters, gas loading, bin counts and tolerances. Strategies now use full gas
molar masses with an explicit ordered three-species map.

The PDF fixture now declares `continuous_pdf` and its radius grid, making
conservation an independent trapezoidal integral of mass times dN/dr; the old
untagged fixture incorrectly summed raw PDF nodes. Discrete cases sum number-
density-weighted mass. Retained total-mass assertions and added tight individual
species inventory comparisons (rtol=1e-12, atol=1e-30). Zero-latent-heat parity,
shape checks and the proposal-based discrete latent-energy oracle remain;
energy explicitly uses extensive applied mass. Rejection checks now match
native validation messages and prove both containers' arrays remain unchanged.
Every old node has its successor in `condensation_assertions.json`.

Validation:
- First focused run: ten passed; two failed solely on legacy error-message
  expectations, corrected to the existing native messages.
- Full condensation tree and integration suites after strengthened assertions:
  **729 passed, 1 skipped in 9.56s**, `/tmp/m6-native-binned-broad.log`.
- Full-package mypy: **554 source files passed**,
  `/tmp/m6-native-binned-mypy.log`.
- Scoped Ruff/format and whitespace checks passed.
No production stepping changed. Remaining shared condensation compatibility
and all final release/publication gates remain pending.

## Native resolved latent-heat condensation regressions

Migrated all 44 remaining latent-heat methods into
`native_latent_condensation_test.py`, guarded against facade construction.
Fixtures now directly create resolved ParticleData and full ordered GasData;
physics and species maps are explicit process inputs. Preserved material
properties, gas saturation/loading, representative volume, latent/transport
settings and numerical tolerances. Updated every original node's successor.

Native shape assertions retain both particle/species axes, including the
single-species case. Former facade-type checks now assert native types and
input identity. Native admission intentionally rejects the old -1e-13 count
instead of the obsolete norm-concentration helper's tolerance clamp; a mapped
replacement verifies rejection without mutation. Error expectations follow
existing native validation rather than compatibility helper messages.

Moved nonfinite-pressure, gas exhaustion and viscosity probes to the active
native calculation boundary. Nonfinite pressure injection is asserted consumed
and produces exact zero lanes. Surface-pressure checks now inspect the mapped
native calculation, including rejection of nonfinite Kelvin values. Gas-update
oracle derives representable per-particle increments from the independent
proposal calculation before count/volume scaling, retaining rtol=1e-15/atol=0.
Energy oracles convert mass-density transfer into extensive mass using volume;
latent energy continues to mean J, not J/m^3. No production math changed.

Validation:
- Focused native latent suite: **44 passed in 0.07s** after correcting native
  shape/error/seam expectations, `/tmp/m6-native-latent-third.log`.
- Final full condensation tree plus integration: **729 passed, 1 skipped
  in 9.21s**, `/tmp/m6-native-latent-broad.log`.
- Full-package mypy: **555 files passed**, `/tmp/m6-native-latent-mypy.log`;
  final changed-file annotation check passed,
  `/tmp/m6-native-latent-types-final.log`.
- Scoped Ruff/format and whitespace checks passed.
Isothermal/staggered fixture migration and shared strategy removal remain.

## Native isothermal/staggered fixtures and empty multi-species admission

Migrated the remaining main condensation fixtures to tagged resolved data,
full ordered gas, explicit process physics and species maps. Removed all facade
construction and conversion calls from the condensation test tree. Native
multi-box rejection fixtures now construct repeated storage directly. Native
input/type/identity tests replace obsolete unwrapping flag assertions; free-slot
zero-rate tests replace the legacy one-metre/max-radius fallback and warning.
Scalar/full-gas transport checks replace the legacy per-particle stepping
helper. Batch-coupling/order tests now inject rates into the active native
batch path, preserving nonzero outcome and current-gas assertions.

Corrected the original staggered total-mass test's mixed-unit expression to
particle mass density plus gas concentration, retaining rtol=1e-10. Empty-state
gas comparison now uses a pre-step snapshot rather than mutated aliases.
Four facade-only comparisons of legacy and untagged routes are removed with
explicit dispositions; existing independent native physics/scalar oracles
retain the scientific checks, unchanged. All 144 original main-suite test
nodes now have reviewed successors/dispositions in condensation_assertions.json.

Migrating the original empty-population case exposed a native empty-material
Kelvin broadcasting error with a two-species surface model. After gas and
activity validation, native pressure evaluation now returns its empty arrays
without asking the surface law to evaluate nonexistent radii. This is an empty
no-op boundary correction, not a change to nonempty numerical stepping. Added
three strategy-specific cases proving empty rate shape, held identity,
unchanged gas, and rejection of nonfinite vapor-pressure configuration despite
empty material.

Validation:
- Migrated main suite before three added empty regressions: **52 passed in
  0.08s**, `/tmp/m6-native-iso-third.log`.
- Condensation tree, dynamics runnables, integration, and CPU condensation
  adapter suites: **1590 passed, 4 skipped in 13.70s**,
  `/tmp/m6-native-iso-broad.log` (four obsolete comparisons removed, three
  empty regressions added relative to the prior equivalent group).
- Full-package mypy: **555 files passed**, `/tmp/m6-native-iso-mypy.log`.
- Scoped Ruff/format and whitespace checks passed.
The now-unneeded production legacy condensation dispatch can be removed next.
Final coverage, examples/notebooks, publication and installed-wheel gates remain
pending; none are inferred from these focused runs.

## Native-only shared CPU condensation strategies

Removed facade imports, type unions, overloads, unwrapping/matching flags,
untagged dispatch, implicit carrier-owned physics, and legacy isothermal,
staggered and latent stepping from condensation_strategies.py. Removed obsolete
per-particle/batch transfer, shape-normalization, radius-fill, raw-concentration
mask and legacy latent bookkeeping helpers. Scientific mass-transfer utilities
remain in their owning modules. Pressure, mass-rate, density-rate and step entry
points now admit only tagged ParticleData/GasData with explicit process maps
and physics. Internal material-only batch scratch remains private and does not
interpret population semantics.

Nine native scientific/control helpers are executable-AST identical to the
preceding commit: native parameters, single step, staggered step, rate from
parameters, latent values, Kelvin lane, theta draws, batch construction and
staggered control validation. Strategy resolution now reads only process-owned
configuration. All 144 original test successor references were checked against
actual source symbols. The remaining private-helper assertions now exercise
native pressure admission, three-particle/two-species shapes and real skip-map
rate masking. Existing latent numerical oracles consume already-masked native
rates instead of calling a deleted compatibility helper.

Added 36 boundary cases across all three strategies and all four public
entrypoints: untagged, wrong particle type and wrong gas type reject without
particle/gas array mutation. Updated native time-step error expectations after
removing the old wrapper. Rewrote public method docstrings and corrected class
examples to require native state and process-owned physics; the user-facing
CPU condensation guide now records completed compatibility removal. GPU
untagged documentation remains pending its separate implementation phase.

Validation:
- Before new boundary cases: **1590 passed, 4 skipped in 13.74s**,
  `/tmp/m6-cond-removal-second.log`; the first run found only three obsolete
  time-step message expectations, corrected without changing rejection behavior.
- Expanded group run concurrently with examples hit the existing theta-mode
  timing ratio limit (3.06x versus 2.6x), with **1625 passed, 4 skipped**. No
  threshold or workload changed. After examples finished, serial rerun of the
  same full affected group passed: **1626 passed, 4 skipped in 13.35s**,
  `/tmp/m6-cond-removal-serial.log`.
- Entire source-example suite with `-Werror --no-cov`: **225 passed,
  2 skipped in 44.20s**, `/tmp/m6-cond-removal-examples.log`.
- Full-package mypy: **555 files passed**,
  `/tmp/m6-cond-removal-final-mypy.log`.
- Base strict documentation passed in **1.68s**,
  `/tmp/m6-cond-removal-docs.log`; scoped Ruff/format and whitespace passed.
- No obsolete helper names or legacy facade imports/unions remain in the CPU
  condensation strategy module or condensation test tree.
Final untargeted coverage, supported notebook execution, full publication and
installed-wheel checks remain pending; no overall readiness claim is made.
