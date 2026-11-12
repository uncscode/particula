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

## Remove the legacy aggregate and its remaining execution fixture

Deleted `_legacy_aerosol.py` after migrating the remaining real CPU-adapter
fixture to native particle/gas/environment storage. Particle count divided by
its explicit 2 m^3 volume retains the previous number-density observation;
partitioning and inert gas lanes retain their concentrations. The original
exponential dilution oracle (`rtol=1e-12`, `atol=0`) and adapter lifecycle
assertions are unchanged.

Reviewed all nine legacy aggregate tests in `aggregate_assertions.json`.
Rendering and same-object replacement have existing native successors. Added
six native builder cases for independent material/gas widths, including an
all-nonpartitioning gas case. The former equal-width restriction is obsolete:
process maps own species correspondence. An absent legacy partitioning species
is represented by explicit all-false gas metadata, not an empty native gas
container (GasData requires at least one named lane).

Added `legacy_api_removal_test.py` to check that the concrete legacy aggregate
module is absent for the intended reason and native public construction/building
still preserve all held identities. Remaining `_legacy_aerosol` references in
package sources are only the removal-safe guard and negative tests.

Validation:
- Focused aggregate, replacement, builder, adapter and guard suites:
  **239 passed in 5.63s**, `/tmp/m6-aggregate-focused.log`.
- Root package tests, integration and execution suites, including the new
  removed-module test: **1759 passed, 53 skipped in 90.67s**,
  `/tmp/m6-aggregate-broad.log`.
- Full-package mypy: **553 files passed**, `/tmp/m6-aggregate-mypy.log`.
- All nine assertion successor symbols resolve to existing native tests;
  scoped Ruff/format and whitespace checks passed.
Final full coverage, remaining facade/GPU removal, notebook publication and
installed-wheel gates remain pending.

## Remove the obsolete facade kernel-binning wrapper

Deleted `particles/change_particle_representation.py` and its facade-only
fixture after tracing all three original test bodies in
`representation_binning_assertions.json`. Native coagulation already owns
radius-grid generation and count-weighted material representatives. Its
scientific implementation is unchanged. Added six cases covering one/two
materials at V=0.25/1/4: exactly ten positive increasing radius edges,
nonnegative finite masses, native result identity/metadata, unchanged input,
total count and per-species extensive inventory with rtol=2e-15 and atol=0.
Existing native tests retain zero-material lanes, median kernel-only charge,
nonaliasing storage and invalid-edge admission. Legacy unweighted median/mean
mass aggregation is not retained as a second population conversion API.

Added the removed concrete module to negative import checks. The only remaining
wrapper-module reference in package/docs/root publication sources is that
negative check; no runtime consumers or package exports remained.

Validation:
- Coagulation, particle and removed-API suites: **1345 passed in 3.21s**,
  `/tmp/m6-binning-removal.log`.
- Full-package mypy: **551 files passed**, `/tmp/m6-binning-mypy.log`.
- Scoped Ruff/format and whitespace checks passed.
Particle representation definitions/builders/factories and facade bridges are
still pending their assertion-level audits; this is not final M6 readiness.

## Remove particle facade conversion bridges

Removed `from_representation` and `to_representation` from particle_data.py and
package exports. ParticleData's complete class AST is unchanged. The bridge
was the container module's remaining dependency on facade and process strategy
classes. All 16 old bridge test bodies are preserved and mapped in
`particle_bridge_assertions.json`; native properties and population tests remain.

Added owned native storage/copy cases at V=0.25/1/4 and one/two/three boxes,
with unequal counts, distinct material masses, charge, densities, extensive
species inventory and derived total mass. Radius construction retains raw
fields and an independent sphere-law check for discrete/PDF data; explicit PDF
number density integrates over radius. Old implicit mass-shape inference,
arbitrary binned volume, PDF slot summation and facade deprecation logging are
intentionally retired. Existing native PDF quadrature evidence remains intact.
Negative import checks cover both bridge names at root, package and concrete
module boundaries, distinguishing missing symbols from a broken module import.
Updated the foundations guide to describe removed bridges and native copying.

Validation:
- Particle, removed-API and native-construction integration suites:
  **494 passed in 0.27s**, `/tmp/m6-particle-bridge-tests.log`.
- GPU conversion and complete source-example suites with warnings as errors:
  **336 passed, 2 skipped in 40.88s**, `/tmp/m6-particle-bridge-examples.log`.
- Full-package mypy: **552 files passed**, `/tmp/m6-particle-bridge-mypy.log`.
- Base strict docs passed in **1.55s**, `/tmp/m6-particle-bridge-docs.log`.
- All 16 successor symbols resolved; scoped Ruff/format and whitespace passed.
Remaining particle facade definitions/builders/factory and gas/GPU compatibility
are still pending. This does not replace final full coverage or publication.

## Remove the obsolete particle representation factory

Removed representation_factories.py and its package export after confirming
there are no remaining production/example consumers. All six original tests
are recorded in particle_factory_assertions.json with existing native or
removed-boundary successors. Added explicit native discrete/resolved material
construction retaining per-particle total mass, per-species density, scalar
charge, concentration and volume. Explicit material axes replace the old
ambiguous one-dimensional mass/multiple-density facade convention. Native
radius initialization intentionally rejects unequal material density inputs;
its existing rejection and sphere-law tests are retained.

Added default/configured native binned and resolved preset checks using the
original configured modal values and 1000-slot resolved size. Existing native
independent quadrature and seeded sampler tests remain unchanged. New sample
checks restore NumPy global RNG state. Removed concrete-module and exported
factory-symbol tests cover the obsolete entry points; invalid factory-key
handling has no surviving supported API.

Validation:
- Particle, removed-API and native-construction integration suites:
  **497 passed in 0.22s**, `/tmp/m6-particle-factory-tests.log`.
- Full-package mypy: **550 files passed**, `/tmp/m6-particle-factory-mypy.log`.
- All six successor symbols resolved; scoped Ruff/format and whitespace passed.
- Remaining factory references in package/docs/root publication sources are
  only the explicit negative removal checks.
Particle facade/builders, gas APIs, GPU admission and final publication/readiness
remain pending.

## Remove obsolete particle builders and lossy normalization

Deleted representation_builders.py, its five package exports and the two old
builder suites after mapping all ten original tests in
particle_builder_assertions.json. Native mass/radius/lognormal construction
remains supported and unchanged. Added lossless singleton row/column/None
charge cases with redundant species density, plus explicit empty population
mass/charge checks. Old normalization that discarded unequal charge lanes or
invented zero density from empty population rows is intentionally retired;
existing native invalid-input tests and the new empty-density rejection cover
those boundaries. Removed-module and root/package builder-symbol checks cover
all five obsolete constructors.

Updated the remaining rectangular wall-loss docstring construction to the
native lognormal initializer. Executing its five statements exposed the old
1-second example step exceeding native admission (maximum rate coefficient
2.3753/s); changed only the example step to 0.1 seconds. All five docstring
statements now execute. No wall-loss executable code or admission changed.

Validation:
- Particle, removed-API and native-construction integration suites:
  **504 passed in 0.23s**, `/tmp/m6-particle-builder-tests.log`.
- Full-package mypy: **547 files passed**, `/tmp/m6-particle-builder-mypy.log`.
- Rectangular strategy docstring: **5 attempted, 0 failed**,
  `/tmp/m6-particle-builder-docstring.log`.
- Base strict docs: **1.43s**, `/tmp/m6-particle-builder-docs.log`.
- All ten native successors resolve; scoped Ruff/format and whitespace passed.
The particle facade itself and a scientific sorted-bin helper still require
review before removal. Gas/GPU and final publication gates remain pending.

## Remove ParticleRepresentation after native scientific assertion audit

Deleted representation.py and its export after auditing all 43 remaining test
bodies in particle_facade_assertions.json. Native storage/copy and initialization
retain raw fields, shape checks, explicit count/volume interpretation and
species provenance. Added the original discrete bulk number/mass sums and
single/multispecies mass-weighted effective density, including the original
2.1526515151515153 mean baseline at unchanged rtol=1e-7. Corrected a recently
introduced explanatory comment to match that retained mass-weighted formula;
no density algorithm or numerical oracle changed.

Retained the direct distribution-strategy scientific tests, including identical
charged append input/output checks and mass/radius update oracles. Facade-only
strategy getters, cached .data wrapping, clone flags, warning logs and disabled
charge sentinels are retired explicitly. Native lossless charge forms and
rejection of missing charge lanes have successor checks. Empty population
material density remains explicit; no zero-density inference survives.
The standalone properties/sort_bins.py implementation is unchanged; all four
existing sorting tests retain their bodies and now import its owning module
instead of relying on the facade module's incidental import.

Added concrete-module and root/package symbol absence checks. Updated migration
and roadmap guidance to reflect actual particle API removal and identify
remaining gas/readiness work. Updated local scope/phase metadata to distinguish
implemented particle removal from final acceptance; review units can contain
multiple coherent prerequisite commits within the single PR.

Validation:
- Particle tree, root package tests and integration: **1072 passed, 1 skipped
  in 11.98s**, `/tmp/m6-particle-facade-tests.log`.
- Complete source examples with warnings as errors: **225 passed, 2 skipped
  in 42.13s**, `/tmp/m6-particle-facade-examples.log`.
- Full-package mypy: **544 files passed**, `/tmp/m6-particle-facade-mypy.log`.
- Base strict docs passed in **1.46s**, `/tmp/m6-particle-facade-docs.log`.
- All 43 successor symbols resolve; scoped Ruff/format and whitespace passed.
Final installed-wheel absence, full coverage and full publication evidence
remain pending alongside gas/GPU compatibility removal.

## Remove gas species conversion bridges

Removed from_species/to_species and their package exports. The complete GasData
class AST is unchanged; only obsolete conversion functions and their imports
were removed. All fourteen bridge test bodies are retained in
`gas_bridge_assertions.json`. Native storage/copy tests cover one/two ordered
species, one/three/five boxes, equal/distinct concentrations, uniform true/false
and mixed partitioning, kg/m^3 passthrough at rtol=1e-10/atol=0, and detached
names/array ownership. The former mixed-partitioning rejection is obsolete;
native data preserves the complete mask. Bridge-specific strategy-length,
box-selection and deprecation behavior are replaced by explicit missing-symbol
checks at root/package/concrete boundaries. No vapor-pressure algorithms changed.
Updated migration guidance to name removed gas bridges and native construction.

Validation:
- Gas tree, removed APIs and integration: **480 passed, 1 skipped in 2.62s**,
  `/tmp/m6-gas-bridge-tests.log`.
- Explicit CPU/Warp conversion suite: **111 passed in 0.19s**,
  `/tmp/m6-gas-bridge-conversion.log`.
- Full-package mypy: **544 files passed**, `/tmp/m6-gas-bridge-mypy.log`.
- Base strict docs: **1.52s**, `/tmp/m6-gas-bridge-docs.log`.
- All fourteen successor symbols resolved; GasData AST unchanged; scoped
  Ruff/format and whitespace passed. Only negative tests reference the removed
  bridge symbols in package sources.
GasSpecies/Atmosphere and their constructors plus GPU admission and final
readiness remain pending.

## Remove Atmosphere and its obsolete builder

Deleted atmosphere.py and atmosphere_builders.py plus their package exports.
All six original tests are recorded in atmosphere_assertions.json. Added native
GasDataBuilder/EnvironmentData/Aerosol construction for the original oxygen +
hydrogen and oxygen + nitrogen inputs, retaining temperature, pressure, names,
molar masses, concentrations and full/partitioning species counts. Extending a
species layout now exercises validated coordinated gas/environment replacement;
an individual incompatible replacement rejects while original carriers remain
unchanged. The old test named 'without species' actually omitted pressure;
its successor checks that native environment construction requires pressure.
No native aggregate or gas/environment scientific code changed.

Removed concrete-module and root/package symbol checks cover Atmosphere and
AtmosphereBuilder. Migration guidance documents the separate native owners and
coordinated replacement requirement. Remaining package references are limited
to the removal-safe guard and explicit negative checks.

Validation:
- Gas tree, package tests and integration: **769 passed, 1 skipped in 11.92s**,
  `/tmp/m6-atmosphere-removal-tests.log`.
- Complete source-example suite with warnings as errors:
  **225 passed, 2 skipped in 42.51s**, `/tmp/m6-atmosphere-removal-examples.log`.
- Full-package mypy: **541 files passed**, `/tmp/m6-atmosphere-removal-mypy.log`.
- Base strict docs: **1.50s**, `/tmp/m6-atmosphere-removal-docs.log`.
- All six native successors resolve; scoped Ruff/format and whitespace passed.
GasSpecies/builders/factory, GPU compatibility and final readiness remain pending.

## Remove obsolete gas species constructors

Deleted species_builders.py and species_factories.py and their package exports.
All fourteen original constructor tests are preserved with native successors or
explicit facade-only dispositions in gas_constructor_assertions.json. Native
GasDataBuilder tests retain Oxygen and Preset100 field values, fluent setters,
required-field admission and rejection of invalid molar mass/concentration
without input mutation. Vapor-pressure strategy ownership remains on processes;
the obsolete generic factory dispatch and gas-owned strategy setter are removed.
Removed-module and root/package symbol tests cover all three constructors.

Validation:
- Gas tree, removed APIs and integration: **488 passed, 1 skipped in 2.60s**,
  `/tmp/m6-gas-constructor-tests.log`.
- Full-package mypy: **538 files passed**,
  `/tmp/m6-gas-constructor-mypy.log`.
- All fourteen successor symbols resolve. Scoped Ruff check passed; package
  import whitespace was formatted after the test run, without behavior changes.
GasSpecies itself, GPU admission and final readiness remain pending.

## GasSpecies assertion audit preparation

Preserved all 37 remaining GasSpecies test bodies in gas_facade_assertions.json;
their individual successor dispositions remain pending review. The facade and
all three original test suites remain present until that review is complete.
Added native builder cases with the original single/multispecies storage values,
zero concentrations, explicit multibox shapes and mixed partitioning. Retained
the original scalar/vector constant-vapor-pressure assertions by calling the
owning scientific strategy directly. Invalid species-lane checks reject without
clamping or mutating input. GasData itself remains a shape-validating storage
container; physical input validation here belongs to GasDataBuilder.

- Focused native construction suite: **26 passed in 0.03s**.
- Scoped Ruff check and formatting passed.
- This is prerequisite assertion evidence, not completion of facade removal,
  final integration, or publication gates.

## Remove GasSpecies after native assertion review

Deleted species.py and its export after assigning all 37 preserved tests native
successors or explicit facade-only dispositions in gas_facade_assertions.json.
Original storage, constant-pressure, valid concentration arithmetic and combined
species ownership checks remain. Native explicit species combination preserves
source containers; aggregate layout changes use validated coordinated replacement.
Mixed masks and multibox storage are supported instead of retaining obsolete
facade restrictions. Implicit scalar/array mode caches, name duplication,
negative clamping, deprecation messages and operator overloads are retired.
GasData and vapor-pressure science are unchanged. Native builder validation
rejects invalid physical inputs; raw array writes remain caller-owned.

Removed-module and root/package absence tests cover GasSpecies. Production
sources contain no remaining consumers; only the removal-safe test guard and
negative boundary checks mention the old module/class. Updated migration and
roadmap guidance plus P3 status to reflect implemented removal, with final
readiness still pending.

Validation:
- Gas tree, package tests and integration: **760 passed, 1 skipped in 11.91s**,
  `/tmp/m6-gas-facade-tests.log`.
- Complete source-example suite with warnings as errors:
  **225 passed, 2 skipped in 41.96s**, `/tmp/m6-gas-facade-examples.log`.
- Full-package mypy: **534 files passed**, `/tmp/m6-gas-facade-mypy.log`.
- Base strict documentation build: **1.57s**, `/tmp/m6-gas-facade-docs.log`.
- All 37 successor mappings resolve; scoped Ruff/format and whitespace passed.
GPU untagged admission, final installed-wheel/full-suite/notebook/publication
validation and the single PR remain outstanding.

## GPU condensation native-only execution work in progress

Removed untagged admission from direct/prepared condensation setup and removed
15 legacy selectors from prepared enqueue. Setup now requires explicit discrete
or resolved metadata and an exact CondensationSpeciesBinding before accessing
payloads. The private prepared record requires all mapping/population fields
without legacy defaults. Native numerical kernels, four equal substeps and
independent scientific tolerances are unchanged. Six new negative cases cover
missing/zero/PDF metadata on both direct and prepared entry points before array
access, allocation or launch.

Preserved 174 GPU condensation test bodies in gpu_condensation_assertions.json;
their migration dispositions remain pending. The legacy standalone finalizer
and old unmapped kernels still require caller/assertion review before deletion.
The historical contract suite fixtures still construct untagged populations;
they must migrate to explicit population semantics and species maps, preserving
all applicable assertions. This work is not ready to commit as a completed unit.

- Native suite after setup/enqueue cleanup: **137 passed, 131 skipped in 12.64s**,
  `/tmp/m6-native-gpu-condensation.log`.
- Scoped mypy passed two files before removing private-record legacy defaults;
  `/tmp/m6-gpu-condensation-mypy.log`.
- Historical suite diagnostic stopped after three expected metadata-admission
  failures (8 skipped), `/tmp/m6-gpu-condensation-migration-diagnostic.log`.
  It is failing evidence, not accepted validation.
- Follow-up native/boundary run after requiring prepared-record fields:
  **151 passed, 131 skipped in 0.94s**,
  `/tmp/m6-native-gpu-condensation2.log`, exit 0.

## GPU condensation fixture migration checkpoint

Migrated the shared numerical fixtures to explicit discrete number-density
semantics and unit representative volume, preserving concentrations, masses and
numerical tolerances. Added explicit synthetic identity species bindings at 98
direct/prepared call sites; production never infers maps. Updated equivalent
mapped-kernel launch hooks and native admission-message assertions. Two P2
energy oracles now weight transfer by number density and representative volume
before latent heat, matching the upstream native extensive-energy contract.
The separate nonidentity/mixed-width native oracle suite remains intact.

Retained the standalone private P2 inventory primitive as density-weighted
scientific functionality: it now requires explicit discrete metadata and rejects
untagged/PDF/resolved inputs before setup. It does not mutate gas concentration
or substitute for the mapped public step. Its original independent NumPy
finalization and atomicity tests pass without relaxed tolerances. This replaces
the earlier pending decision about this helper; remaining unused raw kernels
still require a caller audit.

Graph-capture fixtures now pin species bindings during setup, outside capture,
and use weighted extensive-energy expectations. Real adapter execution fixtures
use native metadata/maps; malformed deferred work still reaches its intended
validator. Metadata-only adapter fixtures and production carrier admission still
have legacy branches and require the next migration before this unit is complete.
All inventoried tests remain in place with resolvable successor names and
original bodies preserved in gpu_condensation_assertions.json.

Validation:
- Contract, native and population boundaries: **390 passed, 178 skipped in
  1.88s**, `/tmp/m6-condensation-fixture-migration7.log`.
- Shared stiffness/graph/autodiff fixtures plus condensation adapter tests:
  **77 passed, 4 skipped in 5.84s**,
  `/tmp/m6-condensation-shared-fixtures3.log`.
- Full-package mypy before the last adapter fixture edits passed **534 files**;
  `/tmp/m6-gpu-condensation-mypy2.log`. Follow-up after those edits also passed
  **534 files**, `/tmp/m6-gpu-condensation-mypy3.log`, exit 0.
- Intermediate diagnostic failures are retained in migration2–6 logs; they
  exposed fixture semantics, message, launch-hook and energy-oracle differences.
  Passing focused results do not establish final package/execution readiness.

## Condensation carrier admission and dependent execution audit

Removed WarpCondensationState's untagged admission branch. It now requires an
explicit population kind and pinned CondensationSpeciesBinding, and adapter
dispatch always forwards that binding by identity. Existing ordered primary
metadata/output validation remains; construction performs no device work.
Migrated 16 metadata-carrier test constructions to explicit one-species maps.
Binding setup precedes operation guards; carrier construction is checked against
allocation, launch, copy and synchronization. Added explicit untagged/missing-map
rejection tests retaining all caller resources and a binding-forwarding assertion.

Migrated the direct integration fixture to discrete density semantics and an
explicit vapor map. The mixed-transfer oracle previously charged gas for raw
proposals lost to particle-mass rounding; it now couples the independently
computed representable mass increment, matching the upstream native contract.
The native extensive-energy oracle weights by number density and unit volume.
No numerical tolerance changed. Preserved integration test originals in the
GPU condensation assertion ledger. Corrected the no-energy launch sentinel to
watch the native energy kernel rather than an unreachable legacy kernel.

Validation:
- Adapter and integration suites: **76 passed, 3 skipped in 4.48s**,
  `/tmp/m6-condensation-carrier5.log`.
- Contract/native kernel suites: **376 passed, 178 skipped in 1.38s**,
  `/tmp/m6-condensation-carrier-kernels.log`.
- Full execution diagnostic finished: **72 failed, 1213 passed, 51 skipped,
  100 errors in 84.69s**, `/tmp/m6-execution-condensation-migration.log`.
  This diagnostic began before the integration-fixture fixes above. Remaining
  failures include shared full-loop/captured-loop constructors, graph fixtures,
  resident enqueue, communication, diagnostics and state-update fixtures.
  Inspect their shared setup before another full run; the broad gate is failing.
- Scoped Ruff and whitespace passed. Mypy exposed a Warp metadata annotation
  mismatch in a test fixture; changed its literal to wp.int32(1). Follow-up
  full mypy passed **534 files**, `/tmp/m6-condensation-carrier-mypy2.log`.

The condensation unit remains uncommitted pending migration of dependent
execution fixtures. Next inspect full_loop_test.py's WarpCondensationState
construction near line 368 and captured_full_loop_test.py's near line 1398;
graph/diagnostic/resident tests share these setup paths. GPU coagulation,
wall-loss, source/communication, lifecycle/checkpoint removal and final
publication/readiness remain required after this dependency cleanup.

## Native full-loop fixture migration

Converted the shared full-loop particle number densities to explicitly tagged
resolved counts using each original representative volume. Its independent
inventory now sums count-weighted particle masses plus volume-weighted gas.
Pinned explicit condensation and total-mass diagnostic species maps. Retained
all original tests and their assertions in the GPU condensation ledger.

Migrated the captured two-box scenario similarly. Its detached NumPy oracle
continues to use density: host comparison snapshots explicitly divide native
counts by current volume. Particle communication work-buffer expectations now
convert independently calculated density debits/credits/requests to native
counts using initial source/destination volumes. No tolerance changed.

Validation:
- Full-loop suite: **17 passed in 0.71s**,
  `/tmp/m6-full-loop-native3.log`.
- Graph-capture and diagnostics suites: **206 passed, 1 skipped in 5.41s**,
  `/tmp/m6-full-loop-dependents1.log`.
- Captured-loop suite: **3 failed, 53 passed, 12 skipped in 1.17s**,
  `/tmp/m6-captured-loop-native3.log`. All remaining failures enter the separate
  `_build_prepared_loop` fixture via `multi_box_loop_test._scheduler_request`,
  whose carrier still lacks native population metadata and a condensation map.
  The migrated two-box scenario and both particle-communication volume cases
  pass. Next migrate the shared multi-box fixture and preserve its independent
  inventory, stochastic, and snapshot assertions.
- Scoped Ruff and whitespace checks passed; formatting applied to both files.
  Final full-suite, type, publication, and installed-wheel gates remain pending.

## GPU condensation removal unit validated

Migrated the remaining shared multi-box fixture to resolved counts while
preserving its original density-defined inventory. Added explicit process and
diagnostic mappings, and the captured-loop zero-source nucleation binding.
The independent scenario oracle remains density-based at its documented host
comparison boundary. All 284 assertion-ledger entries retain resolvable test
successors; original bodies and current body-change flags are recorded.

Removed three now-unreferenced private legacy kernels: unweighted energy
accumulation, unmapped gas-transfer reduction, and unmapped partition gating.
CodeGraph and repository reference searches found no remaining callers. Retained
`condensation_mass_transfer_kernel` because the independent autodiff test uses
it, plus the explicitly discrete standalone inventory-finalization primitive.

Validation:
- Multi-box plus captured-loop: **62 passed, 13 skipped in 2.61s**,
  `/tmp/m6-multibox-native2.log`.
- Full execution suite: **1384 passed, 52 skipped in 83.25s**,
  `/tmp/m6-execution-condensation-migration2.log`. This run started before
  deletion of the three unused kernels; directly affected consumers were rerun
  afterward as recorded below.
- Condensation kernel/native/autodiff after dead-kernel removal: **381 passed,
  179 skipped in 3.44s**, `/tmp/m6-condensation-dead-kernels2.log`.
  The initial command used a nonexistent contract filename and collected no
  tests; the corrected command used `condensation_test.py`.
- Adapter, integration, full-loop, multi-box and captured-loop final consumer
  rerun: **155 passed, 16 skipped in 6.56s**,
  `/tmp/m6-condensation-consumers-final.log`.
- Full-package mypy: **534 files passed**, `/tmp/m6-condensation-mypy5.log`.
- Ruff check and format check passed all ten changed Python files; whitespace
  check passed. Focused checks are assertion evidence, not package coverage.

The condensation removal unit is ready to commit. Next remove untagged GPU
coagulation/wall-loss paths, then source/communication and lifecycle/checkpoint
compatibility. Full-package, installed-wheel, exhaustive example, publication,
and final PR gates remain pending.

## GPU wall-loss removal in progress

Removed untagged admission from direct, prepared, selected, and prepared-selected
wall-loss boundaries. Native geometry and ownership validation now always apply;
no stochastic kernel or coefficient formula changed. Added 16 metadata-only
rejection cases for missing/zero/discrete/PDF semantics across the four paths,
which reject before particle-array access. Original wall-loss test bodies and
successors are preserved in `gpu_wall_loss_assertions.json`.

Migrated direct fixture tags and metadata-only configuration/schema sentinels.
Converted legacy silent skips of underflowed/overflowed geometry to native
rejection with unchanged particle/RNG state. The large-duration and eligible-RNG
cases first assert invalid-state rejection, then explicitly construct valid
inactive slots to retain the original valid removal/stream assertions. No
numerical tolerances changed.

Validation checkpoint:
- Full wall-loss/direct/parity/native diagnostic: **26 failed, 263 passed,
  126 skipped in 1.49s**, `/tmp/m6-wall-native4.log`.
- Current direct/native admission subset: **259 passed, 118 skipped,
  16 deselected in 0.72s**, `/tmp/m6-wall-admission.log`. The deselected rows
  are the former native-versus-legacy stream comparison, still to migrate;
  this subset is not full unit acceptance.
- Scoped Ruff passed; formatting applied.

Next preserve raw coefficient diagnostic fixtures (which intentionally contain
unusable slots) while providing valid inactive sentinels for their execution
parity/statistical counterparts. Replace the removed legacy dispatch comparison
with an independent retained low-level kernel/stream oracle, preserving all
three-step RNG and sink assertions. Then run all wall-loss suites and dependent
execution/source-example tests. This unit is uncommitted; coagulation removal,
remaining GPU/lifecycle cleanup and final gates are still pending.

## Wall-loss parity and execution fixture follow-up

Separated raw coefficient diagnostic fixtures from native execution fixtures.
Raw diagnostic cases still exercise unusable geometry; execution/statistical
cases retain their active physical particles and use valid inactive sentinels.
The former native-versus-legacy three-step stream test now compares native
prepare/direct dispatch against an independently wired invocation of the retained
raw kernels for all four geometry/charge families. It preserves exact per-step
state/RNG comparisons and weighted sink accounting without a legacy admission
path. Statistical intervals and numerical tolerances remain unchanged.

Updated the public wall-loss feature contract to require explicit resolved
metadata, including zero-duration and selected-box calls. Migrated the real
checkpoint process fixture and logical-lane RNG fixtures (unit volume), and the
selected-lane adapter fixture's no-work row to valid inactive storage.

Validation:
- All wall-loss direct/parity/native suites: **305 passed, 126 skipped in
  1.06s**, `/tmp/m6-wall-native5.log`.
- Source examples: **225 passed, 2 skipped in 46.36s**,
  `/tmp/m6-wall-examples.log`, with `-Werror --no-cov`.
- Full-package mypy before the final execution fixture changes: **534 files
  passed**, `/tmp/m6-wall-mypy.log`.
- Full execution plus GPU process-sequence and population-boundary diagnostic:
  **20 failed, 1421 passed, 53 skipped in 84.97s**,
  `/tmp/m6-wall-consumers1.log`. Nine execution failures were untagged fixture
  construction and have since been repaired. Eleven GPU process-sequence rows
  remain: they expose both condensation and wall-loss legacy setup.
- Checkpoint, process-adapter and RNG-invariance follow-up: **108 passed,
  10 skipped in 2.79s**, `/tmp/m6-wall-execution-fixes.log`.
- Scoped Ruff/format and whitespace checks passed.

Next migrate `particula/gpu/tests/process_sequence_test.py` with explicit
population and condensation/nucleation species bindings, preserving independent
inventory, zero-time, invalid-preflight and aggregate wall-loss assertions.
Rerun broad dependent validation before committing wall-loss removal. No live
validation processes remain at this checkpoint.

## Native five-process sequence migration

The direct sequence now explicitly stores resolved counts converted from the
original density fixtures, with named identity condensation and nucleation maps
prepared after any intentional storage replacement. Density-based independent
inventory oracles normalize device counts by the current representative volume.
The original sparse/unequal fixtures remain available for standalone assertions.

Native coagulation requires equal positive counts in each box. The full-sequence
derivative now preserves total particle number, per-species mass, and signed
charge while forming equal weights; new independent tests enforce all three
invariants. Separate tests verify the original unequal weights reject without
mutation. The forced-collision call uses a large duration (1e30 s) consistent
with the corrected low number-density rate so its original collision-buffer and
no-more-pairs/RNG assertions still execute. This is a deterministic boundary
fixture, not a physical integration-duration recommendation. Nucleation's target
one-particle coefficient calculates target density rate before precursor
normalization, avoiding a roundoff-induced nonintegral count without changing
production validation or tolerances.

Validation:
- Process-sequence migration: **43 passed, 1 skipped in 0.62s**,
  `/tmp/m6-process-sequence-native3.log`, before adding four derivative/rejection
  cases.
- Final wall-loss/direct/parity/native plus process-sequence suites:
  **352 passed, 127 skipped in 1.68s**,
  `/tmp/m6-wall-final-focused.log`.
- Full-package mypy: **534 files passed**, `/tmp/m6-wall-mypy3.log`; fixed the
  fixture's Warp distribution-kind annotation after the prior run reported it.
- Standard strict docs: **built in 1.65s**, `/tmp/m6-wall-docs.log`.
- Ruff check/format passed eight changed Python files; whitespace passed.
- All **149 original assertion records** resolve to retained successor tests.

The full execution/process-sequence/population-boundary rerun is recorded below
when terminal; the wall-loss removal unit remains uncommitted until then.

Full dependent rerun finished: **1445 passed, 53 skipped in 80.14s**,
`/tmp/m6-wall-consumers2.log`. It covers the complete execution suite, the
five-process direct sequence, and population-boundary tests. The only subsequent
Python edit was the fixture's explicit `wp.int32(3)` annotation, covered by the
final focused run and full mypy above. Wall-loss removal is ready to commit.
GPU coagulation admission/selector cleanup is next; remaining source,
communication, transfer/checkpoint cleanup and final publication gates remain.

## GPU coagulation admission removal in progress

Direct/prepared coagulation now rejects untagged/discrete/PDF populations before
particle-array access. Native schema, equal-count physics, volume consistency,
and ownership checks apply unconditionally. Removed the prepared carrier's
`native_counts` compatibility flag; enqueue always launches the native
count-normalized selector. Added eight pre-array rejection cases. The retained
low-level legacy selector wrapper still needs its raw-test caller audit before
any deletion; this does not preserve untagged direct/prepared admission.

Saved 243 original coagulation test bodies in `gpu_coagulation_assertions.json`.
Migrated shared particle fixtures to explicit resolved metadata, and direct-step
launch spies to the native selector. Invalid-physics tests now upload valid
native storage before deliberately corrupting device fields where CPU transfer
would otherwise reject first. The additive-rate statistical fixture now stores
the same explicit physical volume it passes to the step. Numerical tolerances,
rate oracles, and RNG assertions are unchanged.

Validation checkpoint:
- Native suite: **172 passed, 164 skipped in 0.83s**,
  `/tmp/m6-coag-native2.log`. The first run's 33 failures all referenced the
  removed prepared compatibility flag; replacement asserts flag absence while
  retaining the full real prepared execution/conservation checks.
- Older direct/validation/stochastic diagnostic, latest bounded run:
  **6 failed, 80 passed in 1.71s**, `/tmp/m6-coag-consumers5.log`.
  Remaining next cases: combined multibox native physical-state fixture,
  charged/combined statistical fixtures with mismatching stored/passed volume,
  and charged invalid-physics cases rejected by CPU transfer before GPU entry.
  Earlier bounded migration logs 1–4 preserve intermediate failures.
- Scoped Ruff/format and whitespace passed before the final one-line fixture
  volume assignment; full mypy result is recorded in `/tmp/m6-coag-mypy1.log`.

The unit remains uncommitted. Continue the retained fixture migration, audit raw
selector callers, run complete coagulation and dependent execution suites, and
update relevant public documentation before committing. Remaining GPU source,
communication, transfer/checkpoint removal and final M6 gates remain pending.


## GPU coagulation retained-fixture migration follow-up

Completed the older direct/validation/stochastic fixture migration. Invalid
physics is uploaded as valid native state and then deliberately corrupted on
the device, so these tests still exercise GPU preflight rather than failing at
CPU transfer. Raw snapshots preserve invalid-state no-mutation assertions
without constructing a validated CPU carrier. Underflow and zero-mass active
slots now explicitly reject; valid inactive derivatives retain the charged cap
assertions. Explicit-volume fixtures store the same representative volume used
by their rate oracle and execution call.

The original mixed NPF/droplet unequal-count fixture remains, with a new
no-mutation rejection test. Its executable successor retains the particle radii,
masses and charges with one represented particle per slot, matching the existing
independent K/V pair-rate oracle. This is a distinct population, not an
inventory-preserving conversion. Existing sigma bounds, conservation checks,
raw diagnostic/production comparisons and RNG assertions remain unchanged.
Public coagulation guidance now states explicit-tag admission for direct and
prepared entry points.

Validation:
- Original full diagnostic: **56 failed, 505 passed, 19 skipped in 10.91s**,
  `/tmp/m6-coag-consumers-full1.log`.
- Invalid-physics/cap targeted checks: **19 passed, 416 deselected in 0.69s**,
  `/tmp/m6-coag-fixtures7.log`.
- Intermediate full diagnostic: **13 failed, 548 passed, 19 skipped in 7.78s**,
  `/tmp/m6-coag-consumers-full2.log`; all remaining failures were mixed-size
  population/volume fixtures.
- Final direct/validation/stochastic suites: **562 passed, 19 skipped in
  6.01s**, `/tmp/m6-coag-consumers-full3.log`.
- Full mypy: **534 files passed**, `/tmp/m6-coag-mypy3.log`, after widening the
  test-only unchanged-state helper annotation to accept raw snapshots.
- Scoped Ruff check/format passed all five changed Python files.

The dependent native/execution/process-sequence/population run is pending in
`/tmp/m6-coag-dependent1.log`. Raw selector compatibility remains to audit:
`brownian_coagulation_kernel` still passes zero to the shared `native_counts`
switch, whereas supported direct/prepared dispatch always uses the native
kernel. Resolve its raw diagnostic callers and assertion dispositions before
committing this unit. Final M6 publication and readiness gates remain pending.


Dependent run completed: **10 failed, 1607 passed, 217 skipped in 80.71s**,
`/tmp/m6-coag-dependent1.log`. All ten failures came from the same untagged
`coagulation_integration_test.py::_warp_bundle` construction. Added its explicit
resolved tag, retaining original equal counts, physical volume, inactive
sentinels, conservation oracles and RNG tests. The complete affected module then
passed: **12 passed, 2 skipped in 0.47s**,
`/tmp/m6-coag-integration2.log`. Ruff and whitespace checks passed. Standard
strict documentation built in **1.71s**, `/tmp/m6-coag-docs1.log`.
No validation process remains live at this checkpoint. The full dependent suite
has not yet been rerun after that final fixture tag. Raw-selector audit,
assertion-ledger reconciliation, and committing the coagulation unit are next.


## GPU coagulation selector closure

Audited all references to `brownian_coagulation_kernel`: its only executable
consumers were four raw-test launch sites. Migrated these to the retained
`_native_coagulation_kernel`, preserving charged multispecies property checks,
private sampler diagnostics, inactive-slot assertions, and bounded schedules.
Removed the obsolete wrapper and the shared selector's `native_counts` switch;
all mechanism schedules now use the admitted common raw count. No legacy
scheduling branch remains. Concrete-module export coverage now asserts removal
of the wrapper alongside positive native/internal imports. The similarly named
local teaching function in `docs/Theory/nvidia-warp/examples/fluids.md` is an
independent Brownian equation illustration, not an import of the removed API.

Reconciled **267 original test records** in `gpu_coagulation_assertions.json`,
including the execution integration and export suites. Each resolves to its
current successor; changed flag/selector assertions and changed invalid-state
admission have explicit dispositions. Original bodies remain preserved.

Validation after raw-selector removal:
- Direct/native/validation/stochastic/export suites: **757 passed, 183 skipped
  in 17.26s**, `/tmp/m6-coag-selector1.log`.
- Full-package mypy: **534 files passed**, `/tmp/m6-coag-mypy4.log`.
- Ruff check and format passed all seven changed Python files.
- Full execution/process-sequence/population rerun is pending in
  `/tmp/m6-coag-dependent2.log`; record its terminal result before committing.


Final dependent rerun passed: **1445 passed, 53 skipped in 88.56s**,
`/tmp/m6-coag-dependent2.log`. This covers the full execution suite plus
process-sequence and population-boundary contracts after raw-selector removal
and the final integration fixture tag. Source-example validation is running in
`/tmp/m6-coag-examples1.log` before the coagulation commit.

Source examples also passed: **225 passed, 2 skipped in 48.40s**, with
`-Werror --no-cov`, `/tmp/m6-coag-examples1.log`. Coagulation removal is ready
to commit. No validation processes remain live. Remaining M6 units begin with
GPU nucleation/exhaustion/communication compatibility, followed by transfer and
checkpoint admission, full installed integration, publication and closeout.


## GPU exhaustion removal in progress

Coagulation cleanup committed as `9708297bf`. Began the next logical unit with
exhaustion: removed the unreferenced legacy `_execute_resampling_step_gpu` and
`_execute_representative_volume_scaling_step_gpu` implementations plus their
orphaned `_any_nonzero_demand` kernel. Production resampling/scaling already
used the prepared path; it now requires explicit resolved metadata before any
payload access, always performs native population validation, and has no
prepared `native` compatibility flag.

Preserved original exhaustion test bodies in `gpu_exhaustion_assertions.json`.
The existing fixtures now expose resolved metadata separately from enumerated
device fields, retaining all numerical and raw snapshot assertions. Prepared
no-host-work tests assert the removed implementations are absent while keeping
allocation/readback guards and real enqueue outcomes. Added 16 pre-payload
rejection cases across direct/prepared resampling/scaling for missing, zero,
discrete, and PDF metadata. Module guidance now describes native admission.

Focused exhaustion/native/population suites: **68 passed, 8 skipped in 0.52s**,
`/tmp/m6-exhaustion2.log`. An initial 23 failures were snapshot helpers treating
the new metadata integer as a device array; placing metadata on the fixture
class preserves exact original payload enumeration. Full mypy: **534 files
passed**, `/tmp/m6-exhaustion-mypy1.log`.

Dependent nucleation/execution/process-sequence diagnostics are running in
`/tmp/m6-exhaustion-consumers1.log` (session 82502). Early failures identify
remaining untagged nucleation callers; do not commit exhaustion until dependent
paths are migrated and validated. Preserved **92 original nucleation test
bodies** in `gpu_nucleation_assertions.json` before changing nucleation.
Nucleation still has legacy standalone staged helpers and a prepared native
flag; its required species binding, count workspace, gas mapping and source
conservation must remain explicit during removal. No nucleation source has
been edited at this checkpoint.


Exhaustion dependent diagnostic finished: **45 failed, 1565 passed, 93 skipped
in 84.65s**, `/tmp/m6-exhaustion-consumers1.log`. Every failure is in the older
nucleation direct/staged or parity fixtures; execution and native nucleation
consumers passed in this combined run. The old fixtures reach native exhaustion
with untagged particles. Keep the rejection and migrate their construction,
species bindings and source/count oracles; do not restore untagged admission.
The two direct P4 staged cases also expose the old `_preflight_nucleation` /
`_orchestrate_nucleation_exhaustion` route, whose density workspace differs from
the native prepared count workspace. Preserve their scientific assertions when
removing or migrating that route. No validation processes remain live. Scoped
Ruff and whitespace checks pass; exhaustion is uncommitted pending this work.


## Native public nucleation admission and parity migration

The public/prepared nucleation path now requires explicit resolved metadata and
an explicit species binding before any payload access. Removed the prepared
`native` flag and both legacy enqueue branches: P4 always stages admitted raw
counts and P5 always uses the mapped native writer. Private staged legacy
helpers remain pending; this is not yet complete nucleation removal.

Migrated the independent parity fixture to explicit resolved counts, preserving
the original active density by multiplying by each representative volume. Added
complete GPU gas storage and explicit named identity bindings at each call,
after deliberate array replacements. The independent NumPy P2 demand/removal
oracle remains unchanged. Slot-count expectations now use demand times volume;
particle inventory comparisons normalize raw counts by current volume.

The scaling case now checks native density and extensive inventories separately:
halving representative volume and existing counts preserves particle/gas
density, halves extensive pre-existing inventory, and scales four admitted
source events to two. Source density remains four events per cubic metre after
scaling. The previous gas-density-halving expectation belonged to the retired
legacy path; the native assertions retain original tolerances and add an
explicit extensive combined-inventory check.

Extended metadata-only rejection coverage to both direct and prepared calls,
including missing/zero/discrete/PDF metadata and resolved metadata without a
binding. Updated the public nucleation guide to describe mandatory native
metadata and binding.

Validation:
- Native/parity suite after schema migration: **42 passed, 40 skipped, 1 failed**
  (legacy scaling-density expectation), `/tmp/m6-nucleation-native2.log`.
- Native/parity after independent oracle migration: **43 passed, 40 skipped in
  0.69s**, `/tmp/m6-nucleation-native3.log`.
- Combined native/parity/exhaustion suites with expanded rejection coverage:
  **94 passed, 48 skipped in 0.77s**, `/tmp/m6-source-native4.log`.
- Full mypy: **534 files passed**, `/tmp/m6-nucleation-mypy1.log`.

Next migrate `nucleation_test.py` and its private staged `_preflight_nucleation`,
`_plan_nucleation_demand`, `_orchestrate_nucleation_exhaustion` and
`_commit_nucleation_p5` callers. The standalone staged P4 workspace is still
legacy demand-density while native prepared P4 holds raw counts; preserve the
original staged scientific/rejection assertions through explicit native mapping
and count normalization. The unreferenced `_execute_nucleation_step_gpu` wrapper
and old P5 writer then become removal candidates. No source algorithms or test
tolerances have been changed by this public-path cleanup.


Post-public-removal dependent validation passed: **1431 passed, 53 skipped in
79.15s**, `/tmp/m6-nucleation-execution1.log`, covering the complete execution
suite and direct five-process sequence. Standard strict docs built in **1.59s**,
`/tmp/m6-nucleation-docs1.log`. Scoped Ruff/format and whitespace checks passed.
No validation processes remain live. Exhaustion/nucleation changes remain
uncommitted until standalone-phase migration and its retained tests pass.

Static ownership audit for the remaining source work: `_preflight_nucleation`
is called by standalone P2, the obsolete `_execute_nucleation_step_gpu` wrapper,
and native preparation. Standalone `_orchestrate_nucleation_exhaustion` and
`_commit_nucleation_p5` are called only by the obsolete wrapper in production;
the old `_commit_nucleation_p5_kernel` is called only by standalone P5.
`_plan_nucleation_demand` and the obsolete wrapper have no production callers.
Their test assertions are preserved in the 92-record nucleation ledger. There
are eight `_NucleationPreflight` constructor sites to update if its explicit
native binding is retained through staged handoffs. Do not infer identity maps
or restore untagged admission while migrating those tests.


## Standalone native source phases and obsolete-route removal

Migrated standalone P1/P2/P3/P4/P5 to explicit resolved metadata and species
bindings. `_NucleationPreflight` retains the binding across all eight return
paths. P2 uses full gas width; staged P4 now copies raw admitted counts into its
workspace; P5 uses the same mapped unit-count writer as prepared execution.
Removed the unreferenced `_execute_nucleation_step_gpu`, old density-slot P5
writer, orphaned workspace-copy kernel, and four kernel compatibility flags.
No untagged admission, native-mode flag, or legacy scheduling branch remains
in nucleation/exhaustion production code.

Migrated the existing phase tests with an explicit shared identity-map fixture
helper. Original validation, sidecar, ownership and invalid-handoff assertions
remain. Four legacy expected values changed with explicit native interpretation:
slots hold unit counts (their density is count/volume); a half-volume scaling
of four events produces two events at unchanged demand density; gas density
remains unchanged before source transfer. The renamed staged P5 gas test is
`test_p5_preserves_gas_density_after_volume_scaling` (formerly
`test_p5_scales_gas_before_finalized_source_removal`).

Added staged end-to-end native checks at V=0.25/1/4 with independent particle
and gas widths, nonidentity species mapping, preserved inert gas, explicit
per-event masses, and tight independent per-species density conservation.
Added rejected full-gas storage rebinding and finite-count/overflowed-density
checks without mutation. Updated native source/exhaustion guidance.

Validation:
- First phase migration: **182 passed, 40 skipped, 4 failed**, with the four
  legacy expectation mismatches above, `/tmp/m6-nucleation-stages2.log`.
- Phase/native/parity/boundary after expectation migration: **210 passed,
  40 skipped in 1.21s**, `/tmp/m6-nucleation-stages3.log`.
- Combined source/exhaustion/boundary after removing kernel mode switches:
  **254 passed, 48 skipped in 3.28s**, `/tmp/m6-source-stages5.log`.
- All nucleation suites including five new staged native cases: **191 passed,
  45 skipped in 1.13s**, `/tmp/m6-nucleation-stages6.log`.
- Full mypy: **535 files passed**, `/tmp/m6-source-mypy2.log`.

The broad GPU-kernel/execution/process/population run is active in
`/tmp/m6-source-dependent2.log` (session 47003). Final source review identified
that six early-gated standalone P1 returns still rely on the old fallback
protected-field list, which omits new mapping and vapor-pressure fields. Extend
those retained protection lists and add zero-rate alias coverage before commit;
public preparation already retains the complete protection list.

### Source/exhaustion commit closure

The broad run completed: **3495 passed, 648 skipped in 110.90s**,
`/tmp/m6-source-dependent2.log`. Extended all six early-return protected-field
lists to include species maps and gas vapor-pressure storage, with zero-rate
and active-rate alias rejection coverage. The final focused source/exhaustion
and population-boundary run passed **262 tests, 55 skipped in 1.33s**,
`/tmp/m6-source-final-focused.log`. The broad run precedes this final narrow
ownership extension; the final M6 integration gate remains required.

The 46 exhaustion and 92 nucleation original test records now identify their
validated native successors. Four numerical expectation groups changed only
for the explicit count/density and representative-volume semantics described
above; no tolerances were relaxed. Removed-wrapper assertions verify absence.

Final supporting checks: examples **225 passed, 2 skipped in 42.23s**
(`/tmp/m6-source-examples2.log`); mypy **535 files passed**
(`/tmp/m6-source-mypy3.log`); standard strict docs **1.44s**
(`/tmp/m6-source-docs2.log`); scoped Ruff, format and whitespace checks passed.
Communication, transfers/resident/checkpoint admission and final publication
readiness remain pending. These results do not close the full M6 gate.

## GPU communication native-only migration

Source/exhaustion closure was committed as `2b6a6df24`. Communication now
requires explicit resolved metadata at direct volume/particle entry points,
resident wrappers (before payload access), and prepared binding setup for
particle transport or prescribed volume changes. Removed the planner's native
mode flag and both prepared-helper mode flags, particle-density transport
credits, and particle-density volume writers. Gas transport and its independent
width/accounting remain unchanged. Prepared enqueue still uses pinned arrays.

The new communication ledger retains 62 original tests. Fixtures now explicitly
label raw particle weights as counts; count-weighted number, species mass and
charge conservation replace legacy density-times-volume expectations. These
fixtures are native count derivatives, not claims that unchanged raw weights
represent the old density inventory. The independent NumPy particle-density
oracle is retained unchanged (renamed `_particle_density_oracle`); a test-only
normalization wrapper converts counts to density and converts final weights,
debits, credits and source requests back to counts. No tolerance changed.
The old scaled-particle underflow/overflow cases now also put that value in
gas density, preserving protection of the retained scaled-density writer.

Initial focused run: **84 passed, 12 skipped, 13 failed**
(`/tmp/m6-communication1.log`), identifying legacy expectations and the removed
raw-kernel mode argument. After native expectation migration: **97 passed,
12 skipped in 0.42s** (`/tmp/m6-communication2.log`). Added resident-wrapper
missing/zero/PMF/PDF metadata rejection coverage; native boundary suite:
**23 passed, 11 skipped in 0.39s** (`/tmp/m6-communication-boundary.log`).
Mypy passed all **535 files** (`/tmp/m6-communication-mypy1.log`); scoped
Ruff/format and standard strict docs passed (`/tmp/m6-communication-docs1.log`).
The full execution/communication/example dependent run remains active in
`/tmp/m6-communication-dependent1.log` (session 28020); do not commit until its
terminal result and assertion reconciliation are recorded.

The first dependent run completed with **1484 passed, 64 skipped, 8 failed in
82.03s**. All failures were untagged resident fixture admission, not numerical
mismatches. Added explicit resolved fixture construction (tag, resident
metadata, independent gas dimension) for communication/restart cases; shared
helpers retain their other callers' existing default until the resident
lifecycle migration. Updated native count inventory expectations and removed
physical-volume scaling only from the particle branch of the transport-loop
oracle. Both independent particle-density oracles are AST-identical to their
original bodies after renaming; explicit normalization wrappers preserve them.
The ledger now contains **83 original tests** including resident communication,
transport-loop and restart-loop assertions. Their focused run passed **28 tests
in 0.94s**, `/tmp/m6-communication-dependent2.log`. Full dependent rerun is
active in `/tmp/m6-communication-dependent3.log` (session 7752).

### Communication commit closure

Full dependent rerun passed: **1500 passed, 64 skipped in 84.25s**,
`/tmp/m6-communication-dependent3.log`. All source examples passed:
**225 passed, 2 skipped in 46.24s**, `/tmp/m6-communication-examples1.log`.
Mypy again passed **535 files**, `/tmp/m6-communication-mypy2.log`.
All 83 original assertion records have validated native successors. Scoped
Ruff, format and whitespace checks passed. No tests or numerical tolerances
were removed. Standard strict documentation passed earlier at the same
production/doc revision. No validation processes remain live for this unit.

Next unit: explicit population metadata at CPU-Warp transfers and resident
construction; remove the unused `require_legacy_particle_metadata` guard;
reject untagged schema-v5 restart before setup/upload, preserving tagged v5.
The remaining production audit also identifies ambiguous fallback gas width,
resident diagnostics and construction compatibility, and stale condensation/
coagulation docstrings. Shared-mask conversion is independent retained science.
Final published documentation/notebook/readiness gates remain open.

## Explicit transfer, resident and checkpoint metadata (in progress)

Communication closure committed as `6412a19d9`. CPU-Warp upload and restore
now reject missing/untagged provenance before backend setup or readback;
explicit PMF and resolved tags retain raw values, PDF remains unsupported.
Removed the unused `require_legacy_particle_metadata` guard. Zero remains only
an invalid/unset low-level tag sentinel so each owning consumer can report its
own unsupported-population admission error; it is no longer interpreted as a
transferable population.

Resident dimensions require explicit independent gas width; resident metadata
rejects omitted/unknown population kind. CPU resident preflight checks tag
admission before payload inspection, preserves the existing field-specific
schema rejection order, then validates physical interpretation before upload.
Schema v5 is unchanged: untagged or missing semantics now reject before restart
setup/upload, and tagged native snapshots retain their canonical byte authority.
Removed the diagnostics native/legacy mode flag and unmapped mass reducer;
all total-mass diagnostics require explicit complete material-to-gas maps.

The boundary assertion ledger currently retains 357 original test bodies.
Shared resident fixtures now declare tags and gas width. Legacy unequal-volume
diagnostics density fixtures are converted to counts using their original
volumes, preserving the original independent number, mass and residual oracles.
A PMF tagging attempt was corrected because non-unit-volume PMF is unsupported.
No tolerance or scientific oracle was loosened. Raw conversion characterization
retains PMF unit-volume and resolved V=0.25/1/4 transfer checks; old PDF and
non-unit-volume PMF cases now assert the required unsupported admission.

Initial collection exposed one generated fixture syntax error, fixed before
execution (`/tmp/m6-boundary1.log`). First focused run: **445 passed, 26 skipped,
36 failed in 42.76s** (`/tmp/m6-boundary2.log`); failures identified old
untagged fixture/protocol expectations. After explicit fixture metadata and
preserved schema ordering: **487 passed, 25 skipped in 42.18s**
(`/tmp/m6-boundary3.log`). Subsequent transfer/diagnostics/checkpoint selector:
**221 passed, 25 skipped, 2 failed** (unsupported non-unit PMF cases), then
**223 passed, 25 skipped in 1.07s**, `/tmp/m6-boundary5.log`.

Full GPU/execution run is active in `/tmp/m6-boundary-broad1.log` (session
57174); second mypy is active/completion unpolled in
`/tmp/m6-boundary-mypy2.log` (session 71473). Changes are uncommitted pending
full consumer migration, rejection tests and final validation.

### Explicit boundary validation closure

Recovered terminal results for this revision: full GPU/execution run passed
**4182 passed, 679 skipped in 100.15s**, `/tmp/m6-boundary-broad2.log`;
source examples passed **225 passed, 2 skipped in 45.38s**,
`/tmp/m6-boundary-examples1.log`. The preceding broad run exposed untagged
dilution and precision-study fixtures; those now declare their population
meaning. Independent-gas-width dilution fixtures resize all gas fields
consistently. Focused consumers passed **246 passed, 17 skipped in 0.67s**,
`/tmp/m6-boundary-consumers2.log`. No numerical tolerance was loosened.

The boundary ledger contains 429 original test bodies, each with an existing
native successor. It includes explicit unsupported-transfer dispositions and
raw-value precision derivatives; the latter do not claim process equivalence
to former density interpretations. Mypy passed 535 files
(`/tmp/m6-boundary-mypy3.log`); package-wide Ruff and format checks and
`git diff --check` passed. Standard strict docs passed in 1.63s
(`/tmp/m6-boundary-docs1.log`). Final publication and readiness gates remain
pending.

Opt-in benchmark collection passed (31 tests in 0.39s,
`/tmp/m6-benchmark-collection1.log`), but this is collection evidence only.
Two benchmark fixture constructors still omit explicit population metadata;
the coagulation workload also uses unsupported unequal active weights.
Migrate these consumers as a separate integration prerequisite, preserving
historical fixture provenance and identifying any changed workload. Do not
interpret collection or optional CUDA skips as measured benchmark evidence.

### Next integration audit findings

Boundary unit committed as `72f179eeb`. Inspection of the opt-in scaling
benchmarks found additional migration requirements beyond constructor tags:
`test_condensation_scaling` lacks an explicit species binding; its CPU helper
performs one uncoupled mass update while the supported direct GPU path uses
four inventory-finalized gas-coupled substeps. Its constant thermodynamics
parameters are zero despite a separately supplied nonzero vapor-pressure
fixture. These timings must not be presented as equivalent-method speedups
without an explicitly matched comparator and regression evidence.
`test_coagulation_scaling` consumes mixed-scale unequal weights, while native
coagulation requires equal active counts. Retain the original mixed-scale
assertions as unsupported-input evidence and introduce a separately identified
native workload; do not silently relabel historical timings or overwrite their
meaning. Check the CPU collision helper's physical weighting when choosing
the matched native workload.

Full publication audit started at this commit using
`.venv/bin/mkdocs build --config-file mkdocs-full.yml --strict`, output
`/tmp/m6-full-publication-audit1.log`, live session 18918. Repeated unclosed
Div rendering warnings are already observed; terminal result remains pending.
This is an audit run, not a passing readiness gate. Poll the existing handle
before starting another publication build.

### Native benchmark migration (uncommitted)

Added explicit resolved tags to raw benchmark fixtures, retaining their values.
The historical unequal-weight mixed-radius fixture remains and has a real Warp
CPU rejection check. Timed coagulation now selects a separately identified
unit-count derivative preserving masses, volumes and material layout; its
artifact metadata explicitly excludes historical timing comparability. Real
Warp CPU execution preserves per-material inventory at rtol=1e-12, atol=1e-30.
The CPU coagulation comparator still needs review: its arithmetic effective
material density reconstruction must not be treated as validated conservation
or equivalent-method timing evidence.

Condensation now prepares an explicit identity species binding, uses a constant
800 Pa thermodynamics fixture consistently across boxes and CPU/GPU paths, and
identifies the new workload separately. A new NumPy comparator performs four
inventory-limited gas-coupled substeps with resolved count/volume normalization
and final slot retirement. The original single-proposal helper and its tests
remain as independent raw-proposal characterization. Real Warp CPU tests cover
uptake/evaporation at 1e-6 and 0.5 seconds, compare particle and gas outputs, and
check extensive species conservation at unchanged tight tolerances. Initial
zero-vapor-pressure test cases were invalid under the existing positive-constant
contract; replaced with positive 1e-8 Pa uptake inputs, without changing the
validator. Final helper run: **92 passed in 0.89s**,
`/tmp/m6-benchmark-native5.log`. Scoped Ruff/format, mypy and whitespace checks
pass. The benchmark assertion ledger preserves 79 original test functions.
This is correctness/helper evidence, not CUDA timing or final benchmark
qualification. No production numerical kernel changed.

Full publication audit session 18918 is terminal: exit 0, build completed in
**214.36 seconds**. Its log contains **711 warning/error lines**, including
unclosed Div warnings from notebook conversion. Therefore process success is
not warning-free publication evidence. Repair the emitted warnings before
claiming the full publication gate. Log: `/tmp/m6-full-publication-audit1.log`.

### Benchmark correctness closure

Added a distinct native CPU coagulation comparator using the existing grid
sampler and direct per-material/charge accumulation for unit-count collisions.
The historical density-reconstruction helper and every original test remain
for characterization; timed calls use the conservative native comparator.
Forced empty, single and repeated-destination collision tests preserve material
inventory and charge and verify exact source retirement. Metadata names the
CPU interpolated-grid and GPU direct-pair algorithms and explicitly disclaims
algorithm parity. Their measured timing ratio alone cannot qualify statistical
or trajectory equivalence. Historical workload timings remain noncomparable.

The complete helper suite passed **95 tests in 0.92s**,
`/tmp/m6-benchmark-native6.log`; package mypy passed **535 files**,
`/tmp/m6-benchmark-mypy1.log`; scoped Ruff and formatting passed. This closes
native benchmark consumer correctness for the covered workloads; CUDA timing
is unmeasured, not a required performance claim. Original assertion ledger
successors all resolve. No numerical kernel or tolerance changed.

Publication warning diagnosis: all 711 lines are Pandoc unclosed-Div warnings.
mkdocs-jupyter calls Jupytext on every Markdown page before checking include
patterns, and generated `::: particula...` API directives trigger the parser.
A publication hook now keeps ordinary Markdown in MkDocs/mkdocstrings and
leaves Python/ipynb notebook eligibility unchanged. No Markdown notebooks were
found in the supported docs. The routing regression passed (1 test, 0.02s).
Full strict publication rerun is active in session 84473, log
`/tmp/m6-full-publication-routing1.log`; poll this handle before retrying.

### Warning-free publication routing closure

Full strict publication passed with exit 0 in **49.06 seconds** and **zero
warning/error lines**, `/tmp/m6-full-publication-routing1.log`. Generated
Aerosol and ParticleData API HTML contains rendered API content rather than
literal directives; the activity tutorial notebook HTML remains rendered.
The ordinary strict build also passed in **1.56 seconds**,
`/tmp/m6-standard-routing1.log`. The infrastructure suite including the routing
regression passed **132 tests in 25.74s**, `/tmp/m6-scripts-routing1.log`.
The hook corrects notebook eligibility before parsing; it suppresses no warning
and removes no supported API or Python/ipynb notebook from publication.
All three runs are terminal. Later documentation edits require renewed final
publication validation; these results do not execute notebook workloads.

Untargeted normal-coverage repository runner started at benchmark commit
`48771293c` with the routing-only docs changes present:
`PATH="$PWD/.venv/bin:$PATH" WARP_CACHE_PATH=/tmp/particula-m6-warp
.opencode/tools/run_pytest.py`. Session 88972 remains confirmed live; buffered
output is `/tmp/m6-full-runner1.log`. Poll this session before restarting.
