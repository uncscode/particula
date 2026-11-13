# Phase Details

Final validation includes E9 appendix D1–D2: distribution metadata copies,
transfers and checkpoint interpretation; radius-PDF integrals; corrected
CPU/direct/prepared/resident number and per-species mass conservation at
V=0.25/1/4. Legacy ambiguous checkpoints must not silently acquire new semantics.

Entry: E9-M5 and all its predecessors are completed and validated, not merely
drafted. P1 -> P2 -> P3 -> P4 -> P5 -> P6 acceptance gates remain ordered.
The post-M5 commit sequence below moves consumer cleanup prerequisites ahead
of definition deletion so that intermediate commits retain coherent imports.
No standalone unit-testing phase: every removal/function change includes its
tests. P5 is cross-layer integration, P6 final documentation/readiness.

Checkboxes below represent final acceptance, not implementation progress.
Implementation validation and publication are complete. They remain open
for the named maintainer acceptance review. No separate child issues are required.

## Accepted single-PR implementation sequence

The detailed maintainer-approved plan supersedes the earlier ten-commit scope.
One PR, fifteen logical review units, each with affected fixtures and
regressions. Implementation splits larger units into serial prerequisite commits
to preserve scientific assertion provenance and coherent imports:

1. Finalize removal/assertion-retention inventories and baseline evidence.
2. Remove CPU condensation facade dispatch.
3. Remove CPU coagulation and wall-loss facade dispatch.
4. Retire legacy aggregate, runnable, dilution/nucleation and CPU adapter paths.
5. Remove particle representation APIs, bridges and obsolete construction.
6. Remove gas/atmosphere APIs, bridges and obsolete construction.
7. Remove untagged GPU condensation execution.
8. Remove untagged GPU coagulation and wall-loss execution.
9. Remove remaining untagged GPU source/exhaustion/communication paths.
10. Require explicit metadata at transfers/resident lifecycle; reject untagged
    v5 checkpoints before setup/upload while retaining tagged v5 restart.
11. Close remaining API seams and validate native/installed-wheel integration.
12. Finalize migration guidance and execute all supported M5 example workloads.
13. Repair scientific API docstrings without executable AST changes.
14. Repair remaining publication/rendering warnings; pass both strict builds.
15. Record final validation/readiness evidence and create the single PR.

Consumers and their tests migrate before definition deletion. Keep coherent
imports and collection at each commit. No new compatibility shim, numerical
rebaseline, inferred population meaning, schema bump or release publication.
Raw CPU scratch material may remain incompletely tagged; supported execution
and population interpretation may not. Scientific algorithms, explicit
transfers, native metadata, RNG continuation and device ownership are retained.

Every enduring old test assertion must have an explicit native successor or
justified facade-only disposition before deletion. Record final executable
revision separately from evidence-only edits. Both strict builds and the
full published notebook workloads are mandatory, not optional follow-ups.

- [ ] **E9-M6-P1: Audit final legacy consumers and temporary compatibility**
  - Issue: #1607 | Size: S | Status: Audit reconciled after generated outputs; final review pending
  - Goal: Approve an exhaustive deletion and scientific-retention inventory.
  - Work: Reconcile M1–M5 ledgers with searches across production, tests, docs,
    Python/notebook sources, root snippets and API navigation. Seed names and
    paths are in scope/appendix; expand aliases, qualified strings, builders,
    warnings and temporary seam names from actual consumers. Classify each
    match as removal, retained science/transfer, negative boundary test, or
    explicitly labeled history. Identify stale generated notebook outputs.
  - Tests: Collect and run affected native assertion suites with `--no-cov`;
    map each old scientific test to the migrated case and retained oracle.
    Any audit helper added needs adjacent unit tests in this phase.
  - Gate: No unmigrated supported consumer; all temporary seams have removal
    owners; approved retained-test and public-import inventory. Gaps reopen
    upstream gates before any facade is deleted.

- [ ] **E9-M6-P2: Remove particle facades and obsolete construction with boundary tests**
  - Issue: #1607 | Size: S | Status: Implemented and integration-tested; final review pending
  - Goal: Remove ParticleRepresentation without losing native initialization.
  - Work: Delete obsolete particle representation definitions/builders/factories
    and facade-specific conversions per P1. Update package/root imports,
    annotations and facade-only fixtures atomically. Retain useful native
    presets and representation-changing scientific algorithms where required.
  - Tests: Adjacent particle tests plus proposed
    `particula/tests/legacy_api_removal_test.py` cover removed attributes,
    root/package/former-module imports and positive native construction.
    Preserve normalization at non-unit volumes, distribution behavior,
    zero-particle handling, identity and rejection cases in native suites.
  - Gate: Particle import graph remains usable, native behavior passes, every
    deleted scientific-looking assertion has a reviewed retained disposition.

- [ ] **E9-M6-P3: Remove gas and atmosphere facades with boundary tests**
  - Issue: #1607 | Size: S | Status: Implemented and integration-tested; final review pending
  - Goal: Remove GasSpecies and Atmosphere without splitting gas authority.
  - Work: Delete obsolete species/atmosphere modules, builders/factories and
    bridges plus package/root exports and obsolete aggregate branches. Keep
    native GasData/EnvironmentData builders, vapor-pressure science and M2
    construction. Remove facade-only tests after assertion-level review.
  - Tests: Adjacent gas and Aerosol suites plus removed-boundary tests cover
    former concrete imports, mixed/all-nonpartitioning gas, ordered species,
    environment ownership, native triple construction/replacement and rejection.
    Preserve vapor-pressure/thermodynamic numerical assertions.
  - Gate: Native aggregate works without either deleted class; all declared
    gas categories and environmental fields retain their scientific meaning.

- [ ] **E9-M6-P4: Clear temporary compatibility branches and messages with regressions**
  - Issue: #1607 | Size: S | Status: Compatibility removal implemented and regression-tested
  - Goal: Close the entire upstream temporary compatibility inventory.
  - Work: Remove residual adapters, type branches, alias lookups, warning/error
    text and facade-only tests in migrated strategies, runnable/aggregate and
    CPU adapters. P2/P3 already remove all references necessary for coherent
    imports; this phase cannot defer broken imports. Preserve unrelated domain
    warnings and supported GPU conversion helpers. Repeat all P1 searches.
  - Tests: Co-located native strategy, dilution/nucleation, runnable and CPU
    adapter tests verify real supported paths, rejection ordering and failure
    semantics. Add focused assertions for any changed message contract.
  - Gate: Zero remaining temporary seams and executable legacy consumers;
    every remaining old-name match has an approved bounded disposition.

- [ ] **E9-M6-P5: Validate native integration and retained CPU-Warp contracts**
  - Issue: #1607 | Size: S | Status: Cross-layer package and complete example validation passed; final review pending
  - Goal: Demonstrate post-removal cross-layer scientific and ownership behavior.
  - Work: Extend native integration fixtures only where audit reveals a missing
    cross-layer scenario; no new physics or GPU interface. Exercise flat
    construction/replacement through composed CPU execution and explicit Warp
    transfer, then retained direct/resident boundaries independently.
  - Tests: Run testing_requirements command groups, including BOTH wall-loss
    suites, removed API tests, native non-unit-volume/mixed-species regressions,
    transfer/export/ownership and resident lifecycle suites. Warp CPU baseline
    applies when installed; CUDA is optional pass-or-clean-skip evidence only.
  - Gate: All required assertion groups and untargeted repository runner pass;
    preserve literal outputs and record unavailable evidence as blocking.

- [ ] **E9-M6-P6: Update development documentation**
  - Issue: #1607 | Size: S | Status: Documentation, output review and both strict publication builds passed; maintainer review pending
  - Goal: Publish evidence-backed v0.3.0 migration/readiness guidance.
  - Work: Reconcile migration/API pages, roadmap, root/developer instructions
    and M5's full supported example ledger against deleted APIs. Proposed
    `docs/Features/Roadmap/data-native-v030-closeout.md` records removals,
    replacement APIs, validation and unresolved limitations. Label historical
    references explicitly; do not present former construction as runnable.
  - Tests: Rerun supported scripts and affected notebook pairs after Python
    lint/sync/execute; retain both files. Run documentation contract tests,
    `mkdocs build --strict`, lint/type checks and the untargeted runner at the
    final revision. Repeat consumer/compatibility audit after generated outputs.
  - Gate: Every required readiness row passes, narrow optional skips explained,
    no open removal seam, and named maintainer approval. Missing validation
    means BLOCKED, never inferred success or automatic release publication.
