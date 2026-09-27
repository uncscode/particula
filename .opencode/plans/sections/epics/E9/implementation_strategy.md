# Implementation Strategy

## Architecture and ownership

Use one flat aggregate of existing `ParticleData`, `GasData` and
`EnvironmentData`. Preserve container schemas and direct field authority.
Processes own activity, surface, vapor-pressure and other physics strategies;
do not relocate them onto a replacement facade. Reuse native condensation
configuration (`condensation_strategies.py:294–364`) and coagulation's
single-box helpers (`coagulation_strategy_abc.py:49–111`) rather than rewriting
physics. Preserve `RunnableABC.__or__` and sequence order (`runnable.py:108–137`).

M1 must freeze an explicit table of units, shapes, raw versus normalized
concentrations, copy/view/mutation rules and ordered mapping between gas
species, particle mass lanes, environment saturation-ratio lanes and process
configuration. `GasData.partitioning` distinguishes gas categories in unified
storage; nonpartitioning gas must still dilute. Do not silently sort, drop or
reinterpret species. Final mapping/API spelling remains an M1 decision.

Whole-container getters return held objects. Individual replacement validates
the candidate against the other held containers. Coordinated replacement
validates the entire candidate triple before assigning anything. Successful
setters retain supplied identity; rejection preserves old references and data.
Do not call coercing constructors to validate already-held objects in a way
that mutates them. Direct mutation is allowed but processes must revalidate
their required physical/schema invariants. Replacing CPU containers does not
rebind retained GPU arrays, prepared execution state or resident sessions.

## Migration sequence

1. **M1:** Specify contracts, inventory needed helpers and implement them with
   adjacent tests. Audit every concentration accessor by distribution type and
   test non-unit volumes: raw field access is not automatically equivalent to
   `ParticleRepresentation.get_concentration()`.
2. **M2:** Introduce flat construction and atomic replacement, migrate useful
   presets/builders, and add identity/rejection tests. Inventory any temporary
   adapter needed to keep unmigrated consumers running; give it a deletion
   owner/gate and no permanent public promise.
3. **M3:** Migrate native condensation/coagulation and wall-loss scientific
   paths with process-owned strategies. Preserve geometry, distribution,
   nonnegative concentration, conservation and single-box validation.
4. **M4:** Migrate orchestration and CPU adapters. Dilution must retain
   all-state preflight and documented restoration on setter failure, covering
   both gas categories. Nucleation retains identity, current-gas sequential
   substeps and atomicity per attempted substep, not whole-call rollback.
5. **M5:** Execute the full supported example inventory, not only a new quick
   start. Update paired Python sources first, synchronize and execute notebooks,
   retain both files, and publish ownership and migration guidance.
6. **M6:** Audit imports, exports, builders, type annotations, compatibility
   branches, warnings, executable docs and tests. Delete obsolete APIs only
   after replacement coverage and consumers are verified. Preserve unrelated
   warnings and CPU↔Warp helpers. Finish integrated validation and release docs.

## Testing requirements

Repository policy controls actual coverage settings; the generic template's
fixed coverage example does not add a new threshold to this epic.

1. Test coverage thresholds must NEVER be lowered.
2. Each phase must include self-contained tests.
3. Tests are committed in the same PR as the implementation.
4. Test files use `*_test.py` suffix in module-level `tests/` directories.
5. Use repository-configured full-package coverage via the untargeted runner;
   do not invent local coverage thresholds or focused coverage evidence.

For every owning track, run focused direct-pytest assertion checks first,
then `.opencode/tools/run_pytest.py` without a target. Preserve independent
scientific oracles and explicit existing tolerances: test per-species outputs
and concentration-weighted inventories, not just aggregate totals or
per-particle masses. Migrate enduring assertions rather than deleting tests
because fixtures mention legacy objects. Add multi-box rejection, non-unit
volume, alignment mismatch, direct-mutation revalidation and identity checks.

Representative focused commands (refine file inventory per child diff):

```bash
pytest particula/particles/tests/ particula/gas/tests/ -q --no-cov
pytest particula/dynamics/condensation/tests/ particula/dynamics/coagulation/ -q --no-cov
pytest particula/dynamics/tests/wall_loss_strategies_test.py -q --no-cov
pytest particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q --no-cov
pytest particula/dynamics/tests/dilution_test.py particula/dynamics/tests/nucleation_runnable_test.py -q --no-cov
pytest particula/execution/tests/ -q --no-cov
pytest particula/gpu/tests/conversion_test.py particula/gpu/tests/kernel_exports_test.py -q --no-cov
.opencode/tools/run_pytest.py
.opencode/tools/run_linters.py
mypy particula/ --ignore-missing-imports
python3 .opencode/tools/build_mkdocs.py --validate-only --strict
mkdocs build --strict
```

The adjacent CPU wall-loss suite is excluded from normal recursive collection
and therefore must be invoked explicitly. Add applicable example/export and
release-documentation tests from the testing guide. Do not add local `-Werror`
or raw comprehensive coverage overrides. Warp CPU is the installed-Warp
baseline; optional CUDA must pass or cleanly skip without fallback. No new
performance evidence is required or claimed.

Record literal commands, revision, date, outcomes and availability at each
handoff. Required failures/unavailable commands keep the gate open. Commands
above are a future validation plan, not results from this drafting run.
