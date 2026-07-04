# Testing Strategy

## Co-Located Testing Policy

This feature is primarily documentation and decision-record work. Any test or
validation update should ship with the phase that introduces the checkable
artifact. Do not create a separate testing-only phase.

## Phase Validation

- **P1 Inventory:** Compare the inventory against existing tests in
  `particula/particles/tests/`, `particula/gas/tests/`, and
  `particula/gpu/tests/`. Run docs/link validation if the inventory is committed
  as a new document.
- **P2 Ownership Decisions:** Review the decision table for every current field
  plus future environment fields required by E2-F2/E2-F3. Add lightweight tests
  only if examples or snippets are included.
- **P3 Shape Conventions:** Verify documented shapes against constructor tests,
  builder tests, Warp type tests, conversion tests, and the current
  condensation/coagulation boundary evidence. Ensure single-box shapes keep a
  leading dimension, the roadmap subsection remains the single source of truth,
  and migration/discoverability links point to the canonical shape anchor.
- **P4 Handoff Documentation:** Run documentation/link validation and verify all
  sibling tracks are named with explicit handoff decisions.

## Suggested Commands

```bash
pytest particula/particles/tests/particle_data_test.py \
  particula/gas/tests/gas_data_test.py \
  particula/gas/tests/gas_data_builder_test.py \
  particula/gpu/tests/warp_types_test.py \
  particula/gpu/tests/conversion_test.py \
  particula/dynamics/condensation/tests/condensation_strategies_test.py \
  particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py
```

If docs tooling is available for changed files, also run the repository's
markdown/link validation workflow or the closest available documentation check.

For issue #1185 docs-first follow-up work, the fallback docs validation path is
to verify that `data-oriented-gpu.md#canonical-shape-conventions-for-container-workflows`
exists, that `docs/Features/particle-data-migration.md` links directly to that
anchor, and that `docs/Features/Roadmap/index.md` points readers to that
subsection without creating a second source of truth.

## Acceptance Evidence

- Decision record exists and is linked from roadmap/migration docs.
- Shape tables cover CPU and GPU containers.
- Tests or documented evidence show that current code matches the published
  schema inventory.

## Current Shipped Evidence

- Issue #1183 shipped as a documentation-only update.
- The published inventory in `docs/Features/Roadmap/data-oriented-gpu.md`
  points each schema claim back to source or existing tests in
  `particula/particles/tests/`, `particula/gas/tests/`, and
  `particula/gpu/tests/`.
- Issue #1184 shipped as a documentation-only ownership-decision update with
  links from `docs/Features/Roadmap/index.md` and
  `docs/Features/particle-data-migration.md`.
- The current shipped evidence for P2 is the published roadmap section and its
  discoverability links rather than new automated tests.
- Issue #1184 fix-pass validation re-ran the targeted regression evidence tests
  in `particula/particles/tests/particle_data_test.py`,
  `particula/gas/tests/gas_data_test.py`,
  `particula/gpu/tests/warp_types_test.py`, and
  `particula/gpu/tests/conversion_test.py`.
- In the current ADW fix environment, docs validation evidence was recorded by
  confirming that `docs/Features/Roadmap/data-oriented-gpu.md` still exposes
  the ownership anchor and that `docs/Features/Roadmap/index.md` plus
  `docs/Features/particle-data-migration.md` still link to
  `#authoritative-field-ownership-decisions`.
- Issue #1185 shipped as a documentation-only canonical-shape update in
  `docs/Features/Roadmap/data-oriented-gpu.md`.
- The shipped P3 evidence is the roadmap canonical shape subsection, its direct
  migration-guide anchor link, the roadmap-index discoverability wording, and
  existing test/source evidence for shapes plus current single-box
  condensation/coagulation execution boundaries.
- Issue #1186 shipped as a documentation-only downstream handoff publication
  pass touching `docs/Features/Roadmap/data-oriented-gpu.md`,
  `docs/Features/Roadmap/index.md`,
  `docs/Features/particle-data-migration.md`, and `docs/index.md`.
- The current P4 evidence is the published roadmap handoff map for `E2-F2`
  through `E2-F9`, the direct handoff-anchor links from roadmap/migration/top-
  level docs entry points, and the roadmap publication note recording
  `python3 .opencode/tools/build_mkdocs.py --validate-only --strict` as the
  docs validation command.
