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
  builder tests, Warp type tests, and conversion tests. Ensure single-box shapes
  keep a leading dimension.
- **P4 Handoff Documentation:** Run documentation/link validation and verify all
  sibling tracks are named with explicit handoff decisions.

## Suggested Commands

```bash
pytest particula/particles/tests/particle_data_test.py \
  particula/gas/tests/gas_data_test.py \
  particula/gpu/tests/warp_types_test.py \
  particula/gpu/tests/conversion_test.py
```

If docs tooling is available for changed files, also run the repository's
markdown/link validation workflow or the closest available documentation check.

## Acceptance Evidence

- Decision record exists and is linked from roadmap/migration docs.
- Shape tables cover CPU and GPU containers.
- Tests or documented evidence show that current code matches the published
  schema inventory.
