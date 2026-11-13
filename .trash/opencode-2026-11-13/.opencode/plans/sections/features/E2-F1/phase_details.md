# Phase Details

- [x] **E2-F1-P1:** Inventory current CPU and GPU container schemas
  - Issue: #1183 | Size: XS | Status: Completed
  - Goal: Produce a concise inventory of existing container fields, shapes,
    dtypes, mutability, and round-trip behavior.
  - Shipped: a docs-only authoritative inventory in
    `docs/Features/Roadmap/data-oriented-gpu.md` covering `ParticleData`,
    `GasData`, `WarpParticleData`, and `WarpGasData`, plus the current lossy
    gas round-trip caveats.
  - Tests: No new tests added; the inventory cites existing container and
    conversion tests as executable evidence.

- [x] **E2-F1-P2:** Record authoritative field ownership decisions
  - Issue: #1184 | Size: S | Status: Completed
  - Goal: Decide which container owns each particle, gas, and environment field,
    including shared/per-box fields and lossy GPU transfer semantics.
  - Files: schema decision record, `docs/Features/Roadmap/data-oriented-gpu.md`,
    and links from relevant feature docs.
  - Shipped: an `Authoritative field ownership decisions` section in
    `docs/Features/Roadmap/data-oriented-gpu.md`, direct roadmap and migration
    links to that section, and top-level docs links that improve discoverability.
  - Tests: Docs-first update; no new automated tests added in this phase.
  - Fix-pass evidence: targeted regression evidence tests were re-run in
    `particula/particles/tests/particle_data_test.py`,
    `particula/gas/tests/gas_data_test.py`,
    `particula/gpu/tests/warp_types_test.py`, and
    `particula/gpu/tests/conversion_test.py`.
  - Docs validation: fallback anchor/link verification confirmed that
    `docs/Features/Roadmap/data-oriented-gpu.md` still exposes
    `#authoritative-field-ownership-decisions` and that
    `docs/Features/Roadmap/index.md` plus
    `docs/Features/particle-data-migration.md` still link to it.

- [x] **E2-F1-P3:** Document shape conventions across container workflows
  - Issue: #1185 | Size: S | Status: Completed
  - Goal: Publish shape rules for single-box, multi-box, particle-resolved, and
    binned workflows across CPU and GPU containers.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/particle-data-migration.md`, and
    `docs/Features/Roadmap/index.md`.
  - Shipped: the canonical `Canonical shape conventions for container
    workflows` subsection is published in
    `docs/Features/Roadmap/data-oriented-gpu.md` with workflow rules, concrete
    example shapes, explicit field tables for `ParticleData`, `GasData`, future
    `EnvironmentData`, `WarpParticleData`, and `WarpGasData`, plus a caution
    that current CPU condensation and coagulation paths remain single-box even
    though the containers can store multi-box state.
  - Discoverability: the migration guide links directly to the canonical
    shape-conventions anchor, and the roadmap index points to that subsection
    while keeping the roadmap subsection as the single source of truth.
  - Validation: docs-first update; validation relied on existing
    container, builder, Warp, conversion, condensation, and coagulation
    evidence plus anchor/link smoke checks rather than new automated tests.

- [x] **E2-F1-P4:** Publish downstream handoff map and development documentation
  - Issue: #1186 | Size: XS | Status: Completed
  - Goal: Document explicit ownership decisions needed by E2-F2 through E2-F9 and
    make the decision record discoverable for implementers.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/Roadmap/index.md`,
    `docs/Features/particle-data-migration.md`, and `docs/index.md`.
  - Shipped: the roadmap now publishes a final downstream handoff map with one
    concrete contract note each for `E2-F2` through `E2-F9`, plus a publication
    note stating this phase shipped the finalized P2/P3 ownership/shape
    contract rather than new schema semantics.
  - Discoverability: roadmap, migration, and top-level docs entry points now
    link directly to the canonical handoff anchor so downstream implementers can
    find the authoritative contract without searching plan prose.
  - Tests: documentation validation recorded in the roadmap publication note as
    `python3 .opencode/tools/build_mkdocs.py --validate-only --strict`.

## Phase Ordering Notes

- P1 must finish before P2 so ownership decisions reference an explicit current-state
  inventory rather than assumptions.
- P2 is the gate for P3 because shape conventions should follow the approved field
  ownership and lossy-transfer decisions.
- P4 is last and should publish only the decisions finalized in P2 and P3 so sibling
  tracks inherit one authoritative handoff.
