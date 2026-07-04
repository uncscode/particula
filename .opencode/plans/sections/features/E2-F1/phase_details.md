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

- [ ] **E2-F1-P2:** Record authoritative field ownership decisions
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Decide which container owns each particle, gas, and environment field,
    including shared/per-box fields and lossy GPU transfer semantics.
  - Files: schema decision record, `docs/Features/Roadmap/data-oriented-gpu.md`,
    and links from relevant feature docs.
  - Tests: Co-located docs checks or table/example validation where practical.

- [ ] **E2-F1-P3:** Document shape conventions across container workflows
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Publish shape rules for single-box, multi-box, particle-resolved, and
    binned workflows across CPU and GPU containers.
  - Files: shape conventions section in the decision record or feature docs;
    links from particle/gas migration docs.
  - Tests: Co-located docs checks plus review against existing container tests.

- [ ] **E2-F1-P4:** Publish downstream handoff map and development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document explicit ownership decisions needed by E2-F2 through E2-F9 and
    make the decision record discoverable for implementers.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`, feature index or related
    docs, and any architecture cross-reference.
  - Tests: Documentation/link validation.

## Phase Ordering Notes

- P1 must finish before P2 so ownership decisions reference an explicit current-state
  inventory rather than assumptions.
- P2 is the gate for P3 because shape conventions should follow the approved field
  ownership and lossy-transfer decisions.
- P4 is last and should publish only the decisions finalized in P2 and P3 so sibling
  tracks inherit one authoritative handoff.
