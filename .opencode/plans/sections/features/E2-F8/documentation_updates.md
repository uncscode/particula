# E2-F8 Documentation Updates

## Primary Documentation Targets

- P1 result: no user-facing doc files changed.
- P2 result: no user-facing doc files changed.
- P3 completed the user-facing contract update in:
  - `docs/Features/particle-data-migration.md`
    - Reworked `Using ParticleData/GasData in dynamics` into the canonical CPU
      support contract.
    - Added the compact CPU support table separating container acceptance from
      current execution support.
    - Added a supported single-box `n_boxes == 1` example.
    - Added clearly labeled caller-managed per-box loop pseudocode for
      unsupported multi-box CPU workflows.
    - Added a troubleshooting cross-reference back to the canonical support
      section.
  - `docs/Features/Roadmap/data-oriented-gpu.md`
    - Qualified roadmap wording so container compatibility is not read as
      current CPU multi-box execution support.
    - Pointed readers back to the migration guide for the canonical contract.

## Error Message Documentation

Document the intended interpretation of current behavior:

- `ParticleData`/`GasData` may be multi-box containers.
- Current covered CPU strategies are single-box-only unless otherwise stated.
- Users may run per-box loops manually until first-class multi-box strategy
  execution is implemented.

No runtime error-text update was needed in P3 because the existing tested
`n_boxes == 1` boundary already matched the final documentation wording.

## Examples Added

- Single-box data-container dynamics example using `n_boxes=1`.
- A compact support table for current CPU container workflows.
- Caller-managed loop pseudocode that extracts one box at a time.

## Documentation Done Criteria

The shipped doc updates now let a reader distinguish these statements:

1. The containers can store multiple boxes.
2. Current CPU strategy code is single-box only unless a strategy explicitly
   documents broader support.
3. Full strategy-level multi-box execution is future work unless a strategy
   explicitly documents otherwise.
