# E2-F8 Documentation Updates

## Primary Documentation Targets

- P1 result: no user-facing doc files changed.
- Reviewed future clarification targets only:
  - `docs/Features/particle-data-migration.md`
    - Revisit "Using ParticleData/GasData in dynamics".
    - Revisit "Single-box vs multi-box data".
    - Preserve the current baseline distinction:

      | CPU dynamics path | Containers accepted | Current multi-box strategy behavior |
      | --- | --- | --- |
      | Condensation | `ParticleData` + `GasData` | `n_boxes=1` only; multi-box raises the existing `ValueError` |
      | Coagulation | `ParticleData` | Multi-box inputs are accepted today, but helper reads and particle-resolved mutation stay box-0-only |

  - `docs/Features/Roadmap/data-oriented-gpu.md`
    - Clarify that data-container shape support is a prerequisite for future
      multi-box execution, not proof that all current CPU strategies execute all
      boxes.

## Error Message Documentation

Document the intended interpretation of current behavior:

- `ParticleData`/`GasData` may be multi-box containers.
- Current CPU strategies may be single-box-only or transitional box-0-only
  unless otherwise stated.
- Users may run per-box loops manually until first-class multi-box strategy
  execution is implemented.

## Examples to Add or Update Later

- Single-box data-container dynamics example using `n_boxes=1`.
- A short warning block for `n_boxes > 1` CPU strategy calls.
- Optional caller-managed loop pseudocode that extracts one box at a time, if
  consistent with final implementation behavior.

## Documentation Done Criteria

When later doc updates land, they must let a reader distinguish these
statements:

1. The containers can store multiple boxes.
2. Legacy-compatible CPU strategy code is currently single-box only unless a
   strategy explicitly documents broader support.
3. Full strategy-level multi-box execution is future work unless a strategy
   explicitly documents otherwise.
