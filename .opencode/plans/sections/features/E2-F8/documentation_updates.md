# E2-F8 Documentation Updates

## Primary Documentation Targets

- `docs/Features/particle-data-migration.md`
  - Expand "Using ParticleData/GasData in dynamics".
  - Expand "Single-box vs multi-box data".
  - Add a support table similar to:

    | CPU dynamics path | Containers accepted | Multi-box strategy execution |
    | --- | --- | --- |
    | Condensation | `ParticleData` + `GasData` | `n_boxes=1` only; multi-box raises `ValueError` |
    | Coagulation | `ParticleData` | `n_boxes=1` only; multi-box raises `ValueError` |

- `docs/Features/Roadmap/data-oriented-gpu.md`
  - Clarify that data-container shape support is a prerequisite for future
    multi-box execution, not proof that all current CPU strategies execute all
    boxes.

## Error Message Documentation

Document the intended interpretation of errors:

- `ParticleData`/`GasData` may be multi-box containers.
- Current CPU strategies require single-box execution unless otherwise stated.
- Users may run per-box loops manually until first-class multi-box strategy
  execution is implemented.

## Examples to Add or Update

- Single-box data-container dynamics example using `n_boxes=1`.
- A short warning block for `n_boxes > 1` CPU strategy calls.
- Optional caller-managed loop pseudocode that extracts one box at a time, if
  consistent with final implementation behavior.

## Documentation Done Criteria

Docs must let a reader distinguish these statements:

1. The containers can store multiple boxes.
2. Legacy-compatible CPU strategy code is currently single-box only unless a
   strategy explicitly documents broader support.
3. Full strategy-level multi-box execution is future work unless a strategy
   explicitly documents otherwise.
