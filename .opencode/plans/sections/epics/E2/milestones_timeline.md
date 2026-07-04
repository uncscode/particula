## Milestones and Timeline

Epic plans in the current plan schema do not accept executable phases, so the
workstreams below act as milestone groupings for child feature sequencing.

| Milestone | Child Tracks | Target Exit Criteria |
|-----------|--------------|----------------------|
| M1: Schema foundation | E2-F1 | `docs/Features/Roadmap/data-oriented-gpu.md` and the linked decision record capture field ownership, `(n_boxes, ...)` shape rules, and CPU/GPU round-trip loss semantics for downstream implementers. |
| M2: Environment containers | E2-F2, E2-F3 | `particula/gas/environment_data.py`, `particula/gpu/warp_types.py`, and `particula/gpu/conversion.py` land with co-located tests proving one-box and multi-box environment validation plus CPU↔Warp round trips. |
| M3: Gas/environment boundary | E2-F4 | `particula/gpu/tests/conversion_test.py` and migration docs lock down `GasData` name, partitioning, and `vapor_pressure` ownership/round-trip behavior with no silent contract gaps. |
| M4: Kernel migration path | E2-F5 | `condensation_step_gpu` and `coagulation_step_gpu` accept normalized per-box environment inputs, preserve legacy scalar call shapes, and reject `n_boxes` or device mismatches before launch. |
| M5: Numerical evidence | E2-F6, E2-F7 | Reproducible precision and stiffness reports publish concrete tables, commands, and recommendations for the `fp64` baseline mass dtype policy plus whether gas-coupled production condensation integration remains inside E2-F7 or moves to a named follow-up feature. |
| M6: Support boundaries and handoff | E2-F8, E2-F9 | CPU support-boundary tests, user-facing docs, and guarded examples are published so current single-box limits and GPU transfer expectations are discoverable without reading source. |

### Sequencing Notes

- E2-F6 and E2-F7 should start once E2-F1 stabilizes enough to define reference
  inputs and outputs; they do not need to wait for all container implementation
  tracks to ship.
- E2-F9 should avoid speculative docs and instead publish behavior proven by
  earlier tracks.
- Each child track must include co-located tests with implementation changes.
