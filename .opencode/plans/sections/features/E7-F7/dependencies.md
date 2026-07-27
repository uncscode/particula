# Dependencies

## Upstream

- **E7-F4 / T4 — GPU-resident session lifecycle:** mandatory owner of resident
  particle/gas/environment state, dimensions, reusable sidecars, checkpoints,
  and fault lifecycle.
- **E7-F5 / T5 — deterministic process scheduler:** mandatory owner of typed
  nodes, canonical barriers, update invalidation, and mutation-window ordering.
- **E7-F6 / T6 — fallback and API-stability policy:** mandatory capability,
  availability, export, error, and no-runtime-fallback boundary.
- E7-F4/F5/F6 inherit E7-F1's execution context. E7-F7 must not duplicate
  backend selection or establish a parallel public execution API.
- Existing multi-box containers, conversion helpers, direct dilution, fixed-slot
  activation/exhaustion primitives, and kernel validation patterns are shipped
  technical prerequisites, not new scope.

## Sibling and Downstream

- **E7-F8 / T8** consumes the resident multi-box/session boundary for persistent
  per-box RNG but must remain independent of deterministic communication maps.
- **E7-F9 / T9** depends on E7-F7 for complete multi-timestep regressions,
  checkpoint-only transfer evidence, user examples, and publication of the
  supported full-loop matrix.
- Epic H may later graph-capture stable communication layouts; E7-F7 makes no
  capture or performance claim.
- Epic I may later differentiate supported loop components; E7-F7 makes no
  autodiff claim.

## Phase Ordering

P1 fixes map and validation semantics before kernels. P2 establishes volume and
extensive-inventory normalization. P3 adds gas transfer on that ledger. P4 adds
capacity-sensitive particle transport. P5 wires only proven operations into the
session/scheduler. P6 records combined scientific evidence. P7 is the required
final documentation phase. Unit tests ship with P1-P5; P6 is integration and
validation, not a substitute for co-located tests.
