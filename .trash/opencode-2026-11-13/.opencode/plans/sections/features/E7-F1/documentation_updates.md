# Documentation Updates

- Update `docs/Features/data-containers-and-gpu-foundations.md` with the new
  execution-context import, typed backend/device selection, CPU reference
  behavior, ownership/mutation contract, and explicit no-fallback rule.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` to record E7-F1 phase
  progress and the chosen separate-context API location without claiming that
  GPU adapters or resident loops have shipped.
- Add or update an architecture guide under `.opencode/guides/architecture/`
  describing the boundary between strategies, runnables, execution contexts,
  process adapters, and future sessions.
- Add a small CPU-only usage snippet showing explicit request construction,
  context validation, execution, and result/state identity. Keep direct GPU
  examples unchanged until E7-F2 through E7-F5 supply executable paths.
- Document the capability-extension contract for E7-F2, E7-F3, and E7-F4 and
  the policy handoff to E7-F6.
- Update these E7-F1 plan sections and phase statuses as implementation ships.
- Do not update `README.md` or `AGENTS.md` unless implementation introduces a
  generally advertised quick-start API or contributor workflow requirement.

Validation: run relevant documentation regressions, verify all relative links,
and run `mkdocs build --strict` with warnings treated as failures.

## P2 Record (issue #1463)

No user-facing documentation was updated: `ExecutionContext` registration is
private and P2 has no executable adapter or stable public export. The structured
plan now records the selection-only boundary; publish user imports and examples
only after the later public-export and execution-contract phases.

## P3 Record (issue #1464)

No user-facing documentation was updated. P3 contracts are internal to
`particula.execution`, have no public export, and deliberately provide no
 executable adapter path. Publication and examples remain deferred to the later
 export/documentation phases.

## P4 Record (issue #1465)

No user-facing documentation was updated. `CPUExecutionState` and
`CPUExecutionAdapter` are unexported direct-module execution seams; they add no
stable public import or GPU execution path. The structured plan records the
completed CPU contract, while publication and examples remain deferred to P5/P6.

## P5 Record (issue #1466)

No user-facing documentation files were updated. P5 deliberately publishes only
the ten dependency-neutral selection/context symbols and the typed,
context-local registration seam; its public-surface tests document the exact
boundary. P3/P4 and GPU contracts remain excluded. User-facing usage guidance
and broader contract documentation remain P6 work.
