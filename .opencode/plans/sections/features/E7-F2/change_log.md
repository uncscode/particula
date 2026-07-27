# Change Log

## 2026-07-27 — E7-F2-P3 shipped (Issue #1472)

- Added concrete-only selected isothermal CPU and Warp P3 carriers/adapters in
  `particula/execution/adapters/condensation.py`.
- Added exact preflight, one unchanged native call, and identity-preserving
  `ExecutionResult` normalization; Warp kernel resolution remains lazy.
- Kept the boundary free of transfer, restoration, synchronization, fallback,
  and failure recovery, and retained the narrow public export surface.
- Added CPU/Warp dispatch and export-boundary coverage.

## 2026-07-27 — E7-F2-P2 shipped (Issue #1471)

- Migrated `particula/execution.py` to the `particula.execution` package while
  preserving legacy selection imports and the exact ten-name public `__all__`.
- Added concrete-only `CondensationExecutionConfig`, `CPUCondensationState`,
  and lazy `WarpCondensationState` carriers in
  `particula/execution/adapters/condensation.py`.
- Added identity retention, ordered metadata-only validation, and contiguous
  writable-output alias/overlap protection without execution, transfer,
  allocation, or synchronization.
- Added carrier/import-export tests covering CPU and Warp construction,
  validation ordering, ownership checks, and accepted/rejected non-mutation.

## 2026-07-27 — E7-F2-P1 shipped (Issue #1470)

- Added direct-module-only condensation configuration vocabularies and frozen
  validation in `particula/execution.py`.
- Added exact four-axis requirements mapping and immutable catalogue entries for
  36 CPU configurations and 8 declarative Warp profiles.
- Added focused `particula/tests/execution_test.py` coverage for catalogue and
  mapping semantics, deterministic rejection order, purity/immutability, and
  optional-import isolation.
- Kept runtime availability, native-device handling, adapter selection, public
  exports, and GPU APIs unchanged; those concerns remain outside P1.

## 2026-07-26 — Initial Draft

- Created E7-F2 from issue #1451 Track T2 under parent epic E7.
- Recorded hard dependencies on E7-F1 and E7-F6 and the downstream handoff to
  E7-F4, E7-F5, and E7-F9.
- Added six issue-sized phases covering capability mapping, typed state and
  sidecars, isothermal adaptation, latent heat and unsupported modes, bounded
  parity/conservation evidence, and development documentation.
- Preserved issue scope: CPU remains the reference; Warp CPU is the baseline;
  CUDA is optional; staggered GPU/BAT expansion, hidden transfers/fallback,
  physics rewrites, resident scheduling, performance, and autodiff are excluded.
- Codebase-researcher dispatch was blocked by the subagent-depth limit. The
  draft instead used issue #1451, E7 epic sections, E7-F1 handoff sections, the
  Epic G roadmap, and direct condensation/kernel/test source references.
