# Documentation Updates

- Update `docs/Features/data-containers-and-gpu-foundations.md` with the
  resident-session authority boundary, one-time setup, reusable sidecars,
  CPU-only metadata, and explicit checkpoint/finalize behavior.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` Track T4 status and record
  the shipped contract without claiming E7-F5 scheduling, E7-F7 transport, or
  E7-F8 RNG semantics.
- Add or update a development guide under `.opencode/guides/` describing
  lifecycle states, ownership, fixed-shape resource manifests, failure
  semantics, and extension rules for process adapters.
- Update `AGENTS.md` quick-reference guidance with public imports, transfer
  boundaries, Warp CPU validation commands, and the prohibition on implicit
  fallback/restore.
- Add a focused documented example showing setup, multiple resident lifecycle
  steps, nonterminal checkpoint, restart, and finalization. Keep process
  scheduling illustrative until E7-F5 and preserve lazy no-Warp imports.
- Document checkpoint schema version, ordered gas-name handling, intentionally
  lossy CPU gas restore fields, mutable resource/RNG payload ownership, and
  compatibility rejection behavior.
- Update these structured plan sections with final implementation paths,
  phase issues, shipped status, and validation evidence.
- Validate all documentation with `mkdocs build --strict`, link checks, and
  executable documentation regression tests on the no-Warp and Warp CPU paths.

## P1 Status

Issue #1484 updated the structured plan only. No user-facing documentation,
examples, exports, or lifecycle API were added because P1 is a concrete-only
 construction boundary; broader documentation remains P7 work.

## P2 Status

Issue #1485 updated only code-level module/factory docstrings and these
structured plan sections. No user-facing documentation, example, export, or
lifecycle API was added: `setup_resident_session()` remains concrete-only and
direct-import-only. P7 remains responsible for broader documentation after the
remaining lifecycle surface exists.

## P4 Status

Issue #1487 updated the concrete `gpu_session`/resource-registry contract and
the architecture references only. The updates document direct-import-only
`ResidentStepGuard`/`ResidentStepToken`, identity-token completion, the
metadata-only `validate_pinned_session()` seam, and the required
`assert_step_closed()` hook for future P5/P6 lifecycle boundaries. No public
API, user example, checkpoint/restore implementation, or broad documentation
surface was added.

## P5 Status

Issue #1488 added `docs/Features/gpu_resident_checkpoints.md`. It documents the
concrete-only direct import, explicit registry/guard ownership, nonterminal
checkpoint and idempotent terminal finalization, immutable canonical payloads,
lossy inspection vapor pressure, same-device fresh restart, host-copy cost, and
the exclusions for package exports, serialization, migration, fallback, and
rollback. The feature page is validated with `mkdocs build --strict`.
