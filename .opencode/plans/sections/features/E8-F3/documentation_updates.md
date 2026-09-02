# Documentation Updates

## P1 Status

No user-facing documentation changed for issue #1561. The immutable inventory
and `logical_resource_report()` remain direct-module-only in
`particula.execution.gpu_resources`; package and top-level exports are unchanged.
The focused contract is covered in `particula/execution/tests/gpu_resources_test.py`.

Issue #1562 likewise adds no user-facing documentation: the dilution descriptor
family and `PreparedResourceViews` remain concrete-only, with no public export.

Issue #1563 likewise changes no user-facing documentation. Capture registration
and selected-resource reporting remain concrete-only in
`particula.execution.gpu_resources`; no public export, scheduler workflow, or
checkpoint contract changed. Module docstrings and execution tests record the
identity, metadata-only, and reporting boundary.

- Update `docs/Features/Roadmap/data-oriented-gpu.md` Epic H sections with the
  implemented registry-owned capture resource inventory, setup/reuse boundary,
  persistent RNG handling, and deterministic logical-byte accounting semantics.
- Update `docs/Features/data-containers-and-gpu-foundations.md` if the concrete
  resident ownership table needs process/control/diagnostic resource details;
  keep all new APIs direct-import-only.
- Update `AGENTS.md` only after implementation to summarize validated
  preallocation, identity pinning, and focused/full validation commands.
- Update `.opencode/guides/testing_guide.md` only if new canonical resident
  test or coverage commands are introduced; do not restate transient phase
  commands as permanent policy.
- Cross-reference E8-F1 lifecycle and E8-F2 prepared-enqueue documentation, and
  hand deterministic byte records to E8-F6 without claiming allocator-reserved,
  checkpoint-copy, or future autodiff-tape totals.
- Do not add a user-facing example in this track; E8-F8 owns the complete graph
  capture example and limitation guide.
- Promote phase statuses and update this plan's change log after each shipped
  increment. Validate all links with `mkdocs build --strict` when documentation
  changes.
