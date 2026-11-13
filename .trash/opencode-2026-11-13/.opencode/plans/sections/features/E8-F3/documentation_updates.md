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

Issue #1564 likewise changes no user-facing documentation. P4's
`CaptureResourceRequirements`, `CaptureResourceSet`, preparation transaction,
and metadata-only validator remain direct-module-only in
`particula.execution.gpu_resources`. The optional resident-enqueue reference
does not create a user workflow or change READY/capture behavior. Module
docstrings and execution tests record the atomicity, identity-reuse, and retry
boundary.

Issue #1565 updated `docs/Features/Roadmap/data-oriented-gpu.md` and
`AGENTS.md`. The durable contract now requires callers to publish the complete
set before final resident-request construction; E8-F1 CAPTURED admission and
E8-F2 READY preparation then perform cached metadata-only validation and freeze
the exact requirements/set/report identities in `configurations`. It also
records the logical-byte exclusions (allocator-reserved bytes, pointers/payload
reads, checkpoint copies, and future autodiff tapes). The runnable resident
example was updated to follow the required setup order. No public exports or
native capture/replay user workflow were added.

- [x] Updated `docs/Features/Roadmap/data-oriented-gpu.md` and `AGENTS.md` with
  the implemented publication, cached-validation, identity-pinning, and
  logical-byte-accounting contract.
- Update `.opencode/guides/testing_guide.md` only if new canonical resident
  test or coverage commands are introduced; do not restate transient phase
  commands as permanent policy.
- Cross-reference E8-F1 lifecycle and E8-F2 prepared-enqueue documentation, and
  hand deterministic byte records to E8-F6 without claiming allocator-reserved,
  checkpoint-copy, or future autodiff-tape totals.
- Do not add a user-facing example in this track; E8-F8 owns the complete graph
  capture example and limitation guide.
- [x] Updated the real resident example and its documentation coverage for the
  pre-publication setup order; no public user workflow was added.
