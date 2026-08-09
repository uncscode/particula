# Documentation Updates

- [x] Updated `docs/Features/Roadmap/data-oriented-gpu.md` Track T7 to shipped
  prescribed-map and optional-volume-evolution status while retaining the full
  CFD boundary.
- [x] Updated `docs/Features/data-containers-and-gpu-foundations.md` with
  volume authority, direct/resident communication ownership, supported map
  forms, fixed-capacity behavior, and explicit transfer/synchronization rules.
- [x] Updated `.opencode/guides/architecture/architecture_guide.md` and
  `.opencode/guides/architecture_reference.md` with map construction,
  scheduler ordering, saturation-only invalidation, failure boundaries, and
  checkpoint/restart responsibilities.
- [x] Documented amount-ledger transport and expansion equations:
  `amount = concentration * volume` and
  `new_concentration = final_amount / new_volume`.
- [x] Documented particle slot semantics, matching/free-slot rules, capacity
  gating, conservation diagnostics, and unsupported representation combinations.
- [x] Recorded Warp CPU as routine evidence and CUDA as optional evidence; no
  performance or exact cross-device replay claim was added.
- [x] No focused runnable communication/expansion example was added; E7-F9
  retains complete-loop example and publication ownership.
- [x] Reconciled `.opencode/plans/sections/features/E7-F7/` with shipped phase
  status, resolved decisions, exact files, tolerances, and passed reproduction
  commands.
- P1 (#1507) updated the structured plan only. No user-facing documentation or
  exports were added because `particula.execution.communication` is a
  concrete-only validation seam; its module and API docstrings record the
   read-only and P3-overdraw handoff boundaries.
- P2 (#1508) likewise added no user-facing documentation or export. Its
  concrete-module docstring records the direct active-device volume-evolution
  contract, complete preflight, write-free equal-volume behavior, and
   post-launch no-rollback boundary.
- P3 (#1509) added no user-facing documentation or export. The concrete module
  and `GasCommunicationBuffers` docstrings document caller-owned `(B, S)`
  amount/work/accounting ledgers, synchronous explicit-Euler closed/open-map
   semantics, one gas commit, and the no-transfer/no-sync/no-fallback boundary.
- P4 (#1510) added no user-facing documentation or export. The concrete module
  and `ParticleCommunicationBuffers` docstrings document the direct-only
  import, exact buffer schemas, immutable pre-step planning, exact-match or
  pre-step-free-slot selection, closed-map conservation, caller synchronization,
   and gated-commit/no-post-launch-rollback boundary.
- P5 (#1511) updated the concrete resident-session contract in
  `.opencode/guides/architecture/architecture_guide.md`,
  `.opencode/guides/architecture/architecture_outline.md`, and
  `docs/Features/gpu_resident_checkpoints.md`: direct-only closed-map resource
  acquisition, communication-then-volume barrier order, saturation-only
  invalidation, and schema-v2/schema-v1 restart boundaries. It adds no public
  export or public complete-loop API.
- P6 (#1512) adds no user-facing documentation or API change. It records the
  delivered test-only independent NumPy `float64` multi-box parity and
  conservation evidence in this structured plan.
- [x] P7 (#1513) published the feature page, T7 roadmap status, and E7
  architecture/reference guidance, including
  `.opencode/guides/architecture/architecture_outline.md` and its
  [ADR-018](../../../../guides/architecture/decisions/ADR-018-resident-communication-integration.md)
  reference. It documents direct open-ledger gas accounting, closed fixed-slot
  particle transport, separate standalone volume evolution, resident
  closed-family ordering, exact-device fresh-identity restart, and deferred
  scope. It adds focused documentation-contract/link/plan-state tests.
  No runnable example was added because E7-F9 owns complete-loop publication.
- [x] Passed `pytest particula/execution/tests/gpu_resident_session_docs_test.py -q -Werror`,
  `pytest particula/tests/execution_selection_docs_test.py -q -Werror`, and
  `mkdocs build --strict` for P7 publication.
