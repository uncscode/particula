# Documentation Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` Track T7 status and retain
  the boundary between prescribed maps and full CFD coupling.
- Update `docs/Features/data-containers-and-gpu-foundations.md` with volume
  authority, communication ownership, supported map/representation table,
  fixed-capacity behavior, and explicit transfer/synchronization rules.
- Add or update an E7 execution guide under `.opencode/guides/` describing map
  construction, scheduler ordering, derived-state invalidation, failure
  boundaries, and checkpoint/restart responsibilities.
- Document equations for amount-ledger transport and expansion:
  `amount = concentration * volume` and
  `new_concentration = final_amount / new_volume`.
- Document particle slot semantics, matching/free-slot rules, capacity errors,
  conservation diagnostics, and unsupported representation combinations.
- Document Warp CPU validation as routine evidence and CUDA as optional evidence;
  avoid performance or exact cross-device replay claims.
- Add a focused prescribed 1D communication/expansion example only if it can be
  kept runnable and regression-tested without absorbing E7-F9's complete-loop
  example ownership.
- Update `.opencode/plans/sections/features/E7-F7/` with shipped phase status,
  resolved decisions, exact files, tolerances, and reproduction commands as
  implementation lands.
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
- Run `mkdocs build --strict` and relevant docs tests before completing P7.
