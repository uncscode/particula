# Documentation Updates

- Completed for P8 (#1559): updated
  `docs/Features/Roadmap/data-oriented-gpu.md`,
  `docs/Features/data-containers-and-gpu-foundations.md`, and `AGENTS.md` to
  distinguish validated setup from concrete-only prepared enqueue, record
  ownership and forbidden enqueue-time work, and retain the Warp CPU/CUDA
  evidence limits. Focused assertions passed (22 passed in 0.09s), and
  `mkdocs build --strict` passed (exit 0 in 14.67s).
- Completed for P8 (#1559): updated E8-F2 concrete-module docstrings to identify
  setup ownership, retained-reference enqueue, and the post-launch recovery
  limit; these names remain concrete-only and are not public imports.
- Completed for P1 (#1552): `.opencode/guides/architecture_reference.md` and
  `.opencode/guides/architecture/architecture_outline.md` describe
  `resident_enqueue` as the direct-import-only READY preparation boundary, its
  frozen identity metadata and shared read-only validation, and its exclusions.
  They retain `graph_capture` lifecycle ownership and scheduler CAPTURED
  admission/dispatch ownership.
- Completed for P2 (#1553): no user-facing documentation or export change was
  required. The concrete-only setup/enqueue contract is recorded in this plan;
  changed execution modules retain local implementation documentation while
  standalone APIs preserve their existing contract.
- Completed for P3 (#1554): no user-facing documentation or export change was
  required. Concrete-only execution and kernel docstrings describe closed-map
  prepared binding, fixed native enqueue, equal-volume write-free behavior, and
  post-launch recovery limits; this plan records the implementation boundary.
- Completed for P4 (#1555): no user-facing documentation, export, scheduler,
  checkpoint, or resource-schema update was required. Private Google-style
  docstrings in `particula/gpu/kernels/condensation.py` and
  `particula/execution/adapters/condensation.py` describe setup ownership,
  retained identities, enqueue-only restrictions, and the post-launch failure
  boundary. The focused kernel and adapter docstring validation covers these
   local concrete-only descriptions.
- Completed for P5 (#1556): no user-facing documentation or export update was
  required because the change is a private refactor with unchanged direct and
  resident contracts. Local documentation in the changed kernel and adapter
  modules records frozen-reference setup/enqueue ownership and the writer
  failure boundary; this plan records the implementation disposition.
- Completed for P8 (#1559): extended deterministic documentation contract
  assertions and reconciled E8-F2 and parent E8 sections. E8-F8, not this
  feature, remains the pending owner of the complete runnable capture example
  and end-user limitations guide.
