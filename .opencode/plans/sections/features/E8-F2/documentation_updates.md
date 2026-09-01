# Documentation Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` to mark T2/E8-F2 phases
  and clearly distinguish validated setup from capture-ready enqueue.
- Update `docs/Features/data-containers-and-gpu-foundations.md` with the
  concrete-only prepared resident boundary, ownership rules, and explicit list
  of forbidden enqueue-time host operations.
- Update `AGENTS.md` with the implemented setup/enqueue contract, focused test
  commands, Warp CPU baseline, and optional CUDA capture gate.
- Update docstrings in every changed execution and kernel module to identify
  which functions validate, which only enqueue, and the post-launch recovery
  limit.
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
- Add or update documentation contract assertions under `particula/tests/` or
  `particula/execution/tests/` when user-facing contract text changes.
- Update all E8-F2 plan sections and parent E8 child/dependency status during
  phase shipping. E8-F8, not this feature, owns the complete runnable capture
  example and end-user limitations guide.
- Validate links and rendering with `mkdocs build --strict`.
