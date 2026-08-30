# Documentation Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` to mark the prepared
  fixed-sequence capture, guarded replay, and invalidation implementation while
  leaving E8-F5 through E8-F8 evidence explicitly pending.
- Update `docs/Features/data-containers-and-gpu-foundations.md` with the
  concrete-only graph owner, exact ownership, supported CUDA boundary, no Warp
  CPU capture claim, mutable payload/RNG rule, and teardown/recapture triggers.
- Update `AGENTS.md` with the focused graph lifecycle and full-loop validation
  commands and the no fallback/automatic recapture contract.
- Update `.opencode/guides/testing_guide.md` only if permanent graph-capture test
  locations, marker usage, or validation commands change; preserve focused
  coverage-disabled versus untargeted full-package coverage policy.
- Update E8 and E8-F4 plan sections with final file paths, phase status,
  implementation decisions, and handoffs to E8-F5 through E8-F8.
- Add or update hardware-free documentation contract assertions for all new
  claims and run `mkdocs build --strict`.
- Defer the runnable graph-capture example and user-facing tutorial to E8-F8;
  E8-F4 documentation is an implementation and validation contract only.
