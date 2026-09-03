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
- Update E8-F4 plan sections with the P1 delivery: direct-import-only prepared
  qualification, lazy adapter probes, exact identity retention, READY-preserving
  no-handle/no-cleanup scope, actual test files, and P2/P3 handoff.
- Add or update hardware-free documentation contract assertions for all new
  claims and run `mkdocs build --strict`.
- No user-facing documentation changed for P1. Defer the runnable graph-capture
  example and tutorial to E8-F8; native capture/replay documentation awaits P2/P3.
