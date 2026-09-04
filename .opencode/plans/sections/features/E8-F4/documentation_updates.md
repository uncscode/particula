# Documentation Updates

- No public documentation was changed for P2. Issue #1568 deliberately keeps
  capture concrete-only and excludes README, package-export, and user-workflow
  publication.
- The internal implementation contract is recorded in E8-F4 plan sections:
  native begin/frozen dispatch/end capture, opaque handle ownership, CUDA-only
  smoke coverage, and denied exports. Replay, user guidance, examples, and
  broader evidence remain downstream work.
- Update `.opencode/guides/testing_guide.md` only if permanent graph-capture test
  locations, marker usage, or validation commands change; preserve focused
  coverage-disabled versus untargeted full-package coverage policy.
- E8-F4 plan sections now record P1 qualification, P2 capture, P3 guarded
  replay, and P4 teardown delivery, including actual implementation and test
  file paths. P4 remains concrete-only; no user-facing documentation changed.
  User guidance, examples, and broader replay evidence remain downstream work.
- Add or update hardware-free documentation contract assertions for all new
  claims and run `mkdocs build --strict`.
- No user-facing documentation changed for P1, P2, or P3. Defer the runnable
  graph-capture example and tutorial to E8-F8.
