# Documentation Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` with the shipped E6-F6
  defaults, precedence, diagnostics, conservation target, and plan-ID link.
- Update `docs/Theory/Technical/Dynamics/Nucleation_Equations.md` with the
  bounded computational-particle interpretation of resampling and
  representative-volume scaling, including equations, units, and limits.
- Add or update a `docs/Features/` page describing CPU/direct-Warp APIs, policy
  combinations, fixed-shape sidecars, failure behavior, and downstream use.
- Update `AGENTS.md` with direct imports, focused test commands, Warp CPU/CUDA
  policy, and supported/deferred boundaries.
- Update `.opencode/guides/architecture/` only if the policy transaction or
  sidecar ownership establishes a repository-wide decision.
- Update E6/E6-F5/E6-F7/E6-F8/E6-F9 plan cross-references if final API names or
  diagnostics differ from this first pass.

Documentation must state that resampling defaults on, volume scaling defaults
off, resampling runs first when both are enabled, disabling both fails on actual
exhaustion before mutation, and no demand is silently discarded.
