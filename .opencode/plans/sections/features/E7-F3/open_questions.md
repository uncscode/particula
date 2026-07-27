# Open Questions

- [x] Does T3 include every mechanism currently executable by the direct Warp
  coagulation kernel?
  - Resolved 2026-07-26: No. Issue #1451 names backend-selected **Brownian**
    coagulation; charged, sedimentation, turbulent, and combined selection are
    outside E7-F3 unless a later scoped plan adds them.
  - Evidence: issue #1451 Track T3 and Epic G suggested phases.

- [x] Should a repeated `rng_seed` automatically reseed a supplied RNG buffer?
  - Resolved 2026-07-26: No. Seed once, reuse existing state, and reset only by
    explicit intent.
  - Evidence: `particula/gpu/kernels/coagulation.py:2215-2222,2341-2356`.

- [x] Must CPU and Warp produce identical collision trajectories?
  - Resolved 2026-07-26: No. Preserve backend-specific stochastic algorithms;
    verify deterministic rate semantics, statistical behavior, conservation,
    and within-backend reset replay.
  - Evidence: issue #1451 excludes exact CPU/CUDA stochastic trajectory equality.

- [x] Should T3 require caller-provided Warp RNG state for every selected call,
  or permit the direct kernel's allocate-and-seed convenience path for a
  documented one-shot request?
  - Resolved 2026-07-27: Require caller/session-owned persistent RNG state for
    every backend-selected Warp adapter call; retain omitted-state convenience
    only on the unchanged low-level direct kernel.
  - Rationale: The direct call does not return its call-local RNG buffer, so the
    selected resident API could not checkpoint or continue that stream.
  - Evidence:
    - `particula/gpu/kernels/coagulation.py:2189` - omitted state is call-local,
      while supplied state is persistent and explicitly reset.
    - `.opencode/plans/sections/features/E7-F3/implementation_tasks.md:12` - the
      accepted adapter plan requires a persistent RNG resource contract.
  - Resolved by: plan-question-resolver

- [x] Which exact module owns the process-specific adapter and stable wrapper
  types after E7-F1/E7-F6 implementation?
  - Resolved 2026-07-27: Put Brownian process configuration and CPU/Warp state
    views in `particula/execution/adapters/coagulation.py`; keep generic request,
    result, capability, context, protocol, error, and fallback types in the
    `particula.execution` package.
  - Rationale: This isolates optional Warp imports in the concrete adapter while
    preventing process-specific types from widening the neutral public layer.
  - Evidence:
    - `.opencode/plans/sections/features/E7-F3/implementation_tasks.md:5` - the
      canonical task names the concrete adapter module and lazy import boundary.
    - `.opencode/plans/sections/features/E7-F1/architecture_design.md:39` - the
      neutral layer owns generic request, result, state, and adapter protocols.
  - Resolved by: plan-question-resolver
