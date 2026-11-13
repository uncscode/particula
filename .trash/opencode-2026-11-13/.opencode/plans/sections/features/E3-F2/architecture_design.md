# E3-F2 Architecture Design

## Current Architecture

`coagulation_step_gpu(...)` validates inputs, allocates/reuses buffers, prepares
RNG state, and launches Warp kernels. Candidate collision selection currently
uses one thread per box and a single global majorant `k_max` derived from the
largest size-disparity pair in the box. Candidate pairs are sampled sequentially
inside the box thread and accepted by comparing a random draw with
`brownian_kernel_pair_wp(...) / k_max`.

## Proposed Design Direction

1. Add test/debug diagnostics that make acceptance behavior measurable without
   changing the production CPU/GPU transfer boundary.
2. Harden the proposal step inside the existing global-majorant sampler by
   selecting uniformly over currently active particles instead of retrying raw
   particle indices until two active slots are found.
3. Keep all physics unchanged: Brownian pair rates remain the source of truth,
   and changes affect only proposal efficiency/characterization.

## Landed Phase-E3-F2-P1 Design

- The shipped implementation kept all instrumentation inside
  `particula/gpu/kernels/tests/coagulation_test.py`.
- P1 added a private mixed-scale `ParticleData` fixture and a private mirrored
  Warp kernel that records rounded attempted counts plus accepted counts for the
  existing sampler logic.
- Production `particula/gpu/kernels/coagulation.py` was left unchanged; the
  diagnostic helper compares its accepted counts against
  `coagulation_step_gpu(...)` from the same seeded setup.
- This preserves the E3-F1 caller-owned RNG-state contract and keeps
  `.numpy()`/`wp.synchronize()` at explicit test boundaries only.

## Landed Phase-E3-F2-P2 Design

- `particula/gpu/kernels/coagulation.py` now replaces the retry-based
  raw-index proposal loop with bounded active-particle rank selection.
- Each scheduled trial draws two distinct active ranks, resolves both indices in
  a single pass over `active_flags`, and exits the scan early once both active
  particle indices are found.
- Accepted-pair behavior stays unchanged after selection: production code still
  uses `brownian_kernel_pair_wp(...) / k_max`, writes sorted accepted pairs into
  `collision_pairs`, clears both active flags, decrements `active_count` by 2,
  and preserves the existing collision-capacity exit conditions.
- The mirrored diagnostic kernel in
  `particula/gpu/kernels/tests/coagulation_test.py` implements the same bounded
  selector so seeded parity checks continue comparing accepted counts, pair
  prefixes, post-apply masses, concentrations, and RNG-state behavior.
- No fixed-bin or alternate-majorant structure shipped in P2; the hardening was
  intentionally limited to bounded candidate selection within the existing
  global-majorant architecture.

## Boundary and API Constraints

- No implicit `.numpy()` calls, host readbacks, or hidden synchronization in
  production code.
- Respect E3-F1 RNG ownership and seed-once behavior for `rng_states`.
- Preserve existing `coagulation_step_gpu(...)` return shape and buffer reuse
  contracts unless a clearly test-only diagnostic helper is added.
- Keep CUDA optional and Warp CPU-supported.

## Decision Gate

The implementation may ship either a conservative improvement or a documented
accepted limitation. In both cases, the final state must include a reproducible
mixed-scale metric, stochastic correctness evidence, and mass conservation
evidence.
