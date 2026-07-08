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
2. Evaluate a small, fixed-size stratification design suitable for Warp:
   - derive radius bins from device-local data or existing min/max summaries;
   - compute conservative per-bin-pair majorants with
     `brownian_kernel_pair_wp(...)`;
   - sample candidate pairs from selected bin pairs while preserving the same
     collision-pair output contract;
   - fall back to the existing global-majorant path when bins are sparse or the
     hardening path cannot be made conservative.
3. Keep all physics unchanged: Brownian pair rates remain the source of truth,
   and changes affect only proposal efficiency/characterization.

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
