# Outcomes and Guardrails

## Target Outcomes

1. **Persistent coagulation RNG state.** Caller-owned `rng_states` can be
   initialized once, passed through repeated `coagulation_step_gpu` calls, and
   observed to advance rather than reset.
2. **Characterized mixed-scale rejection sampling.** Mixed NPF/droplet cases
   have deterministic or statistical tests that expose acceptance behavior and
   document whether Epic C hardens the sampler or records a scoped limitation.
3. **Documented coagulation threading decision.** The one-thread-per-box design
   has a decision note, benchmark evidence, and clear follow-up criteria for a
   parallel-within-box variant.
4. **Discoverable low-level kernel API.** Users can find supported import paths
   for condensation and coagulation kernels and run a direct GPU quick-start
   without hidden transfer behavior.
5. **Formal device-aware test policy.** Warp CPU runs are required where Warp is
   available; CUDA tests run when CUDA is available and skip cleanly otherwise;
   stochastic tests use distribution/tolerance checks instead of exact per-seed
   equality unless exact parity is part of the contract.
6. **Latent-heat CPU reference material.** A runnable docs example and a CPU
   integration-level conservation baseline exist for future GPU parity tests.

## Guardrails

- No high-level backend selection or automatic CPU/GPU dispatch is introduced.
- No new GPU physics is introduced; coagulation and condensation behavior stays
  within existing model scope.
- No hidden CPU/GPU transfers or hidden synchronization are added.
- CUDA remains optional. Tests must pass or skip cleanly on systems without
  CUDA hardware.
- Kernel APIs should remain explicit about device arrays, scalar inputs, and
  caller-owned data.
- Unit tests ship with each feature track alongside the code being changed; no
  standalone testing-only implementation phase is planned.
- Test coverage thresholds must not be lowered.

## Exit Bar

- All seven child feature tracks have shipped or have explicit deferral notes.
- `pytest particula/gpu/kernels/tests particula/gpu/tests -q` passes in a Warp
  CPU environment, with CUDA coverage enabled when available.
- Relevant docs/examples run through existing docs-example smoke-test patterns.
- The roadmap entry for Epic C reflects final decisions, limitations, and
  follow-up links.
