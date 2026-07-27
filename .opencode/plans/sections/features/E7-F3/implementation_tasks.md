# Implementation Tasks

## Backend

- [ ] Create `particula/execution/adapters/coagulation.py`; defer Warp imports
  until Warp adapter resolution so neutral execution imports remain CPU-safe.
- [ ] Add Brownian coagulation capability declarations to the E7-F1 matrix,
  gated by E7-F6 backend/device availability and error policy.
- [ ] Define `BrownianCoagulationConfig` plus typed CPU and resident Warp state
  views in `particula/execution/adapters/coagulation.py` without changing
  `Aerosol`, `ParticleData`, or `WarpParticleData` schemas.
- [ ] Define collision-output ownership and a persistent RNG resource contract
  with seed-once, reuse, and explicit-reset operations.
- [ ] Implement the CPU adapter by delegating exact `time_step`/`sub_steps` to
  `particula.dynamics.Coagulation.execute()`.
- [ ] Implement the Warp adapter by delegating Brownian particle-resolved work
  to `particula.gpu.kernels.coagulation_step_gpu()` exactly once per call.
- [ ] Preserve particles, supplied output buffers, and RNG identities in the
  E7-F1 result/mutation metadata; avoid host readback of collision counts.
- [ ] Reject charged, sedimentation, turbulent, combined, unsupported
  distribution, unavailable-device, and malformed state requests before
  concrete invocation where selection owns the check.
- [ ] Propagate concrete runtime failures without CPU retry, transfer,
  synchronization, RNG reset, or rollback claims.
- [ ] Add only E7-F6-approved public exports and retain concrete mechanism and
  scratch types at current module locations.

## Tooling / Tests

- [ ] Add `particula/execution/tests/coagulation_adapter_test.py` coverage for
  capabilities, typed state, dispatch, validation ordering, identities, and
  failure propagation; reserve cross-backend fixtures for
  `particula/execution/tests/coagulation_integration_test.py`.
- [ ] Add repeated-call tests proving RNG progression without reseeding and
  explicit reset reproducibility from the same seed.
- [ ] Add one-box and multi-box Warp CPU invariant tests for mass/charge
  conservation, inactive slots, output bounds, and caller-owned buffers.
- [ ] Add bounded CPU-reference comparisons for deterministic Brownian rates
  and aggregate stochastic outcomes; do not require exact trajectories.
- [ ] Add transfer/synchronization spies proving selected steps do not invoke
  conversion helpers, restore helpers, or implicit synchronization.
- [ ] Add optional CUDA rows that skip cleanly and preserve the same contract.
- [ ] Maintain at least 80% changed-module coverage and do not lower any
  repository threshold.
- [ ] Run focused pytest, Ruff, mypy, export regressions, and strict docs build.
