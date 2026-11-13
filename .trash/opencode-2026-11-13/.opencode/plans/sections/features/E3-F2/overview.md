# E3-F2 Overview

## Problem Statement

The current GPU Brownian coagulation rejection sampler uses a single global
`k_max` for each box. Mixed NPF/droplet populations can span many orders of
magnitude in radius and mass, so a global majorant may collapse acceptance rates
or obscure whether the sampler remains statistically useful for those cases.
This feature hardens that behavior where practical, or records a measured,
bounded limitation with reproducible diagnostics.

## Value Proposition

- Provides regression coverage for mixed NPF/droplet particle populations.
- Gives developers an acceptance-rate metric that characterizes rejection
  sampling efficiency without adding hidden CPU/GPU transfers.
- Evaluates bounded sampling improvements, such as fixed size-bin majorants or
  stratified pair selection, against conservation and Brownian-rate tolerances.
- Keeps E3's GPU correctness track focused on low-level behavior rather than
  introducing new coagulation physics.

## Shipped So Far

- Phase E3-F2-P1 shipped in issue #1241 as test-only characterization work in
  `particula/gpu/kernels/tests/coagulation_test.py`.
- The landed baseline adds a deterministic mixed NPF/droplet fixture, a
  mirrored test-local attempt/accept diagnostic path, and focused mixed-scale
  Warp CPU/CUDA regression coverage.
- Phase E3-F2-P2 shipped in issue #1242 with a bounded active-particle
  rank-selection path inside `particula/gpu/kernels/coagulation.py` while
  preserving the existing global-majorant Brownian physics, collision buffer
  contract, and caller-owned RNG-state semantics.
- Phase E3-F2-P3 shipped in issue #1243 as evidence-focused follow-through in
  `particula/gpu/kernels/tests/coagulation_test.py` plus a concise roadmap note
  in `docs/Features/Roadmap/data-oriented-gpu.md`.
- The same bounded selector was mirrored in
  `particula/gpu/kernels/tests/coagulation_test.py`, and the mixed-scale test
  surface now covers selector validity, sparse/degenerate active sets,
  exactly-two-active fallback behavior, accepted-count bounds, and total-mass
  conservation.
- The shipped mixed-scale evidence now also covers repeated seeded Brownian-rate
  checks against expected mean/sigma tolerances, repeated-run conservation even
  when some trials accept zero collisions, and caller-owned `rng_states`
  reuse/reset semantics for the same fixture.
- No public `coagulation_step_gpu(...)` API, exported helper surface, or
  production synchronization behavior changed in this phase.

## User Stories

- As a GPU-kernel maintainer, I want a mixed-scale fixture and diagnostics so
  that acceptance collapse is detected before it becomes a production bug.
- As a scientific user, I want mixed NPF/droplet coagulation to conserve mass
  and match expected stochastic rates within documented tolerances.
- As an E3 reviewer, I want either a measured improvement or an explicit,
  reproducible limitation so that follow-on tracks can depend on known behavior.

## Parent Epic Context

Parent epic: E3. This feature follows E3-F1, which defines RNG API
compatibility and seed-once initialization. E3-F2 should reuse that RNG contract
and avoid reintroducing state-reset behavior while evaluating sampler changes.
