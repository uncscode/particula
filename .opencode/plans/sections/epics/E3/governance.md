# Governance

## Decision Owners

- GPU kernel correctness decisions should be reviewed by maintainers familiar
  with `particula.gpu.kernels` and stochastic coagulation behavior.
- Public API/export decisions should include review from maintainers owning the
  `particula.gpu` package boundary and user-facing documentation.
- Test-policy decisions should include review from maintainers responsible for
  CI, optional CUDA behavior, and benchmark markers.
- Latent-heat baseline decisions should include review from condensation-domain
  maintainers.

## Review Rules

- Any change to kernel call signatures or export paths must document backwards
  compatibility impact.
- Any stochastic assertion must state why exact equality is or is not expected.
- Any CUDA-specific test must skip cleanly when CUDA is unavailable.
- Any documentation example must be runnable or explicitly marked as conceptual.
- Any limitation discovered during E3-F2 or E3-F3 must be recorded in the
  roadmap rather than left as implicit test behavior.

## Decision Log Candidates

- Whether caller-supplied `rng_states` suppresses initialization by default.
- Whether low-level kernels are promoted to `particula.gpu` or documented under
  `particula.gpu.kernels` only.
- Whether one-thread-per-box is accepted for Epic C and what evidence triggers a
  future parallel-within-box feature.
- What stochastic tolerance language becomes canonical for GPU coagulation
  tests.

## Change Control

Child tracks may refine implementation details, but they must preserve the epic
guardrails: no high-level backend selection, no new GPU physics, no hidden
transfers, and optional CUDA.

## Plan Metadata Decisions

- Confirmed plan owner: Gorkowski.
- Confirmed plan start date: 2026-07-08.
- Target dates and phase issue numbers remain `TBD`/unassigned until
  implementation issues are generated.
