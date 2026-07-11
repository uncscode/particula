## Change Log

### 2026-07-08

- Created first-pass feature plan for `E3-F6`.
- Added three issue-sized phases covering runnable example creation, notebook
  and index documentation, and final validation/CPU-only guardrails.
- Captured codebase research for `CondensationLatentHeat`, latent heat
  factories/builders, `MassCondensation`, existing condensation examples, and
  validation commands.
- Recorded constraint that the feature is CPU-only and must not claim GPU
  latent-heat parity.

### 2026-07-08 — Completeness Review

- Replaced generic success bullets with measurable pass/fail criteria and
  validation evidence for runnable execution, energy bookkeeping,
  discoverability, notebook hygiene, and CPU-only scope control.

### 2026-07-11 — Issue #1263 implementation update

- Recorded that `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py`
  shipped as a runnable CPU-only latent-heat example.
- Recorded shipped smoke/invariant coverage in
  `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`.
- Marked notebook artifact generation and Dynamics docs index wiring as
  intentionally deferred from the shipped issue scope.
