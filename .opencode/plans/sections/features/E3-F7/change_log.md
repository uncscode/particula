# Change Log

## 2026-07-08

- Drafted E3-F7 feature plan for the CPU integration-level latent-heat
  conservation baseline.
- Added three issue-sized phases covering fixture adaptation, conservation and
  energy assertions, and CPU-only Epic D baseline documentation.
- Captured dependency on E3-F6 and explicitly preserved the constraint that
  this feature makes no GPU latent-heat production parity claim.

## 2026-07-08 — Completeness Review

- Expanded success criteria into measurable pass/fail checks, evidence metrics,
  and a definition of done centered on a deterministic single-species CPU
  baseline, conservation assertions, and CPU-only reference documentation.

## 2026-07-11

- Updated the plan sections to reflect shipped issue #1267 / E3-F7-P1 work.
- Recorded that
  `particula/integration_tests/condensation_latent_heat_conservation_test.py`
  now provides the CPU-only integration baseline via public `particula` APIs,
  a constant latent-heat strategy, and `MassCondensation.execute()`.
- Noted that P1 shipped only lightweight execution/bookkeeping assertions and
  did not change production code or user-facing documentation.
