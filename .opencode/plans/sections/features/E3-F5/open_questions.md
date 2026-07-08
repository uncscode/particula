# Open Questions

Status: reviewed and answered on 2026-07-08.

## Resolved Decisions

- Use explicit marker names for the distinct policy axes: `warp`, `cuda`,
  `gpu_parity`, and `stochastic`. They are more actionable than one broad GPU
  marker and align with optional CUDA plus always-available Warp CPU coverage.
- Add marker registration and helper behavior in `particula/conftest.py`. A new
  CUDA-selection option is optional; start with CUDA-if-available
  parametrization and add an option only if E3-F5 implementation needs forced
  CUDA selection.
- Keep tolerance policy in documentation plus explicit per-test assertions. Add
  local helper constants only where repeated tests need the same tolerance.
- `.opencode/guides/testing_guide.md` should own the release/manual validation
  checklist, with roadmap links for Epic C context.
- Fold in helper patterns from E3-F1 and E3-F2 after those features ship.
- E3-F5-P5 does not need to block on E3-F4. Marker/helper policy may land first
  if release-validation wording includes a follow-up pass after E3-F4 finalizes
  the public quick-start import path and troubleshooting text.
