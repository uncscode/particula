# Open Questions

Status: reviewed and answered on 2026-07-08.

## Resolved Decisions

- Use explicit marker names for the distinct policy axes: `warp`, `cuda`,
  `gpu_parity`, and `stochastic`. They are more actionable than one broad GPU
  marker and align with optional CUDA plus always-available Warp CPU coverage.
- Add marker registration in `particula/conftest.py` and keep helper behavior
  for later phases. P1 confirmed that no new CUDA-selection option is needed;
  `--benchmark` remains the only pytest option and the only collection-affecting
  hook surface.
- Keep `cuda_available()` and `warp_devices()` stable while standardizing one
  shared CUDA-only skip string. P2 resolved this by exporting
  `CUDA_SKIP_REASON = "Warp/CUDA not available"` from
  `particula/gpu/tests/cuda_availability.py` and reusing it in benchmark skip
  helper coverage.
- Keep tolerance policy in documentation plus explicit per-test assertions. Add
  local helper constants only where repeated tests need the same tolerance. P3
  resolved the documentation side by publishing the shared policy in
  `.opencode/guides/testing_guide.md` and
  `docs/Features/Roadmap/data-oriented-gpu.md`.
- Apply the shared policy to real GPU suites without changing production code or
  making CUDA mandatory. P4 resolved this by adding module-level Warp marks,
  preserving `pytest.importorskip("warp")`, and applying targeted
  `gpu_parity`, `stochastic`, and `cuda` marks only to representative tests in
  coagulation, condensation, environment, and conversion coverage.
- `.opencode/guides/testing_guide.md` should own the release/manual validation
  checklist, with roadmap links for Epic C context. P3 now establishes the
  shared tolerance wording there; any later release-command expansion belongs to
  follow-up documentation work rather than unresolved policy semantics.
- P5 resolved the remaining release-command question by keeping the canonical
  policy in `.opencode/guides/testing_guide.md` and
  `docs/Features/Roadmap/data-oriented-gpu.md` only. `docs/contribute/`
  `CONTRIBUTING.md` stays unchanged because a broader contributor-doc refresh was
  out of scope for this docs-only closeout.
- Fold in helper patterns from E3-F1 and E3-F2 after those features ship.
- E3-F5-P5 does not need to block on E3-F4. Marker/helper policy may land first
  if release-validation wording includes a follow-up pass after E3-F4 finalizes
  the public quick-start import path and troubleshooting text.
