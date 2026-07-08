# Open Questions

- Should the final marker names be `warp`, `cuda`, `gpu_parity`, and
  `stochastic`, or should the project prefer fewer broader markers?
- Is a new pytest command-line option needed for CUDA selection, or is
  CUDA-if-available parametrization plus markers sufficient?
- Should tolerance constants live in test helpers, or should tolerance policy
  remain documentation plus explicit per-test assertions?
- Which release document should own the final CUDA local/manual validation
  checklist if `.opencode/guides/testing_guide.md` is not sufficient?
- After E3-F1 and E3-F2 ship, do their final tests introduce additional helper
  patterns that should be folded into this policy?
- [ ] Should E3-F5-P5 wait for E3-F4 to finalize the public quick-start import
  path and troubleshooting wording, or is a follow-up wording pass acceptable if
  marker/helper policy lands first? (reviewer: plan-review-dependencies)
  - Open: Release-validation guidance and user-facing GPU workflow text will
    drift if policy docs freeze before the supported direct-kernel example path
    is settled.
