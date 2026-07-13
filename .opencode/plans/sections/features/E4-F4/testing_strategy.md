# Testing Strategy

Tests ship with each implementation phase in
`particula/gpu/dynamics/tests/condensation_funcs_test.py` or
`particula/gpu/kernels/tests/condensation_test.py`; coverage thresholds are
never lowered.

- **P1:** In `condensation_funcs_test.py` and `condensation_test.py`, compare
  fp64 Warp conductivity/resistance with CPU references. Cover finite,
  nonnegative, shape, dtype, device, and no-mutation failures.
- **P2:** In `condensation_test.py`, compare corrected rates with CPU, verify
  rate reduction, exact zero-latent fallback, four refreshes, determinism, and
  scratch identity.
- **P3:** In `condensation_test.py`, verify positive/negative/zero energy and
  whole-call per-box/species `Q = sum(bounded Δm * L)`, including clamped and
  isolated cases.
- **P4:** Compose E4-F1/F2/F3 on mandatory Warp CPU and optional CUDA with
  clean skips; retain scalar/environment API regressions.

Preserve issue #1272 bookkeeping tolerances (`rtol=1e-12`, `atol=1e-18`) where
fp64 representation permits. Record separate justified full-physics parity
tolerances; never reuse E4-F3's `5e-2` stiffness bound for energy. Maintain at
least 80% changed-code coverage and all repository thresholds.
