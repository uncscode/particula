# Testing Strategy

Tests ship with each implementation phase in
`particula/gpu/dynamics/tests/condensation_funcs_test.py` or
`particula/gpu/kernels/tests/condensation_test.py`; coverage thresholds are
never lowered.

- **P1 (shipped, issue #1297):** `condensation_funcs_test.py` compares fp64
  Warp conductivity, resistance, and latent-corrected rates with CPU references
  for positive and negative pressure deltas, and verifies exact zero-latent
  isothermal identity. `_condensation_test_support.py` covers valid single- and
  multi-species sidecars, both sidecars together, finite/nonnegative/shape/
  dtype/device/non-Warp failures, and atomic no-allocation/no-launch/
  no-mutation preflight. CUDA device mismatch remains optional with clean skips.
- **P2 (shipped, issue #1298):**
  `_condensation_test_support.py`, exercised by `condensation_test.py`, extends
  the CPU four-substep oracle with the same shared surface pressure and latent
  rate. Regression coverage verifies multi-species activity/Kelvin parity,
  reduced condensing rate, all-zero and isolated zero-latent exact isothermal
  behavior, coupled mixed-latent oracle parity, sidecar immutability, malformed
  latent/thermal-work atomic rejection, four launch ordering, determinism, and
  complete caller-owned scratch/returned-total identity reuse. Optional CUDA
  remains cleanly skippable.
- **P3 (shipped, issue #1299):** `_condensation_test_support.py`, exercised by
  `condensation_test.py`, verifies caller-output identity/reuse and overwrite,
  valid NaN/Inf write-only storage, four-substep condensation/evaporation/zero/
  clamp oracle parity, and box/species aggregation. It also verifies atomic
  failures for missing latent heat and invalid output metadata, no energy
  kernels when omitted, and optional `cuda`/`gpu_parity` multi-box oracle
  coverage with clean skips.
- **P4:** Compose E4-F1/F2/F3 on mandatory Warp CPU and optional CUDA with
  clean skips; retain scalar/environment API regressions.

Preserve issue #1272 bookkeeping tolerances (`rtol=1e-12`, `atol=1e-18`) where
fp64 representation permits. Record separate justified full-physics parity
tolerances; never reuse E4-F3's `5e-2` stiffness bound for energy. Maintain at
least 80% changed-code coverage and all repository thresholds.
