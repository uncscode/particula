# Success Criteria

- Pytest markers and any selected options for Warp/GPU parity are registered in
  both `particula/conftest.py` and `pyproject.toml` without unknown-marker
  warnings.
- Warp CPU tests run consistently when Warp is installed.
- CUDA tests run automatically when CUDA is available to Warp and skip cleanly
  when unavailable.
- GPU device helper behavior is covered by co-located tests that do not require
  real CUDA.
- Coagulation and condensation GPU kernel tests use the standardized marker and
  helper policy where practical.
- Stochastic parity tolerance policy is documented, including aggregate
  seed/step behavior and `3 sigma` or equivalent tolerance bands.
- CUDA local/manual validation expectations are documented and do not make CUDA
  a required CI dependency.
- Existing benchmark gating via `--benchmark` remains intact.
- Focused validation commands pass or skip as expected in a CPU-only
  environment.
