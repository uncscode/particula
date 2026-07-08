# Dependencies

## Internal Dependencies

- **E3-F1:** Seed-once persisted RNG behavior for GPU coagulation. E3-F5
  should align stochastic parity policy with the final RNG contract and any
  persisted `rng_states` fixture patterns.
- **E3-F2:** Mixed-scale stochastic sampling hardening. E3-F5 should encode
  any accepted mixed-scale limitations or statistical validation approach in the
  documented tolerance policy.
- **E3-F3:** Coagulation benchmark evidence may inform which tests remain
  benchmark/performance-marked versus default parity tests.
- **E3-F4:** Low-level kernel API documentation and examples should use the
  same CUDA-optional validation language.
  Marker/helper policy work does not need to block on final E3-F4 quick-start
  wording; plan a follow-up wording pass once that path is finalized.

## Code Dependencies

- `particula/conftest.py` for pytest hooks and marker registration.
- `pyproject.toml` for pytest marker configuration.
- `particula/gpu/tests/cuda_availability.py` for device discovery helpers.
- GPU kernel test modules under `particula/gpu/kernels/tests/`.
- Existing GPU conversion and data-container tests under `particula/gpu/tests/`.

## External Dependencies

- `pytest` marker, option, and collection hooks.
- Optional `warp` package; tests should skip cleanly when absent.
- Optional CUDA device exposed through Warp; CUDA paths should skip cleanly when
  unavailable.

## Dependency Risks

- If E3-F1 or E3-F2 changes stochastic expectations after E3-F5 drafts its
  tolerance language, the policy text must be revisited before shipping.
- If pytest marker names are changed late, both config locations and docs must
  be updated together to avoid unknown-marker warnings.
