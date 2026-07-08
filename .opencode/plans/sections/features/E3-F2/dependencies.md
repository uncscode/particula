# E3-F2 Dependencies

## Internal Dependencies

- **E3-F1:** RNG API compatibility and seed-once initialization contract.
  E3-F2 must build on the finalized handling of `rng_states` and should not
  reset RNG state in ways that undermine repeated-step stochastic tests.
- **Parent epic E3:** GPU Kernel Correctness and Low-Level API Hardening. This
  feature supplies low-level mixed-scale coagulation evidence for sibling tracks.

## Code Dependencies

- `particula/gpu/kernels/coagulation.py` for sampler implementation and public
  low-level API behavior.
- `particula/gpu/dynamics/coagulation_funcs.py` for Brownian pair-rate
  calculation.
- `particula/gpu/kernels/tests/coagulation_test.py` for regression,
  conservation, and stochastic tests.
- `particula/gpu/tests/cuda_availability.py` for optional CUDA parametrization.

## External Dependencies

- Warp runtime for GPU/CPU kernel execution.
- NumPy and pytest for fixtures, statistical assertions, and regression tests.

## Sequencing

1. Complete or coordinate with E3-F1 before relying on persisted RNG-state
   semantics.
2. Land the mixed-scale fixture and diagnostics before sampler hardening so the
   baseline is measurable.
3. Final documentation depends on implementation and comparison results.
