# E3-F2 Infrastructure Reuse

## Existing Implementation Targets

- `particula/gpu/kernels/coagulation.py`
  - `coagulation_step_gpu(...)` orchestrates validation, buffer allocation,
    RNG-state handling, and kernel launches.
  - The current rejection sampler computes a single box-level `k_max`, samples
    candidate pairs, and accepts with `kernel_value / k_max`.
  - The one-thread-per-box implementation avoids collision-pair write races and
    constrains how much extra sequential bin work is acceptable.
- `particula/gpu/dynamics/coagulation_funcs.py`
  - `brownian_kernel_pair_wp(...)` is the Warp primitive to reuse for selected
    pair rates and any fixed-bin majorant calculations.

## Existing Tests and Fixtures

- `particula/gpu/kernels/tests/coagulation_test.py`
  - Existing stochastic collision-rate tests provide expected-mean and
    expected-sigma comparison patterns.
  - Existing mass conservation tests verify collision application invariants.
  - Preallocated buffer tests cover `collision_pairs`, `n_collisions`, and
    `rng_states` validation/reuse patterns.
- `particula/gpu/tests/cuda_availability.py`
  - Reuse Warp CPU plus CUDA-if-available parametrization.

## CPU Prior Art

- `particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py`
  demonstrates radius-bin pair iteration and stochastic acceptance logic for the
  CPU particle-resolved path. Use it as design inspiration, not a direct GPU
  port.

## Documentation to Reuse

- `docs/Features/data-containers-and-gpu-foundations.md` for explicit transfer
  boundary language.
- `docs/Features/Roadmap/data-oriented-gpu.md` for E3 roadmap context and the
  mixed-scale acceptance-collapse concern.
