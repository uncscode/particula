# Infrastructure Reuse

## Existing GPU Public Surface

- `particula/gpu/__init__.py` already exports `WARP_AVAILABLE`, explicit
  transfer helpers, and `gpu_context`.
- `particula/gpu/kernels/__init__.py` already exports
  `condensation_step_gpu`, `coagulation_step_gpu`, and lower-level kernel
  symbols.
- `particula/gpu/conversion.py` owns explicit CPU/Warp conversion helpers and
  the `gpu_context` convenience context manager.

## Existing Kernel Implementations

- `particula/gpu/kernels/condensation.py` provides `condensation_step_gpu` with
  explicit environment handling and no hidden data transfer.
- `particula/gpu/kernels/coagulation.py` provides `coagulation_step_gpu` with
  optional caller-owned buffers and RNG state inputs.
- `particula/gpu/kernels/environment.py` centralizes positive finite validation,
  device validation, shape checks, and rejection of mixed direct environment
  inputs with `environment=`.

## Tests to Reuse

- `particula/gpu/tests/warp_types_test.py` contains an existing top-level
  export test pattern.
- `particula/gpu/kernels/tests/condensation_test.py` validates condensation
  low-level behavior.
- `particula/gpu/kernels/tests/coagulation_test.py` validates coagulation
  low-level behavior.
- `particula/gpu/tests/data_containers_example_test.py` shows how to smoke-test
  runnable docs examples with and without Warp.
- `particula/gpu/tests/cuda_availability.py` supplies optional CPU/CUDA device
  discovery helpers for tests that should skip cleanly.

## Documentation and Example Patterns

- `docs/Examples/data_containers_and_gpu_foundations.py` demonstrates
  `WARP_AVAILABLE`, `ParticleData`, `GasData`, and transfer helper usage.
- `docs/Features/data-containers-and-gpu-foundations.md` is the canonical
  feature documentation for explicit transfer boundaries.
- `docs/Features/Roadmap/data-oriented-gpu.md` contains the parent epic and
  track requirements for direct low-level GPU kernel documentation.
