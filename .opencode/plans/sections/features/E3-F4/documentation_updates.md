# Documentation Updates

## Quick-Start Example

Shipped in this phase: the canonical direct low-level GPU kernel quick-start is
now `docs/Examples/gpu_direct_kernels_quick_start.py`. It is runnable as a
script and shows:

- Availability gate with `WARP_AVAILABLE`.
- Minimal `ParticleData` and `GasData` setup.
- Explicit CPU-to-Warp transfers.
- A direct `condensation_step_gpu` call.
- A direct `coagulation_step_gpu` call.
- Explicit Warp-to-CPU transfer for inspection.
- `device="cpu"` as the default accessible execution target.
- Deferred `particula.gpu.kernels` import until the Warp-enabled execution
  branch.
- Caller-owned coagulation `rng_states` initialized with
  `initialize_rng=True`, `rng_seed=41` for the example call.

## Feature Documentation

Phase `E3-F4-P4` shipped the broader guide alignment in:

- `docs/Features/data-containers-and-gpu-foundations.md`
- `docs/Features/Roadmap/data-oriented-gpu.md`

Those updates now explicitly link the canonical quick-start at
`docs/Examples/gpu_direct_kernels_quick_start.py`, restate that supported
direct-step imports come from `particula.gpu.kernels`, and keep top-level
`particula.gpu` scoped to `WARP_AVAILABLE` plus transfer helpers such as
`to_warp_*` / `from_warp_*`.

## Roadmap Documentation

`docs/Features/Roadmap/data-oriented-gpu.md` now records the shipped import-path
decision for phase `E3-F4-P1`: import direct step functions from
`particula.gpu.kernels`, keep top-level `particula.gpu` focused on Warp
availability and transfer helpers, and leave raw helper kernels out of the
documented package-level surface.

## Current Phase Status

- Import-path decision documentation is shipped.
- Focused export regression coverage is shipped and centralized in
  `particula/gpu/tests/kernel_exports_test.py`, including the exact
  `particula.gpu.kernels.__all__` contract, negative package-export checks for
  representative internal helpers, and top-level `particula.gpu`
  non-reexport assertions.
- Duplicate package-export checks were removed from
  `particula/gpu/kernels/tests/coagulation_test.py` as part of the shipped test
  consolidation.
- Runnable quick-start is shipped at
  `docs/Examples/gpu_direct_kernels_quick_start.py`.
- Adjacent example smoke coverage is shipped at
  `particula/gpu/tests/gpu_direct_kernels_example_test.py`, including no-Warp,
  import-deferral, `__main__` / subprocess, Warp CPU, and failure-path checks.
- Broader troubleshooting and cross-document link updates are now shipped in
  the foundation guide and GPU roadmap.

## Troubleshooting Content

Shipped user-facing troubleshooting now covers:

- `WARP_AVAILABLE` is false: install/configure `warp-lang`, or run CPU-only
  non-GPU examples.
- CUDA unavailable: use `device="cpu"`; CUDA snippets are optional.
- Device mismatch: particle, gas, environment, and buffers must be on the same
  Warp device.
- Mixed environment inputs: do not combine scalar `temperature`/`pressure` with
  `environment=`.
- Explicit transfer boundaries: kernels operate on Warp data and do not move
  CPU containers automatically.
- Transfer-helper restores are intentionally lossy in specific cases: callers
  must preserve gas species names explicitly, helper-only GPU state is not
  restored onto CPU `GasData`, and no hidden synchronization is implied.
