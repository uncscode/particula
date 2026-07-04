# E2-F7 Infrastructure Reuse

## Existing GPU Condensation Path

- `particula/gpu/kernels/condensation.py`
  - `condensation_step_gpu` orchestrates the current explicit fixed-step Warp
    condensation path.
  - `condensation_mass_transfer_kernel` computes `mass_rate * time_step` into
    a fixed-shape `mass_transfer` buffer.
  - `apply_mass_transfer_kernel` applies in-place particle mass updates and
    clamps negative mass to zero.
  - `_validate_mass_transfer_buffer` and the optional `mass_transfer` argument
    already support preallocated buffer reuse, which is useful for graph
    capture.

## CPU Reference and Stability Prior Art

- `particula/dynamics/condensation/mass_transfer.py`
  - `get_mass_transfer_rate` and `get_mass_transfer` provide CPU reference
    equations and inventory limiting behavior.
- `particula/dynamics/condensation/mass_transfer_utils.py`
  - `calc_mass_to_change`, condensation limits, evaporation limits, and
    per-bin limits are reusable for stiffness metrics and parity expectations.
- `particula/dynamics/condensation/condensation_strategies.py`
  - `CondensationIsothermal` is the simultaneous explicit CPU reference.
  - `CondensationIsothermalStaggered` provides stability-oriented two-pass
    prior art, but its Python loops, optional randomness, and dynamic batches
    are not directly graph/autodiff friendly.
- `particula/dynamics/particle_process.py`
  - `MassCondensation.execute(..., sub_steps)` shows the existing public
    sub-step interface on the CPU side.

## Containers and Conversion

- `particula/gpu/warp_types.py`
  - `WarpParticleData` and `WarpGasData` define current fixed-shape fp64 GPU
    containers.
- No `WarpEnvironmentData` exists yet; E2-F7 should depend on E2-F2 for the CPU
  environment schema and E2-F3 for the GPU environment transfer boundary.
- `particula/gpu/conversion.py`
  - Existing CPU/GPU conversion patterns should be reused for any benchmark
    fixtures and future environment conversion hooks.

## Test Assets

- `particula/gpu/kernels/tests/condensation_test.py` for GPU condensation
  parity, multi-box coverage, and buffer validation.
- `particula/dynamics/condensation/tests/staggered_stability_test.py` for slow
  stability benchmark patterns.
- `particula/dynamics/condensation/tests/staggered_mass_conservation_test.py`
  and `mass_transfer_test.py` for conservation and limiter expectations.

## Documentation Assets

- `docs/Features/Roadmap/data-oriented-gpu.md` contains the T7 stiffness and
  graph-capture requirements.
- `docs/Features/Roadmap/warp-autodiff-limitations.md` records current Warp
  autodiff constraints and identifies deterministic condensation as an early
  target.
