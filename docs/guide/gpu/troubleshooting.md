# Troubleshooting

## Direct-kernel troubleshooting

Use
[`docs/examples/gpu/direct/gpu_direct_kernels_quick_start.py`](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_direct_kernels_quick_start.py)
as the canonical runnable reference when troubleshooting the low-level GPU
path.

- **Warp missing (`WARP_AVAILABLE == False`)**
  - The direct-kernel path is unavailable until Warp is installed.
  - The canonical quick-start keeps `particula.gpu.kernels` imports deferred
    and completes in a CPU-only documentation mode without importing kernel
    steps or pretending kernels ran.
- **CUDA unavailable**
  - CUDA is optional.
  - The default supported runnable path is Warp `device="cpu"`; only opt into
    `device="cuda"` when a CUDA device is actually available.
- **Explicit CPU↔GPU transfer boundary**
  - Kernels operate on Warp-backed container mirrors, not CPU `ParticleData`,
    `GasData`, or `EnvironmentData` objects directly.
  - Move simulation state across the boundary explicitly with `to_warp_*` and
    `from_warp_*`; kernels and runnables do not perform hidden simulation-state
    transfers or top-level fallback imports. Read-only device validation may
    synchronize and read back status to surface invalid values, including on
    CUDA.
- **Device mismatch across particle, gas, environment, and sidecar buffers**
  - Keep Warp arrays and sidecar buffers such as `rng_states` on the same
    device as the particle/gas/environment inputs used by the kernel call.
  - Treat device-mismatch `ValueError` failures as input validation, not as a
    signal that Particula will migrate arrays automatically.
- **Condensation scratch buffers**
  - Preserve leading `(n_boxes, ...)` layouts. Keep supplied
    `CondensationScratchBuffers` fields on the active device with their
    required stable `wp.float64` shapes: transfer fields use
    `(n_boxes, n_particles, n_species)` and property fields use `(n_boxes,)`.
    Only omitted fields allocate fallback storage independently.
  - Supplied fields preserve caller ownership and identity but are mutable
    work/output storage; do not expect a successful step to leave them
    unmodified.
- **Mixed `environment=` plus scalar `temperature`/`pressure` inputs**
  - Pass either `environment=` or direct temperature/pressure inputs.
  - Do not mix scalar or Warp-array `temperature` / `pressure` values with
    `environment=`; current kernels reject that combination explicitly.
  - When `environment=` is absent, provide both direct inputs. They must be
    positive finite physical values and any direct Warp arrays must be
    compatible with the active device.
- **Species metadata and thermodynamics sidecars**
  - Preserve ordered CPU gas-name metadata when restoring `GasData`, and keep
    thermodynamics-sidecar species order aligned with `gas.molar_mass`.
  - Confirm the configured water-species index addresses a valid species.
- **sequence validation transfer and energy diagnostics**
  - sequence validation inventory-limited applied transfers are expected bounded behavior, not
    CPU-strategy or runnable parity evidence.
  - `energy_transfer` is caller-owned whole-call inventory-limited output. Explicitly
    synchronize before observing it on the host.
- **Gas/environment restore expectations**
  - `from_warp_gas_data()` is intentionally lossy across the helper boundary:
    it restores ordered species names only when you supply them; otherwise it
    generates placeholder names such as `species_0`.
  - GPU-only helper state such as `vapor_pressure` is not restored onto CPU
    `GasData`.
  - `from_warp_environment_data(..., sync=False)` is an explicit expert path;
    manual synchronization remains the caller's responsibility before NumPy
    access.

### Focused reproduction commands

These focused commands provide distinct evidence; none establishes either of
the other evidence classes.

| Command | Evidence |
| --- | --- |
| `python docs/examples/gpu/direct/gpu_direct_kernels_quick_start.py` | Canonical explicit-transfer walkthrough. |
| `pytest examples_tests/gpu_direct_kernels_example_test.py -q` | Quick-start regression. |
| `pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror` | Primary direct CPU-oracle particle-mass/gas-concentration parity matrix. |
| `pytest particula/gpu/kernels/tests/condensation_stiffness_test.py -q -Werror` | Bounded direct-step stiffness coverage. |
| `pytest particula/gpu/kernels/tests/slot_management_test.py -q -Werror` | Fixed-slot activation mapping, caller-owned sidecars, and preflight state-safety coverage. |
| `pytest particula/gpu/dynamics/tests/coagulation_funcs_test.py -q -Werror` | Deterministic GPU coagulation pair-helper parity. |
| `pytest particula/gpu/kernels/tests/coagulation_test.py -q -Werror` | Direct coagulation coverage, including singleton sedimentation and ST1956 configurations, direct/environment inputs, caller-owned output/RNG behavior, conservation, and rejected-call state safety. |
| `pytest particula/gpu/kernels/tests/coagulation_validation_test.py -q -m "warp and gpu_parity" -Werror` | Fixed-mask deterministic/ownership evidence. |
| `pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and stochastic and not cuda" -Werror` | Stochastic evidence. |
| `pytest particula/integration_tests/condensation_latent_heat_conservation_test.py -q` | CPU integration/inventory-conservation evidence (separate particle-plus-gas inventory conservation checks); not direct-GPU validation. |
| `pytest particula/integration_tests/condensation_particle_resolved_test.py -q` | CPU integration evidence for particle-resolved condensation; not direct-GPU validation. |
| `pytest examples_tests/condensation_latent_heat_example_test.py -q -Werror` | CPU example runtime and latent-heat energy/bookkeeping checks; not direct-GPU validation. |

The required baseline is Warp `device="cpu"` when Warp is installed. The
parity matrix, inventory conservation checks, and latent-heat energy/bookkeeping
checks are separate: no one class proves either of the others.

**Optional/local CUDA evidence:**

```bash
pytest particula/gpu/kernels/tests/condensation_test.py -q -m "warp and cuda" -Werror
pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and cuda" -Werror
```

This is additive to the required Warp `device="cpu"` baseline and skips
cleanly when CUDA is unavailable.

The [GPU condensation parity walkthrough](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_condensation_parity_walkthrough.py)
and its [condensation parity walkthrough ownership record](../../contribute/validation/gpu/condensation-parity.md)
document bounded physics, conservation, and energy evidence only. They retain
the fixed-four-substep low-level direct-kernel boundary: Warp CPU is the
installed-Warp baseline and CUDA is optional additive evidence, not required
support. `energy_transfer` remains caller-owned, write-only diagnostic output,
not a return value or temperature feedback mechanism (`kg * J/kg = J`).

