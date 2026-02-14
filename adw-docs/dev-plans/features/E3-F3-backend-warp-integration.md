# Feature E3-F3: Warp Integration and GPU Kernels

**Parent Epic**: [E3: Data Representation Refactor](../epics/E3-data-representation-refactor.md)
**Status**: In Progress
**Priority**: P2
**Start Date**: TBD
**Last Updated**: 2026-02-14

## Summary

Integrate NVIDIA Warp for GPU-accelerated particle simulations using a
**bottom-up approach**: first port all underlying physics functions (`get_*`)
to `@wp.func` with per-function parity tests, then compose them into full
GPU kernels for condensation and coagulation. Define `@wp.struct` for
GPU-side data, implement manual transfer control via `to_warp()`/`from_warp()`.
Designed for multi-box CFD where data stays GPU-resident across many timesteps.

## Goals

1. Define `@wp.struct WarpParticleData` and `WarpGasData` matching batch layouts
2. Implement `to_warp()`/`from_warp()` with manual transfer control
3. Add `gpu_context()` helper for scoped GPU-resident simulations
4. Port all underlying `get_*` property functions to `@wp.func` with tests
   (bottom-up: leaf functions first, then composites)
5. Assemble full GPU kernels for condensation and coagulation from tested
   `@wp.func` building blocks
6. Validate exact numerical parity vs Python/NumPy at every tier
7. Benchmark and optimize for 100k+ particles per box

## Non-Goals

- `ArrayBackend` protocol abstraction (dropped - too much indirection)
- Automatic/lazy conversion (replaced with explicit manual control)
- Support for backends other than NumPy and Warp
- Automatic differentiation (future work)

## Design Principles

### No Approximations: Exact Parity with Python/NumPy

**Every `@wp.kernel` and `@wp.func` must produce numerically identical results
to the equivalent Python/NumPy function.** The GPU implementation is a parallel
execution of the _same physics_, not a simplified or approximated version.

- If the Python version uses Fuchs-Sutugin transition regime physics, the Warp
  kernel uses the exact same formulation.
- If a Python function exists for a calculation (e.g., `radius_from_mass`,
  `cunningham_slip_correction`), the Warp `@wp.func` must match it exactly.
- Tolerance for GPU vs CPU comparison: `rtol=1e-10` (float64 precision).

### Constants from `particula.util.constants` — Never Hardcode

Physical constants (Boltzmann constant, Avogadro number, gas constant, etc.)
must **never** be written as literal values in `@wp.kernel` or `@wp.func` code.
Instead:

1. **Pass as kernel parameters** from `particula.util.constants` on the Python
   side when launching kernels.
2. **Mathematical constants** like π use the full-precision value
   `3.141592653589793` (matching `math.pi` exactly) — not truncated values
   like `3.14159265359`.
3. **Trivial arithmetic values** (`0.0`, `1.0`, `2.0`, `3.0`, `4.0`) used in
   formulas are fine as literals since they are not physical constants.

This ensures a single source of truth and makes GPU/CPU parity auditable.

### Testing via Lightweight Wrapper Kernels

Each `@wp.func` is tested by writing a small `@wp.kernel` in the test file
that calls the function and writes results to an output array. The test then
compares the GPU output against the equivalent Python/NumPy function using
`npt.assert_allclose(..., rtol=1e-10)`.

See the [Testing Guide](../../testing_guide.md#testing-nvidia-warp-kernels)
for the full pattern and checklist.

## Design

### Warp Struct Definition

```python
import warp as wp

@wp.struct
class WarpParticleData:
    """GPU-side particle data container using Warp arrays.
    
    All arrays have batch dimension for multi-box CFD support.
    Mirrors the shape convention of ParticleData.
    """
    masses: wp.array3d(dtype=wp.float64)        # (n_boxes, n_particles, n_species)
    concentration: wp.array2d(dtype=wp.float64)  # (n_boxes, n_particles)
    charge: wp.array2d(dtype=wp.float64)         # (n_boxes, n_particles)
    density: wp.array(dtype=wp.float64)          # (n_species,) - shared
    volume: wp.array(dtype=wp.float64)           # (n_boxes,)


@wp.struct
class WarpGasData:
    """GPU-side gas data container using Warp arrays.
    
    Concentration is in kg/m^3, matching the GasData/GasSpecies convention.
    """
    molar_mass: wp.array(dtype=wp.float64)       # (n_species,) kg/mol
    concentration: wp.array2d(dtype=wp.float64)   # (n_boxes, n_species) kg/m^3
    vapor_pressure: wp.array2d(dtype=wp.float64)  # (n_boxes, n_species) Pa
    partitioning: wp.array(dtype=wp.int32)        # (n_species,) as int for GPU
```

### Manual Transfer Control API

```python
def to_warp(
    data: ParticleData,
    device: str = "cuda",
    copy: bool = True,
) -> WarpParticleData:
    """Transfer ParticleData to GPU with explicit control.
    
    Use this for long GPU-resident simulations where you want to:
    1. Transfer data to GPU once at simulation start
    2. Run many timesteps without CPU round-trips
    3. Transfer back only when needed (checkpoints, final result)
    
    Args:
        data: CPU-side ParticleData container
        device: Target device ("cuda", "cuda:0", "cuda:1", "cpu")
        copy: If True (default), always copy data to device.
              If False, attempt zero-copy via __cuda_array_interface__
              when arrays are already on a compatible device.
    
    Returns:
        WarpParticleData with Warp arrays on specified device
    
    Example:
        # Transfer once, run 10k timesteps on GPU
        gpu_data = to_warp(particles, device="cuda")
        
        for _ in range(10000):
            gpu_data = condensation_step(gpu_data, gas, dt)
            gpu_data = coagulation_step(gpu_data, dt)
        
        # Transfer back when done
        result = from_warp(gpu_data)
    
    Raises:
        RuntimeError: If Warp is not available or device not found
    """
    import warp as wp
    
    return WarpParticleData(
        masses=wp.from_numpy(data.masses, dtype=wp.float64, device=device),
        concentration=wp.from_numpy(data.concentration, dtype=wp.float64, device=device),
        charge=wp.from_numpy(data.charge, dtype=wp.float64, device=device),
        density=wp.from_numpy(data.density, dtype=wp.float64, device=device),
        volume=wp.from_numpy(data.volume, dtype=wp.float64, device=device),
    )


def from_warp(
    gpu_data: WarpParticleData,
    sync: bool = True,
) -> ParticleData:
    """Transfer WarpParticleData back to CPU.
    
    Args:
        gpu_data: GPU-resident WarpParticleData
        sync: If True (default), synchronize device before transfer
              to ensure all GPU operations have completed. Set False
              only if you've already synchronized manually.
    
    Returns:
        CPU-side ParticleData with NumPy arrays
    
    Example:
        # After GPU simulation
        result = from_warp(gpu_data)
        
        # Or with manual sync for batched transfers
        wp.synchronize()
        data1 = from_warp(gpu_data1, sync=False)
        data2 = from_warp(gpu_data2, sync=False)
    """
    import warp as wp
    
    if sync:
        wp.synchronize()
    
    return ParticleData(
        masses=gpu_data.masses.numpy(),
        concentration=gpu_data.concentration.numpy(),
        charge=gpu_data.charge.numpy(),
        density=gpu_data.density.numpy(),
        volume=gpu_data.volume.numpy(),
    )


@contextmanager
def gpu_context(
    data: ParticleData,
    device: str = "cuda",
):
    """Context manager for scoped GPU-resident simulation.
    
    Transfers data to GPU on entry. User is responsible for calling
    from_warp() when ready to transfer back (typically inside the context
    or on exit).
    
    This is a convenience wrapper - for complex workflows, use
    to_warp()/from_warp() directly.
    
    Args:
        data: CPU-side ParticleData
        device: Target GPU device
    
    Yields:
        WarpParticleData on GPU
    
    Example:
        with gpu_context(particles) as gpu_data:
            for _ in range(1000):
                gpu_data = physics_step(gpu_data, dt)
            
            # Transfer back inside context
            result = from_warp(gpu_data)
        
        # Or keep reference and transfer after
        with gpu_context(particles) as gpu_data:
            for _ in range(1000):
                gpu_data = physics_step(gpu_data, dt)
        result = from_warp(gpu_data)  # Still valid
    """
    gpu_data = to_warp(data, device=device)
    yield gpu_data
```

### GPU Kernels (Batched)

```python
@wp.kernel
def condensation_mass_transfer_kernel(
    # Batched particle data
    masses: wp.array3d(dtype=wp.float64),         # (n_boxes, n_particles, n_species)
    concentration: wp.array2d(dtype=wp.float64),  # (n_boxes, n_particles)
    density: wp.array(dtype=wp.float64),          # (n_species,)
    # Batched gas data (concentration in kg/m^3, matching GasData convention)
    gas_concentration: wp.array2d(dtype=wp.float64),  # (n_boxes, n_species) kg/m^3
    vapor_pressure: wp.array2d(dtype=wp.float64),     # (n_boxes, n_species) Pa
    molar_mass: wp.array(dtype=wp.float64),           # (n_species,) kg/mol
    # Physical constants — passed from particula.util.constants, never hardcoded
    boltzmann_constant: float,  # from BOLTZMANN_CONSTANT
    # Parameters
    temperature: float,
    pressure: float,
    dt: float,
    # Output
    mass_transfer: wp.array3d(dtype=wp.float64),  # (n_boxes, n_particles, n_species)
):
    """Compute mass transfer for particle-resolved condensation.
    
    2D grid launch: (n_boxes, n_particles)
    Each thread handles one particle in one box.
    
    IMPORTANT: Physical constants are passed as kernel parameters from
    particula.util.constants — they must NOT be hardcoded in kernel code.
    The physics implemented here must exactly match the Python/NumPy
    equivalent in particula.dynamics.condensation (no approximations).
    """
    box_idx, particle_idx = wp.tid()
    n_species = masses.shape[2]
    
    # Skip inactive particles
    if concentration[box_idx, particle_idx] == 0.0:
        return
    
    # Compute radius from mass and density
    # Uses PI = 3.141592653589793 (matches math.pi exactly)
    PI = 3.141592653589793
    total_volume = 0.0
    for s in range(n_species):
        total_volume += masses[box_idx, particle_idx, s] / density[s]
    radius = wp.pow(3.0 * total_volume / (4.0 * PI), 1.0 / 3.0)
    
    # Compute mass transfer for each species
    for s in range(n_species):
        # Get saturation vapor pressure and gas mass concentration
        p_sat = vapor_pressure[box_idx, s]
        c_gas = gas_concentration[box_idx, s]  # kg/m^3
        
        # ... Fuchs-Sutugin or transition regime physics ...
        # Must match particula.dynamics.condensation Python implementation exactly
        # computed_transfer = ...
        
        mass_transfer[box_idx, particle_idx, s] = 0.0  # TODO: implement condensation mass transfer using proper Fuchs-Sutugin / transition-regime physics


@wp.kernel
def apply_mass_transfer_kernel(
    masses: wp.array3d(dtype=wp.float64),         # (n_boxes, n_particles, n_species)
    mass_transfer: wp.array3d(dtype=wp.float64),  # (n_boxes, n_particles, n_species)
    concentration: wp.array2d(dtype=wp.float64),  # (n_boxes, n_particles)
):
    """Apply mass transfer to particles, clamping to non-negative.
    
    2D grid launch: (n_boxes, n_particles)
    """
    box_idx, particle_idx = wp.tid()
    n_species = masses.shape[2]
    
    if concentration[box_idx, particle_idx] == 0.0:
        return
    
    for s in range(n_species):
        new_mass = masses[box_idx, particle_idx, s] + mass_transfer[box_idx, particle_idx, s]
        masses[box_idx, particle_idx, s] = wp.max(new_mass, 0.0)


@wp.kernel
def brownian_coagulation_kernel(
    # Batched particle data
    masses: wp.array3d(dtype=wp.float64),         # (n_boxes, n_particles, n_species)
    concentration: wp.array2d(dtype=wp.float64),  # (n_boxes, n_particles)
    density: wp.array(dtype=wp.float64),          # (n_species,)
    # Parameters
    temperature: float,
    viscosity: float,
    dt: float,
    volume: wp.array(dtype=wp.float64),           # (n_boxes,)
    # Random state per box
    rng_states: wp.array(dtype=wp.uint64),        # (n_boxes,)
    # Output: collision pairs per box
    collision_pairs: wp.array3d(dtype=wp.int32),  # (n_boxes, max_collisions, 2)
    n_collisions: wp.array(dtype=wp.int32),       # (n_boxes,) collision counts
):
    """Identify collision pairs for Brownian coagulation.
    
    1D grid launch: (n_boxes,)
    
    Design note: we intentionally use one thread per box rather than a
    2D launch over (n_boxes, n_particles). The Brownian coagulation
    algorithm here is formulated as a box-level Monte Carlo process:
    each thread keeps the RNG state, collision counters, and temporary
    accumulators local while iterating over the particles in its box.
    
    This provides ample parallelism across boxes for typical CFD cases
    (O(10^2–10^3) particles per box) and avoids additional complexity
    from per-particle synchronization / reductions that a 2D launch
    would require. If profiling later shows this kernel is a bottleneck,
    we can refactor to a 2D launch with per-particle work sharing.
    """
    box_idx = wp.tid()
    # ... stochastic collision detection for this box ...
```

### High-Level GPU Functions

```python
def condensation_step_gpu(
    particles: WarpParticleData,
    gas: WarpGasData,
    temperature: float,
    pressure: float,
    dt: float,
) -> WarpParticleData:
    """Execute one condensation timestep on GPU.
    
    Data stays on GPU - no CPU transfers.
    
    Args:
        particles: GPU-resident particle data
        gas: GPU-resident gas data
        temperature: Temperature in Kelvin
        pressure: Pressure in Pascals
        dt: Timestep in seconds
    
    Returns:
        Updated WarpParticleData (same arrays, modified in place)
    """
    n_boxes, n_particles, n_species = particles.masses.shape
    
    # Allocate output array
    mass_transfer = wp.zeros(
        (n_boxes, n_particles, n_species),
        dtype=wp.float64,
        device=particles.masses.device,
    )
    
    # Launch kernel
    wp.launch(
        kernel=condensation_mass_transfer_kernel,
        dim=(n_boxes, n_particles),
        inputs=[
            particles.masses,
            particles.concentration,
            particles.density,
            gas.concentration,
            gas.vapor_pressure,
            gas.molar_mass,
            temperature,
            pressure,
            dt,
        ],
        outputs=[mass_transfer],
    )
    
    # Apply mass transfer
    wp.launch(
        kernel=apply_mass_transfer_kernel,
        dim=(n_boxes, n_particles),
        inputs=[particles.masses, mass_transfer, particles.concentration],
    )
    
    return particles
```

## Phase Checklist

### Infrastructure (P1–P3) ✅

- [x] **E3-F3-P1**: Define `@wp.struct` data containers
  - Issue: TBD | Size: M | Status: Shipped
  - Created `particula/gpu/__init__.py` with lazy imports, `WARP_AVAILABLE` sentinel
  - Created `particula/gpu/warp_types.py`
  - Defined `WarpParticleData` struct matching ParticleData batch layout
  - Defined `WarpGasData` struct (excludes name strings, int32 partitioning,
    added vapor_pressure field for GPU kernels)
  - Written `particula/gpu/tests/warp_types_test.py` (17 tests)
  - Tests for struct creation, field shapes, dtypes, field access,
    single/multi-box, numpy round-trip
  - All tests skip if Warp not available
  - **Note**: `WarpGasData.concentration` docstring still says molecules/m³;
    will be corrected when E3-F2-P4 (kg/m³ revision) is implemented

- [x] **E3-F3-P2**: Implement `to_warp()` with manual control
  - Issue: TBD | Size: M | Status: Shipped
  - Created `particula/gpu/conversion.py`
  - Implemented `to_warp_particle_data(data, device, copy)` and
    `to_warp_gas_data(data, device, copy, vapor_pressure)` functions
  - Handles device selection, copy vs zero-copy (wp.from_numpy),
    graceful RuntimeError if Warp not available, device validation
  - Written `particula/gpu/tests/conversion_test.py`
  - Tests for default transfer, shape preservation, value integrity,
    copy independence, zero-copy, device selection, invalid device error

- [x] **E3-F3-P3**: Implement `from_warp()` and `gpu_context()`
  - Issue: TBD | Size: M | Status: Shipped
  - Implemented `from_warp_particle_data(gpu_data, sync)` and
    `from_warp_gas_data(gpu_data, name, sync)` functions
  - Implemented `gpu_context()` context manager
  - Handles sync behavior, placeholder name generation for gas data,
    bool↔int32 partitioning conversion, name length validation
  - Tests for round-trip conversion, sync true/false, context manager
    usage (transfer inside/after context, simulation loop pattern),
    multi-box scenarios, error handling

### Tier 1 — Leaf Physics Functions as `@wp.func` (P4–P5)

These are the foundational building blocks shared by both condensation and
coagulation. Each `@wp.func` is a direct port of an existing Python function.
Every function is tested via a lightweight wrapper `@wp.kernel` that compares
GPU output against the Python/NumPy equivalent with `rtol=1e-10`.

- [ ] **E3-F3-P4**: Port shared gas/particle property functions to `@wp.func`
  - Issue: TBD | Size: L | Status: Not Started
  - Create `particula/gpu/properties/__init__.py`
  - Create `particula/gpu/properties/gas_properties.py`:
    - `dynamic_viscosity_wp` ← `get_dynamic_viscosity`
      (Sutherland formula; constants `REF_VISCOSITY_AIR_STP`,
      `REF_TEMPERATURE_STP`, `SUTHERLAND_CONSTANT` passed as params)
    - `molecule_mean_free_path_wp` ← `get_molecule_mean_free_path`
      (depends on viscosity; `GAS_CONSTANT`, `MOLECULAR_WEIGHT_AIR` as params)
    - `partial_pressure_wp` ← `get_partial_pressure`
      (`GAS_CONSTANT` as param)
  - Create `particula/gpu/properties/particle_properties.py`:
    - `knudsen_number_wp` ← `get_knudsen_number` (pure division)
    - `cunningham_slip_correction_wp` ← `get_cunningham_slip_correction`
      (empirical coefficients 1.257, 0.4, 1.1 are model-specific constants,
      not physical constants — acceptable as literals with comment)
    - `aerodynamic_mobility_wp` ← `get_aerodynamic_mobility` (uses π)
    - `mean_thermal_speed_wp` ← `get_mean_thermal_speed`
      (`BOLTZMANN_CONSTANT` as param)
    - `friction_factor_wp` ← `get_friction_factor` (uses π)
  - Write `particula/gpu/properties/tests/gas_properties_test.py`
  - Write `particula/gpu/properties/tests/particle_properties_test.py`
  - Each test: wrapper `@wp.kernel` → compare vs Python function
  - Test on `"cpu"` (always) and `"cuda"` (skip if unavailable)

- [ ] **E3-F3-P5**: Port condensation-specific property functions to `@wp.func`
  - Issue: TBD | Size: M | Status: Not Started
  - Add to `particula/gpu/properties/particle_properties.py`:
    - `vapor_transition_correction_wp` ← `get_vapor_transition_correction`
      (Fuchs-Sutugin; coefficients 0.75, 0.283 are model-specific)
    - `kelvin_radius_wp` ← `get_kelvin_radius`
      (`GAS_CONSTANT` as param)
    - `kelvin_term_wp` ← `get_kelvin_term` (with safe exp clamping)
    - `partial_pressure_delta_wp` ← `get_partial_pressure_delta`
  - Add to `particula/gpu/properties/tests/particle_properties_test.py`
  - Each test: wrapper kernel → compare vs Python equivalent

### Tier 2 — Composite Functions as `@wp.func` (P6–P7)

These compose the Tier 1 functions into the intermediate calculations that
the full condensation and coagulation kernels require.

- [ ] **E3-F3-P6**: Port condensation composite functions to `@wp.func`
  - Issue: TBD | Size: M | Status: Not Started
  - Create `particula/gpu/dynamics/__init__.py`
  - Create `particula/gpu/dynamics/condensation_funcs.py`:
    - `first_order_mass_transport_k_wp` ← `get_first_order_mass_transport_k`
      (4πrDX; combines radius, vapor transition, diffusion coefficient)
    - `mass_transfer_rate_wp` ← `get_mass_transfer_rate`
      (K·Δp·M/(R·T); `GAS_CONSTANT` as param)
    - `diffusion_coefficient_wp` ← `get_diffusion_coefficient`
      (`BOLTZMANN_CONSTANT` as param; Stokes-Einstein D = kB·T·B)
  - Write `particula/gpu/dynamics/tests/condensation_funcs_test.py`:
    - Wrapper kernels for each `@wp.func`
    - Compare vs Python: `get_first_order_mass_transport_k`,
      `get_mass_transfer_rate`, `get_diffusion_coefficient`
    - Chained test: full condensation rate from raw inputs through
      all tiers, compared against `CondensationIsothermal.mass_transfer_rate`

- [ ] **E3-F3-P7**: Port coagulation composite functions to `@wp.func`
  - Issue: TBD | Size: M | Status: Not Started
  - Create `particula/gpu/dynamics/coagulation_funcs.py`:
    - `brownian_diffusivity_wp` ← `_brownian_diffusivity`
      (`BOLTZMANN_CONSTANT` as param)
    - `particle_mean_free_path_wp` ← `_mean_free_path_l` (uses π)
    - `g_collection_term_wp` ← `_g_collection_term`
      (Fuchs collection distance formula)
    - `brownian_kernel_pair_wp` ← per-pair form of `get_brownian_kernel`
      (Fuchs form: 4π(D₁+D₂)(r₁+r₂)/denominator — scalar pair version
      suitable for GPU thread-per-pair)
  - Write `particula/gpu/dynamics/tests/coagulation_funcs_test.py`:
    - Wrapper kernels for each `@wp.func`
    - Compare vs Python equivalents
    - Chained test: full Brownian kernel for a pair of particles from
      raw inputs through all tiers, compared against
      `get_brownian_kernel_via_system_state` for same pair

### Tier 3 — Full GPU Kernels (P8–P9)

These compose the `@wp.func` functions from P4–P7 into batched GPU kernels.
All physics is already ported and tested — these phases focus on the kernel
launch patterns, batch dimensions, and integration.

- [ ] **E3-F3-P8**: Batched GPU kernel for particle-resolved condensation
  - Issue: TBD | Size: L | Status: Not Started
  - Create `particula/gpu/kernels/__init__.py`
  - Create `particula/gpu/kernels/condensation.py`:
    - `condensation_mass_transfer_kernel`: 2D launch (n_boxes, n_particles),
      calls `@wp.func`s from P4–P6 to compute per-particle mass transfer
    - `apply_mass_transfer_kernel`: clamp masses to non-negative
    - `condensation_step_gpu()`: high-level function that orchestrates
      kernel launches, passes constants from `util.constants`
  - Write `particula/gpu/kernels/tests/condensation_test.py`:
    - End-to-end test: create ParticleData + GasData, run GPU step,
      compare vs Python `CondensationIsothermal` on same inputs
    - `npt.assert_allclose(gpu_result, cpu_result, rtol=1e-10)`
    - Test with n_boxes=1 and n_boxes>1
    - Test mass clamping (no negative masses)
    - Test inactive particle skipping (concentration=0)

- [ ] **E3-F3-P9**: Batched GPU kernel for Brownian coagulation
  - Issue: TBD | Size: L | Status: Not Started
  - Create `particula/gpu/kernels/coagulation.py`:
    - `brownian_coagulation_kernel`: 1D launch (n_boxes), box-level
      Monte Carlo process using `@wp.func`s from P4–P7 plus Warp RNG
    - `coagulation_step_gpu()`: high-level function
  - Write `particula/gpu/kernels/tests/coagulation_test.py`:
    - Kernel matrix test: compute full NxN kernel matrix via GPU,
      compare against `get_brownian_kernel_via_system_state`
    - Statistical test: run many coagulation steps with fixed seed,
      verify expected collision count distribution
    - Test with n_boxes > 1

### Benchmarks (P10) and Documentation (P11)

- [ ] **E3-F3-P10**: Benchmark and optimize GPU kernels
  - Issue: TBD | Size: M | Status: Not Started
  - Create `particula/gpu/tests/benchmark_test.py`
  - Benchmark condensation: 1 box × 100k particles, 100 boxes × 1k particles
  - Benchmark coagulation: same configurations
  - Benchmark individual `@wp.func` vs NumPy equivalents
  - Optimize memory access patterns if needed
  - Document performance characteristics
  - Mark as `@pytest.mark.slow` and `@pytest.mark.performance`

- [ ] **E3-F3-P11**: Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Update index.md and README.md with final status
  - Add completion notes and lessons learned
  - Update E3 epic with final phase outcomes

## Testing Strategy

### Unit Tests

Location: `particula/gpu/tests/`

| Test File | Coverage Target | Phase |
|-----------|----------------|-------|
| `warp_types_test.py` | WarpParticleData, WarpGasData structs | P1 |
| `conversion_test.py` | to_warp(), from_warp(), gpu_context() | P2–P3 |

Location: `particula/gpu/properties/tests/`

| Test File | Coverage Target | Phase |
|-----------|----------------|-------|
| `gas_properties_test.py` | dynamic_viscosity_wp, molecule_mean_free_path_wp, partial_pressure_wp | P4 |
| `particle_properties_test.py` | knudsen_number_wp, slip_correction_wp, aerodynamic_mobility_wp, mean_thermal_speed_wp, friction_factor_wp, vapor_transition_wp, kelvin_radius_wp, kelvin_term_wp, pressure_delta_wp | P4–P5 |

Location: `particula/gpu/dynamics/tests/`

| Test File | Coverage Target | Phase |
|-----------|----------------|-------|
| `condensation_funcs_test.py` | first_order_mass_transport_k_wp, mass_transfer_rate_wp, diffusion_coefficient_wp + chained condensation rate test | P6 |
| `coagulation_funcs_test.py` | brownian_diffusivity_wp, particle_mean_free_path_wp, g_collection_term_wp, brownian_kernel_pair_wp + chained kernel test | P7 |

Location: `particula/gpu/kernels/tests/`

| Test File | Coverage Target | Phase |
|-----------|----------------|-------|
| `condensation_test.py` | End-to-end batched condensation kernel vs CPU CondensationIsothermal | P8 |
| `coagulation_test.py` | End-to-end batched coagulation kernel vs CPU Brownian kernel | P9 |
| `benchmark_test.py` | Performance benchmarks (slow) | P10 |

### Test Approach

1. **Skip if No Warp**: All GPU tests use `pytest.importorskip("warp")`
2. **Lightweight Wrapper Kernels**: Each `@wp.func` is tested via a small
   `@wp.kernel` in the test file that calls the function and writes results
   to an output array — then compared against the Python/NumPy equivalent
3. **Exact Numerical Parity**: GPU results must match CPU within float64
   precision: `npt.assert_allclose(gpu, cpu, rtol=1e-10)` — no approximations
4. **No Hardcoded Constants**: Tests verify that kernel code uses constants
   from `particula.util.constants` (passed as parameters), not literal values
5. **Round-Trip**: `from_warp(to_warp(data))` equals original
6. **Multi-Box**: Test with n_boxes > 1 to verify batch dimension
7. **Dual Device**: Test on `"cpu"` (always available) and `"cuda"` (skipped
   if unavailable)
8. **Performance**: GPU achieves 10x+ speedup for 100k particles (benchmark
   tests only, marked `@pytest.mark.slow`)

## Dependencies

- `numpy>=2.0.0` (existing)
- `nvidia-warp>=1.0.0` (optional, for GPU features)
- E3-F1: ParticleData (for conversion, P1–P3)
- E3-F2: GasData (for conversion, P1–P3)
- `particula.gas.properties` — Python reference functions for P4 parity tests
- `particula.particles.properties` — Python reference functions for P4–P5 parity tests
- `particula.dynamics.condensation` — Python reference for P6, P8 parity tests
- `particula.dynamics.coagulation` — Python reference for P7, P9 parity tests
- `particula.util.constants` — single source of truth for all physical constants

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Warp not installed | Optional dependency, skip GPU tests, document installation |
| GPU memory limits | Document max particles, batch processing for huge datasets |
| Numerical precision | Use float64, compare with tolerance, document precision |
| Kernel complexity | Bottom-up approach: port+test leaf functions first, compose later |
| NxN matrix operations in Warp | `brownian_kernel_pair_wp` is scalar-pair form; NxN assembly done by 2D kernel launch |
| `scipy.interpolate` in particle-resolved coag | Need Warp-native interpolation or pre-computed lookup table; may defer to future phase |
| Multi-GPU | Start with single GPU, add multi-GPU in future |

## Success Criteria

1. `@wp.struct` definitions match ParticleData/GasData layout
2. `to_warp()`/`from_warp()` round-trip preserves data exactly
3. GPU kernels produce **exact numerical parity** with Python/NumPy
   equivalents (`rtol=1e-10`) — no approximations or shortcuts
4. No hardcoded physical constants in any `@wp.kernel` or `@wp.func` — all
   sourced from `particula.util.constants`
5. Every `@wp.func` tested via lightweight wrapper kernel compared against
   the equivalent Python function
6. Multi-box (n_boxes > 1) works correctly
7. GPU achieves 10x+ speedup for 100k particles per box
8. All tests pass (skip gracefully if Warp unavailable)

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-19 | Initial feature document | ADW |
| 2026-01-19 | Replaced ArrayBackend with manual transfer control, added batch dimension | ADW |
| 2026-02-14 | Updated WarpGasData and kernel docs to reflect kg/m³ concentration (aligns with E3-F2-P4 revision) | ADW |
| 2026-02-14 | Added Design Principles: exact parity, no hardcoded constants, lightweight wrapper kernel testing | ADW |
| 2026-02-14 | Expanded from 6 to 11 phases: bottom-up @wp.func porting (P4–P7) before kernel assembly (P8–P9) | ADW |
| 2026-02-14 | Marked P1–P3 shipped after verifying implementation matches plan | ADW |
