# Feature E3-F3: Warp Integration and GPU Kernels

**Parent Epic**: [E3: Data Representation Refactor](../epics/E3-data-representation-refactor.md)
**Status**: Planning
**Priority**: P2
**Start Date**: TBD
**Last Updated**: 2026-01-19

## Summary

Integrate NVIDIA Warp for GPU-accelerated particle simulations. Define
`@wp.struct` for GPU-side data, implement manual transfer control via
`to_warp()`/`from_warp()`, and create GPU kernels for particle-resolved
condensation and Brownian coagulation. Designed for multi-box CFD where
data stays GPU-resident across many timesteps.

## Goals

1. Define `@wp.struct WarpParticleData` matching ParticleData batch layout
2. Implement `to_warp()` with manual control (device, copy options)
3. Implement `from_warp()` with sync option for explicit transfers
4. Add `gpu_context()` helper for scoped GPU-resident simulations
5. Create GPU kernel for particle-resolved condensation
6. Create GPU kernel for Brownian coagulation
7. Benchmark and optimize for 100k+ particles per box

## Non-Goals

- `ArrayBackend` protocol abstraction (dropped - too much indirection)
- Automatic/lazy conversion (replaced with explicit manual control)
- Support for backends other than NumPy and Warp
- Automatic differentiation (future work)

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
    """GPU-side gas data container using Warp arrays."""
    molar_mass: wp.array(dtype=wp.float64)       # (n_species,)
    concentration: wp.array2d(dtype=wp.float64)   # (n_boxes, n_species)
    vapor_pressure: wp.array2d(dtype=wp.float64)  # (n_boxes, n_species)
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
    # Batched gas data
    gas_concentration: wp.array2d(dtype=wp.float64),  # (n_boxes, n_species)
    vapor_pressure: wp.array2d(dtype=wp.float64),     # (n_boxes, n_species)
    molar_mass: wp.array(dtype=wp.float64),           # (n_species,)
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
    """
    box_idx, particle_idx = wp.tid()
    n_species = masses.shape[2]
    
    # Skip inactive particles
    if concentration[box_idx, particle_idx] == 0.0:
        return
    
    # Compute radius from mass and density
    total_volume = 0.0
    for s in range(n_species):
        total_volume += masses[box_idx, particle_idx, s] / density[s]
    radius = wp.pow(3.0 * total_volume / (4.0 * 3.14159265359), 1.0 / 3.0)
    
    # Compute mass transfer for each species
    for s in range(n_species):
        # Get saturation vapor pressure and gas concentration
        p_sat = vapor_pressure[box_idx, s]
        c_gas = gas_concentration[box_idx, s]
        
        # ... Fuchs-Sutugin or transition regime physics ...
        # computed_transfer = ...
        
        mass_transfer[box_idx, particle_idx, s] = 0.0  # placeholder


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
    Each thread handles all particles in one box using Monte Carlo sampling.
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

- [ ] **E3-F3-P1**: Define `@wp.struct` data containers
  - Issue: TBD | Size: M | Status: Not Started
  - Create `particula/gpu/__init__.py` (new module)
  - Create `particula/gpu/warp_types.py`
  - Define `WarpParticleData` struct matching ParticleData batch layout
  - Define `WarpGasData` struct matching GasData batch layout
  - Write `particula/gpu/tests/warp_types_test.py`
  - Tests for struct creation, field access
  - Skip tests if Warp not available

- [ ] **E3-F3-P2**: Implement `to_warp()` with manual control
  - Issue: TBD | Size: M | Status: Not Started
  - Create `particula/gpu/conversion.py`
  - Implement `to_warp(data, device, copy)` function
  - Handle device selection, copy vs zero-copy
  - Add graceful error if Warp not available
  - Write `particula/gpu/tests/conversion_test.py`
  - Tests for device transfer, copy modes

- [ ] **E3-F3-P3**: Implement `from_warp()` and `gpu_context()`
  - Issue: TBD | Size: M | Status: Not Started
  - Implement `from_warp(gpu_data, sync)` function
  - Implement `gpu_context()` context manager
  - Write tests for round-trip conversion
  - Tests for sync behavior, context manager usage

- [ ] **E3-F3-P4**: GPU kernel for particle-resolved condensation
  - Issue: TBD | Size: L | Status: Not Started
  - Create `particula/gpu/kernels/condensation.py`
  - Implement `condensation_mass_transfer_kernel`
  - Implement `apply_mass_transfer_kernel`
  - Create `condensation_step_gpu()` high-level function
  - Write `particula/gpu/kernels/tests/condensation_test.py`
  - Tests comparing GPU vs CPU results (numerical equivalence)

- [ ] **E3-F3-P5**: GPU kernel for Brownian coagulation
  - Issue: TBD | Size: L | Status: Not Started
  - Create `particula/gpu/kernels/coagulation.py`
  - Implement `brownian_coagulation_kernel`
  - Create `coagulation_step_gpu()` high-level function
  - Write `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests comparing GPU vs CPU results (statistical equivalence)

- [ ] **E3-F3-P6**: Benchmark and optimize GPU kernels
  - Issue: TBD | Size: M | Status: Not Started
  - Create `particula/gpu/tests/benchmark_test.py`
  - Benchmark condensation: 1 box x 100k particles, 100 boxes x 1k particles
  - Benchmark coagulation: same configurations
  - Optimize memory access patterns if needed
  - Document performance characteristics
  - Mark as `@pytest.mark.slow` and `@pytest.mark.performance`

## Testing Strategy

### Unit Tests

Location: `particula/gpu/tests/`

| Test File | Coverage Target |
|-----------|----------------|
| `warp_types_test.py` | WarpParticleData, WarpGasData structs |
| `conversion_test.py` | to_warp(), from_warp(), gpu_context() |

Location: `particula/gpu/kernels/tests/`

| Test File | Coverage Target |
|-----------|----------------|
| `condensation_test.py` | GPU condensation kernel vs CPU |
| `coagulation_test.py` | GPU coagulation kernel vs CPU |
| `benchmark_test.py` | Performance benchmarks (slow) |

### Test Approach

1. **Skip if No Warp**: All GPU tests use `pytest.importorskip("warp")`
2. **Numerical Equivalence**: GPU results match CPU within tolerance (1e-10)
3. **Round-Trip**: `from_warp(to_warp(data))` equals original
4. **Multi-Box**: Test with n_boxes > 1 to verify batch dimension
5. **Performance**: GPU achieves 10x+ speedup for 100k particles

## Dependencies

- `numpy>=2.0.0` (existing)
- `nvidia-warp>=1.0.0` (optional, for GPU features)
- E3-F1: ParticleData (for conversion)
- E3-F2: GasData (for conversion)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Warp not installed | Optional dependency, skip GPU tests, document installation |
| GPU memory limits | Document max particles, batch processing for huge datasets |
| Numerical precision | Use float64, compare with tolerance, document precision |
| Kernel complexity | Start with simple physics, optimize iteratively |
| Multi-GPU | Start with single GPU, add multi-GPU in future |

## Success Criteria

1. `@wp.struct` definitions match ParticleData/GasData layout
2. `to_warp()`/`from_warp()` round-trip preserves data exactly
3. GPU kernels produce numerically equivalent results to CPU
4. Multi-box (n_boxes > 1) works correctly
5. GPU achieves 10x+ speedup for 100k particles per box
6. All tests pass (skip gracefully if Warp unavailable)

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-19 | Initial feature document | ADW |
| 2026-01-19 | Replaced ArrayBackend with manual transfer control, added batch dimension | ADW |
