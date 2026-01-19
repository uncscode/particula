# Epic E3: Data Representation Refactor for Extensibility and GPU Backends

**Status**: Planning
**Priority**: P1
**Owners**: TBD
**Start Date**: 2026-01-19
**Target Date**: TBD
**Last Updated**: 2026-01-19
**Size**: Large (4 features, ~20 phases)

## Vision

Refactor how data is stored in particle representation and gas species to
isolate data representation from classes/methods. This enables:

1. **Extensibility Below**: Users can modify dataclasses directly, add custom
   fields, without breaking simulations. Data containers are pure data.

2. **Extensibility Above**: Users can use the API for differential
   optimizations, machine learning pipelines, and custom simulation workflows.

3. **GPU Acceleration**: Data representation is portable/convertible to NVIDIA
   Warp for GPU-accelerated simulation backends, starting with particle-resolved
   mass-speciated representation for condensation and Brownian coagulation.

4. **Multi-Box CFD Ready**: Batch dimension built-in from the start to support
   future multi-box CFD simulations on GPU where each box runs in parallel.

5. **Manual Transfer Control**: Explicit `to_warp()`/`from_warp()` conversion
   allows long GPU-resident simulations without CPU round-trips every timestep.

The refactor uses simple Python dataclasses with batch-aware array shapes.
No `dataclass_array` pattern - we pass raw NumPy arrays to operations and
use `@wp.struct` on the Warp side for GPU data grouping.

## Architecture Overview

### Current State (Tightly Coupled)

```
ParticleRepresentation
├── strategy: DistributionStrategy     # Behavior determines data interpretation
├── activity: ActivityStrategy         # Methods coupled with data
├── surface: SurfaceStrategy
├── distribution: NDArray              # Meaning depends on strategy!
├── density: NDArray
├── concentration: NDArray
├── charge: NDArray
└── volume: float

GasSpecies
├── name: str | NDArray
├── molar_mass: float | NDArray
├── concentration: float | NDArray
├── pure_vapor_pressure_strategy: VaporPressureStrategy | list
└── partitioning: bool
```

**Problems:**
- `distribution` array meaning depends on which strategy is active
- Strategies store parameters AND compute on external data
- Direct mutation via `add_mass()`, `add_concentration()`
- Cannot extend data fields without modifying core classes

### Target State (Decoupled Data + Operations)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Data Layer (Pure Data, Batched)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  @dataclass                                                             │
│  class ParticleData:                                                    │
│      """Batched particle data - simple container, no array operations."""│
│      masses: NDArray[np.float64]        # (n_boxes, n_particles, n_species)
│      concentration: NDArray[np.float64] # (n_boxes, n_particles)        │
│      charge: NDArray[np.float64]        # (n_boxes, n_particles)        │
│      density: NDArray[np.float64]       # (n_species,) shared           │
│      volume: NDArray[np.float64]        # (n_boxes,) per-box volume     │
│                                                                         │
│  @dataclass                                                             │
│  class GasData:                                                         │
│      """Batched gas species data - simple container."""                 │
│      name: list[str]                    # n_species names               │
│      molar_mass: NDArray[np.float64]    # (n_species,)                  │
│      concentration: NDArray[np.float64] # (n_boxes, n_species)          │
│      partitioning: NDArray[np.bool_]    # (n_species,)                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  Warp Conversion Layer (Manual Control)                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  @wp.struct                                                             │
│  class WarpParticleData:                                                │
│      """GPU-side particle data using Warp arrays."""                    │
│      masses: wp.array3d(dtype=wp.float64)       # (boxes, particles, species)
│      concentration: wp.array2d(dtype=wp.float64) # (boxes, particles)   │
│      charge: wp.array2d(dtype=wp.float64)        # (boxes, particles)   │
│      density: wp.array(dtype=wp.float64)         # (species,)           │
│      volume: wp.array(dtype=wp.float64)          # (boxes,)             │
│                                                                         │
│  # Explicit transfer control                                            │
│  def to_warp(data: ParticleData, device="cuda", copy=True) -> WarpParticleData
│  def from_warp(gpu_data: WarpParticleData, sync=True) -> ParticleData   │
│                                                                         │
│  # Context manager for GPU-resident simulations                         │
│  @contextmanager                                                        │
│  def gpu_context(data: ParticleData, device="cuda"):                    │
│      """Run entire simulation on GPU, transfer only at start/end."""   │
│      gpu_data = to_warp(data, device)                                   │
│      yield gpu_data                                                     │
│      # User calls from_warp() when ready                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Operations Layer                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  # CPU operations (existing pattern, works on raw arrays)               │
│  def add_mass(masses, concentration, density, added_mass) -> NDArray    │
│  def collide_pairs(distribution, concentration, indices) -> tuple       │
│                                                                         │
│  # GPU kernels via Warp (operate on WarpParticleData)                   │
│  @wp.kernel                                                             │
│  def condensation_kernel(                                               │
│      masses: wp.array3d(dtype=wp.float64),  # (boxes, particles, species)
│      radii: wp.array2d(dtype=wp.float64),   # (boxes, particles)        │
│      ...                                                                │
│  ): ...                                                                 │
│                                                                         │
│  @wp.kernel                                                             │
│  def coagulation_kernel(...): ...                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Facade Layer (Backward Compatibility)            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  class ParticleRepresentation:                                          │
│      """DEPRECATED: Facade for backward compatibility."""               │
│      _data: ParticleData  # Underlying data (single box = batch dim 1)  │
│      strategy: DistributionStrategy                                     │
│      activity: ActivityStrategy                                         │
│      surface: SurfaceStrategy                                           │
│                                                                         │
│      @deprecated("Use ParticleData directly")                           │
│      def add_mass(self, added_mass: NDArray) -> None:                   │
│          # Delegate to operations on _data                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Multi-Box CFD Usage Pattern

```python
# Initialize particle data for all boxes (batch dimension)
boxes = ParticleData(
    masses=np.zeros((n_boxes, max_particles, n_species)),
    concentration=np.ones((n_boxes, max_particles)),
    charge=np.zeros((n_boxes, max_particles)),
    density=np.array([1000.0, 1200.0]),  # shared across boxes
    volume=np.ones(n_boxes) * box_volume,
)

# Transfer to GPU once at start
gpu_boxes = to_warp(boxes, device="cuda")

# Run full simulation on GPU - no CPU round-trips
for timestep in range(10000):
    gpu_boxes = condensation_step(gpu_boxes, gas_data, dt)
    gpu_boxes = coagulation_step(gpu_boxes, dt)
    
    # Optional: periodic checkpointing
    if timestep % 1000 == 0:
        checkpoint = from_warp(gpu_boxes)
        save_checkpoint(checkpoint, timestep)

# Transfer back when done
result = from_warp(gpu_boxes)
```

## Scope

### In Scope

**E3-F1: Particle Data Container**
- Create `ParticleData` dataclass with batch dimension built-in
- Shape convention: `(n_boxes, n_particles, ...)` for multi-box CFD
- Simple validation in `__post_init__`, no array-like operations
- Create `ParticleDataBuilder` for ergonomic construction with units
- Add conversion utilities for existing `ParticleRepresentation`

**E3-F2: Gas Data Container**
- Create `GasData` dataclass with batch dimension for multi-box
- Shape convention: `(n_boxes, n_species)` for concentrations
- Create builders and conversion utilities
- Integration with existing `GasSpecies` and `Atmosphere`

**E3-F3: Warp Integration and GPU Kernels**
- Define `@wp.struct WarpParticleData` for GPU-side data grouping
- Implement `to_warp()` with manual transfer control (device, copy options)
- Implement `from_warp()` with sync option for explicit transfers
- Add `gpu_context()` helper for scoped GPU-resident simulations
- GPU kernels for particle-resolved condensation
- GPU kernels for Brownian coagulation

**E3-F4: Facade and Migration**
- Create facade layer for `ParticleRepresentation` backward compatibility
- Create facade layer for `GasSpecies` backward compatibility
- Add deprecation warnings with migration guidance
- Update dynamics modules to prefer new data containers
- Documentation and migration guide

### Out of Scope

- Changes to strategy pattern itself (strategies still define behavior)
- Full GPU acceleration of all operations (focus on particle-resolved)
- Breaking changes to public API (facade maintains compatibility)
- `dataclass_array` pattern (dropped - too complex, not needed)
- Array-like operations on data containers (slicing, reshape, stack)
- Automatic differentiation integration (future work, enabled by design)

## Dependencies

- **Internal**: Existing `ParticleRepresentation`, `GasSpecies`, `Aerosol`
- **External**: None for E3-F1, E3-F2, E3-F4 (pure Python/NumPy)
- **Future**: `nvidia-warp` (for E3-F3 GPU features, optional dependency)
- **Blockers**: None - can proceed incrementally

## Features

| ID | Name | Priority | Phases | Status |
|----|------|----------|--------|--------|
| E3-F1 | [Particle Data Container](../features/E3-F1-particle-data-container.md) | P1 | 4 | Planning |
| E3-F2 | [Gas Data Container](../features/E3-F2-gas-data-container.md) | P1 | 3 | Planning |
| E3-F3 | [Warp Integration and GPU Kernels](../features/E3-F3-backend-warp-integration.md) | P2 | 6 | Planning |
| E3-F4 | [Facade and Migration](../features/E3-F4-facade-migration.md) | P1 | 5 | Planning |

## Phase Overview

### E3-F1: Particle Data Container (4 phases)

- **E3-F1-P1**: Define `ParticleData` dataclass with batched array fields
  (masses, concentration, charge, density, volume) and `__post_init__`
  validation with tests
- **E3-F1-P2**: Add computed properties (n_boxes, n_particles, n_species,
  radii, total_mass, effective_density) with tests
- **E3-F1-P3**: Create `ParticleDataBuilder` with validation and unit conversion
  with tests
- **E3-F1-P4**: Add conversion utilities `from_representation()` and
  `to_representation()` with tests

### E3-F2: Gas Data Container (3 phases)

- **E3-F2-P1**: Define `GasData` dataclass with batched fields (name, molar_mass,
  concentration, partitioning) and validation with tests
- **E3-F2-P2**: Create `GasDataBuilder` with validation with tests
- **E3-F2-P3**: Add conversion utilities `from_species()` and `to_species()`
  with tests

### E3-F3: Warp Integration and GPU Kernels (6 phases)

- **E3-F3-P1**: Define `@wp.struct WarpParticleData` and `WarpGasData` with
  Warp array types matching batch dimensions with tests
- **E3-F3-P2**: Implement `to_warp()` with manual control (device, copy options)
  with tests
- **E3-F3-P3**: Implement `from_warp()` with sync option and `gpu_context()`
  helper with tests
- **E3-F3-P4**: Create GPU kernel for particle-resolved condensation with tests
- **E3-F3-P5**: Create GPU kernel for Brownian coagulation with tests
- **E3-F3-P6**: Benchmark and optimize GPU kernels with performance tests

### E3-F4: Facade and Migration (5 phases)

- **E3-F4-P1**: Create `ParticleRepresentation` facade over `ParticleData` with
  deprecation warnings and tests
- **E3-F4-P2**: Create `GasSpecies` facade over `GasData` with deprecation
  warnings and tests
- **E3-F4-P3**: Update `CondensationIsothermal` to accept both old and new data
  types with tests
- **E3-F4-P4**: Update `Coagulation` strategies to accept both old and new data
  types with tests
- **E3-F4-P5**: Documentation, migration guide, and update dev-docs

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds as shipped (80%+)
- **Self-Contained Tests**: Each phase includes `*_test.py` files
- **Test-First Completion**: Tests pass before phase completion
- **Backward Compatibility**: All existing tests must continue passing
- **Performance Benchmarks**: GPU kernels must demonstrate speedup over CPU

## Testing Strategy

### Unit Tests

- `particula/particles/tests/particle_data_test.py` - Data container behavior
- `particula/particles/tests/particle_data_builder_test.py` - Builder validation
- `particula/gas/tests/gas_data_test.py` - Gas data container behavior
- `particula/util/backends/tests/` - Backend protocol compliance

### Integration Tests

- Verify new data containers work with existing dynamics strategies
- Verify facade maintains backward compatibility
- Verify GPU kernels produce same results as CPU versions

### Performance Tests

- Benchmark particle-resolved condensation: CPU vs GPU
- Benchmark Brownian coagulation: CPU vs GPU
- Target: 10x+ speedup for 100k+ particles on GPU

## NVIDIA Warp Integration Notes

### Warp Basics

NVIDIA Warp provides:
- `wp.array`, `wp.array2d`, `wp.array3d`, `wp.array4d`: GPU arrays up to 4D
- `@wp.kernel`: JIT-compiled GPU kernels
- `@wp.struct`: Custom structured types for GPU data grouping
- Zero-copy via `__cuda_array_interface__` protocol
- `wp.from_numpy()` / `array.numpy()` for explicit transfers

### Warp Struct Definition

```python
import warp as wp

@wp.struct
class WarpParticleData:
    """GPU-side particle data container using Warp arrays.
    
    All arrays have batch dimension for multi-box CFD support.
    """
    masses: wp.array3d(dtype=wp.float64)        # (n_boxes, n_particles, n_species)
    concentration: wp.array2d(dtype=wp.float64)  # (n_boxes, n_particles)
    charge: wp.array2d(dtype=wp.float64)         # (n_boxes, n_particles)
    density: wp.array(dtype=wp.float64)          # (n_species,) - shared
    volume: wp.array(dtype=wp.float64)           # (n_boxes,)
```

### Manual Transfer Control API

```python
def to_warp(
    data: ParticleData,
    device: str = "cuda",
    copy: bool = True,
) -> WarpParticleData:
    """Transfer ParticleData to GPU with explicit control.
    
    Args:
        data: CPU-side particle data container
        device: Target device ("cuda", "cuda:0", "cpu")
        copy: If False, attempt zero-copy via __cuda_array_interface__
              (requires contiguous arrays on compatible device)
    
    Returns:
        GPU-resident WarpParticleData
        
    Example:
        # Transfer once at simulation start
        gpu_data = to_warp(particles, device="cuda")
        
        # Run many iterations on GPU
        for _ in range(10000):
            gpu_data = physics_step(gpu_data)
        
        # Transfer back when done
        result = from_warp(gpu_data)
    """
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
        gpu_data: GPU-resident particle data
        sync: If True, synchronize device before transfer to ensure
              all GPU operations have completed
    
    Returns:
        CPU-side ParticleData with NumPy arrays
    """
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
def gpu_context(data: ParticleData, device: str = "cuda"):
    """Context manager for scoped GPU-resident simulation.
    
    Transfers data to GPU on entry. User is responsible for calling
    from_warp() when ready to transfer back.
    
    Example:
        with gpu_context(particles) as gpu_data:
            for _ in range(1000):
                gpu_data = step_kernel(gpu_data)
            result = from_warp(gpu_data)
    """
    gpu_data = to_warp(data, device=device)
    yield gpu_data
```

### GPU Kernel Example (Batched Condensation)

```python
@wp.kernel
def condensation_step_kernel(
    # Batched particle data
    masses: wp.array3d(dtype=wp.float64),         # (n_boxes, n_particles, n_species)
    concentration: wp.array2d(dtype=wp.float64),  # (n_boxes, n_particles)
    density: wp.array(dtype=wp.float64),          # (n_species,)
    # Gas data
    gas_concentration: wp.array2d(dtype=wp.float64),  # (n_boxes, n_species)
    vapor_pressure: wp.array(dtype=wp.float64),       # (n_species,)
    # Parameters
    temperature: float,
    dt: float,
    # Output
    mass_transfer: wp.array3d(dtype=wp.float64),  # (n_boxes, n_particles, n_species)
):
    # 2D thread index: (box_idx, particle_idx)
    box_idx, particle_idx = wp.tid()
    
    # Skip inactive particles
    if concentration[box_idx, particle_idx] == 0.0:
        return
    
    # Compute radius from mass and density
    total_volume = 0.0
    for s in range(masses.shape[2]):
        total_volume += masses[box_idx, particle_idx, s] / density[s]
    radius = wp.pow(3.0 * total_volume / (4.0 * 3.14159265359), 1.0/3.0)
    
    # Compute mass transfer for each species
    for s in range(masses.shape[2]):
        # ... condensation physics ...
        mass_transfer[box_idx, particle_idx, s] = computed_transfer
```

## Extensibility Examples

### Extending Below (Adding Custom Fields)

```python
from dataclasses import dataclass
from particula.particles import ParticleData

@dataclass
class MyParticleData(ParticleData):
    """Extended particle data with custom fields for my simulation."""
    
    # Custom fields - same batch convention (n_boxes, n_particles, ...)
    temperature: NDArray[np.float64]        # (n_boxes, n_particles)
    composition_label: NDArray[np.int32]    # (n_boxes, n_particles)
    
    def __post_init__(self):
        super().__post_init__()
        # Add custom validation
        assert self.temperature.shape == self.concentration.shape
```

### Multi-Box CFD Simulation

```python
from particula.particles import ParticleData
from particula.gpu import to_warp, from_warp

# Initialize 100 boxes with 10k particles each, 3 species
n_boxes, n_particles, n_species = 100, 10000, 3

particles = ParticleData(
    masses=np.random.rand(n_boxes, n_particles, n_species) * 1e-18,
    concentration=np.ones((n_boxes, n_particles)),
    charge=np.zeros((n_boxes, n_particles)),
    density=np.array([1000.0, 1200.0, 800.0]),
    volume=np.ones(n_boxes) * 1e-6,  # 1 cm^3 per box
)

# Transfer to GPU once
gpu_particles = to_warp(particles, device="cuda")

# Run 10k timesteps entirely on GPU
for step in range(10000):
    gpu_particles = condensation_step(gpu_particles, gas_data, dt=0.001)
    gpu_particles = coagulation_step(gpu_particles, dt=0.001)
    
    # Periodic checkpoint (transfers to CPU)
    if step % 1000 == 0:
        checkpoint = from_warp(gpu_particles)
        save_checkpoint(checkpoint, step)

# Final result
result = from_warp(gpu_particles)
print(f"Total mass: {np.sum(result.masses):.3e} kg")
```

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing tests | Low | High | Facade maintains API compatibility |
| Performance regression | Medium | Medium | Benchmark at each phase |
| Warp compatibility issues | Medium | Low | Start with CPU-only, add GPU incrementally |
| Scope creep | Medium | Medium | Strict phase boundaries, defer optimizations |
| Complex migration | Medium | Medium | Clear deprecation path, detailed guide |
| Batch dimension overhead | Low | Low | Single-box uses n_boxes=1, minimal overhead |

## Success Metrics

1. **Extensibility**: Users can subclass data containers and add fields
2. **Performance**: GPU kernels achieve 10x+ speedup for 100k particles
3. **Multi-Box Ready**: Batch dimension works correctly for n_boxes > 1
4. **Manual Control**: `to_warp()`/`from_warp()` enables GPU-resident simulations
5. **Compatibility**: All existing tests pass without modification
6. **Adoption**: New code prefers data containers over legacy classes

## Design Decisions

### Why Not `dataclass_array`?

The Google `dataclass_array` pattern adds array-like operations (slicing,
reshape, stack) to dataclasses. We decided against this because:

1. **Unnecessary complexity**: Operations already work on raw NumPy arrays
2. **No real benefit**: We unpack to arrays for computations anyway
3. **Batch dimension suffices**: Multi-box is handled by leading dimension
4. **Warp has its own model**: `@wp.struct` handles GPU data grouping

### Why Batch Dimension from Start?

Building toward multi-box CFD requires batch dimension. Adding it later would
be a breaking change. Single-box simulations use `n_boxes=1` with minimal
overhead.

### Why Manual Transfer Control?

Automatic lazy conversion (transfer on every kernel call) is wasteful for long
GPU-resident simulations. Manual `to_warp()`/`from_warp()` lets users:

1. Transfer once at simulation start
2. Run thousands of timesteps on GPU
3. Transfer back only when needed (checkpoints, final result)

This matches patterns in production GPU simulation codes.

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-19 | Initial epic creation from collaborative research | ADW |
| 2026-01-19 | Dropped `dataclass_array` pattern, added batch dimension, manual transfer control | ADW |
