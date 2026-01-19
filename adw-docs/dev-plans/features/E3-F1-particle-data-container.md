# Feature E3-F1: Particle Data Container

**Parent Epic**: [E3: Data Representation Refactor](../epics/E3-data-representation-refactor.md)
**Status**: Planning
**Priority**: P1
**Start Date**: TBD
**Last Updated**: 2026-01-19

## Summary

Create a simple data container (`ParticleData`) that isolates particle data from
behavior. Uses a standard Python dataclass with batch dimension built-in from
the start to support multi-box CFD simulations. No `dataclass_array` pattern -
just a clean container with validation.

## Goals

1. Define `ParticleData` dataclass with batch-aware array fields
2. Shape convention: `(n_boxes, n_particles, ...)` for multi-box support
3. Simple `__post_init__` validation, no array-like operations
4. Provide builder with validation and unit conversion
5. Enable conversion to/from existing `ParticleRepresentation`

## Non-Goals

- Array-like operations (slicing, reshape, stack) - not needed
- `dataclass_array` pattern - too complex for our use case
- GPU implementation (deferred to E3-F3)
- Changes to strategy pattern or behavior classes
- Breaking changes to existing APIs

## Design

### Core Data Container

```python
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class ParticleData:
    """Batched particle data container for multi-box simulations.
    
    Simple data container with batch dimension built-in. All per-particle
    arrays have shape (n_boxes, n_particles, ...) to support multi-box CFD.
    Single-box simulations use n_boxes=1.
    
    This is NOT a frozen dataclass - arrays can be updated in place for
    performance in tight simulation loops. Use copy() if immutability needed.
    
    Attributes:
        masses: Per-species masses in kg.
            Shape: (n_boxes, n_particles, n_species)
        concentration: Number concentration per particle.
            Shape: (n_boxes, n_particles)
            For particle-resolved: actual count (typically 1).
            For binned: number per m^3.
        charge: Particle charges (dimensionless integer counts).
            Shape: (n_boxes, n_particles)
        density: Material densities in kg/m^3.
            Shape: (n_species,) - shared across all boxes
        volume: Simulation volume per box in m^3.
            Shape: (n_boxes,)
    
    Example:
        ```python
        # Single-box simulation (n_boxes=1)
        data = ParticleData(
            masses=np.random.rand(1, 1000, 3) * 1e-18,  # 1000 particles, 3 species
            concentration=np.ones((1, 1000)),
            charge=np.zeros((1, 1000)),
            density=np.array([1000.0, 1200.0, 800.0]),
            volume=np.array([1e-6]),  # 1 cm^3
        )
        
        # Multi-box CFD (100 boxes)
        cfd_data = ParticleData(
            masses=np.zeros((100, 10000, 3)),
            concentration=np.ones((100, 10000)),
            charge=np.zeros((100, 10000)),
            density=np.array([1000.0, 1200.0, 800.0]),
            volume=np.ones(100) * 1e-6,
        )
        ```
    """
    
    masses: NDArray[np.float64]
    concentration: NDArray[np.float64]
    charge: NDArray[np.float64]
    density: NDArray[np.float64]
    volume: NDArray[np.float64]
    
    def __post_init__(self):
        """Validate array shapes are consistent."""
        # Validate batch dimensions match
        n_boxes = self.masses.shape[0]
        n_particles = self.masses.shape[1]
        
        if self.concentration.shape != (n_boxes, n_particles):
            raise ValueError(
                f"concentration shape {self.concentration.shape} doesn't match "
                f"expected ({n_boxes}, {n_particles})"
            )
        if self.charge.shape != (n_boxes, n_particles):
            raise ValueError(
                f"charge shape {self.charge.shape} doesn't match "
                f"expected ({n_boxes}, {n_particles})"
            )
        if self.volume.shape != (n_boxes,):
            raise ValueError(
                f"volume shape {self.volume.shape} doesn't match "
                f"expected ({n_boxes},)"
            )
        
        # Validate density is 1D (shared across boxes)
        if self.density.ndim != 1:
            raise ValueError(
                f"density must be 1D (n_species,), got shape {self.density.shape}"
            )
        
        # Validate n_species matches
        n_species = self.density.shape[0]
        if self.masses.shape[2] != n_species:
            raise ValueError(
                f"masses n_species {self.masses.shape[2]} doesn't match "
                f"density n_species {n_species}"
            )
    
    # Computed properties
    @property
    def n_boxes(self) -> int:
        """Number of simulation boxes (batch dimension)."""
        return self.masses.shape[0]
    
    @property
    def n_particles(self) -> int:
        """Number of particles per box."""
        return self.masses.shape[1]
    
    @property
    def n_species(self) -> int:
        """Number of chemical species."""
        return self.masses.shape[2]
    
    @property
    def radii(self) -> NDArray[np.float64]:
        """Compute particle radii from mass and density.
        
        Returns:
            Particle radii in meters. Shape: (n_boxes, n_particles)
        """
        # Volume per species: mass / density
        # Shape: (n_boxes, n_particles, n_species)
        volumes_per_species = self.masses / self.density
        
        # Total volume per particle
        # Shape: (n_boxes, n_particles)
        total_volume = np.sum(volumes_per_species, axis=-1)
        
        # Radius from volume: r = (3V / 4pi)^(1/3)
        return np.cbrt(3.0 * total_volume / (4.0 * np.pi))
    
    @property
    def total_mass(self) -> NDArray[np.float64]:
        """Total mass per particle (sum over species).
        
        Returns:
            Total mass in kg. Shape: (n_boxes, n_particles)
        """
        return np.sum(self.masses, axis=-1)
    
    @property
    def effective_density(self) -> NDArray[np.float64]:
        """Effective density per particle (volume-weighted average).
        
        Returns:
            Effective density in kg/m^3. Shape: (n_boxes, n_particles)
        """
        volumes_per_species = self.masses / self.density
        total_volume = np.sum(volumes_per_species, axis=-1)
        return np.divide(
            self.total_mass,
            total_volume,
            where=total_volume > 0,
            out=np.zeros_like(total_volume),
        )
    
    @property
    def mass_fractions(self) -> NDArray[np.float64]:
        """Mass fraction per species per particle.
        
        Returns:
            Mass fractions (sum to 1). Shape: (n_boxes, n_particles, n_species)
        """
        total = self.total_mass[..., np.newaxis]
        return np.divide(
            self.masses,
            total,
            where=total > 0,
            out=np.zeros_like(self.masses),
        )
    
    def copy(self) -> "ParticleData":
        """Create a deep copy of this ParticleData."""
        return ParticleData(
            masses=np.copy(self.masses),
            concentration=np.copy(self.concentration),
            charge=np.copy(self.charge),
            density=np.copy(self.density),
            volume=np.copy(self.volume),
        )
```

### Builder Pattern

```python
class ParticleDataBuilder:
    """Builder for creating validated ParticleData instances.
    
    Provides:
    - Unit conversion for all inputs
    - Automatic batch dimension handling (adds n_boxes=1 if needed)
    - Validation of array shapes and values
    - Sensible defaults
    
    Example:
        ```python
        # Simple single-box case
        builder = ParticleDataBuilder()
        builder.set_masses([[1e-18, 2e-18]], units="kg")  # 1 particle, 2 species
        builder.set_density([1000, 1200], units="kg/m^3")
        builder.set_concentration([1e9], units="1/m^3")
        data = builder.build()
        
        # Multi-box case
        builder = ParticleDataBuilder()
        builder.set_n_boxes(100)
        builder.set_n_particles(10000)
        builder.set_n_species(3)
        builder.set_density([1000, 1200, 800])
        builder.set_volume(1e-6)  # Same volume for all boxes
        data = builder.build()  # Initializes with zeros
        ```
    """
    
    def __init__(self):
        self._masses = None
        self._concentration = None
        self._charge = None
        self._density = None
        self._volume = None
        self._n_boxes = None
        self._n_particles = None
        self._n_species = None
    
    def set_masses(
        self, masses, units: str = "kg"
    ) -> "ParticleDataBuilder":
        """Set per-species masses with unit conversion.
        
        Args:
            masses: Mass array. If 2D (n_particles, n_species), batch dim added.
                    If 3D (n_boxes, n_particles, n_species), used as-is.
            units: Input units (default "kg")
        """
        ...
        return self
    
    def set_density(
        self, density, units: str = "kg/m^3"
    ) -> "ParticleDataBuilder":
        """Set species densities with unit conversion."""
        ...
        return self
    
    def set_concentration(
        self, concentration, units: str = "1/m^3"
    ) -> "ParticleDataBuilder":
        """Set particle concentrations with unit conversion."""
        ...
        return self
    
    def set_charge(self, charge) -> "ParticleDataBuilder":
        """Set particle charges (dimensionless)."""
        ...
        return self
    
    def set_volume(
        self, volume, units: str = "m^3"
    ) -> "ParticleDataBuilder":
        """Set simulation volume with unit conversion.
        
        Args:
            volume: Volume per box. Scalar applies to all boxes,
                    array specifies per-box volumes.
        """
        ...
        return self
    
    def set_n_boxes(self, n_boxes: int) -> "ParticleDataBuilder":
        """Set number of boxes for zero-initialized arrays."""
        ...
        return self
    
    def set_n_particles(self, n_particles: int) -> "ParticleDataBuilder":
        """Set number of particles for zero-initialized arrays."""
        ...
        return self
    
    def set_n_species(self, n_species: int) -> "ParticleDataBuilder":
        """Set number of species for zero-initialized arrays."""
        ...
        return self
    
    def build(self) -> ParticleData:
        """Build and return validated ParticleData.
        
        Raises:
            ValueError: If required fields are missing or invalid
        """
        ...
```

### Conversion Utilities

```python
def from_representation(
    rep: "ParticleRepresentation",
    n_boxes: int = 1,
) -> ParticleData:
    """Convert existing ParticleRepresentation to ParticleData.
    
    Handles all distribution strategy types by extracting the underlying
    arrays and adding batch dimension.
    
    Args:
        rep: Existing ParticleRepresentation instance
        n_boxes: Number of boxes to replicate data into (default 1)
    
    Returns:
        ParticleData with batch dimension
    """
    ...


def to_representation(
    data: ParticleData,
    strategy: "DistributionStrategy",
    activity: "ActivityStrategy",
    surface: "SurfaceStrategy",
    box_index: int = 0,
) -> "ParticleRepresentation":
    """Convert ParticleData back to ParticleRepresentation.
    
    Extracts single box from batched data and wraps with strategies.
    
    Args:
        data: ParticleData instance
        strategy: Distribution strategy for the representation
        activity: Activity strategy
        surface: Surface strategy
        box_index: Which box to extract (default 0)
    
    Returns:
        ParticleRepresentation for single box
    """
    ...
```

## Phase Checklist

- [ ] **E3-F1-P1**: Define `ParticleData` dataclass with batched fields
  - Issue: TBD | Size: M | Status: Not Started
  - Create `particula/particles/particle_data.py`
  - Define dataclass with masses, concentration, charge, density, volume
  - Shape convention: (n_boxes, n_particles, ...) for batch dimension
  - Add `__post_init__` validation for shape consistency
  - Add type hints using numpy.typing
  - Write `particula/particles/tests/particle_data_test.py`
  - Tests for instantiation, validation errors, shape checking

- [ ] **E3-F1-P2**: Add computed properties
  - Issue: TBD | Size: S | Status: Not Started
  - Add n_boxes, n_particles, n_species properties
  - Add radii property (computed from mass/density)
  - Add total_mass, effective_density, mass_fractions properties
  - Add copy() method for deep copying
  - Extend test file with property tests
  - Tests for computed values, edge cases (zero mass, etc.)

- [ ] **E3-F1-P3**: Create `ParticleDataBuilder` with validation
  - Issue: TBD | Size: M | Status: Not Started
  - Create `particula/particles/particle_data_builder.py`
  - Implement setter methods with unit conversion
  - Auto-add batch dimension if 2D arrays provided
  - Add validation for positive values, shape compatibility
  - Support zero-initialization via set_n_boxes/particles/species
  - Write `particula/particles/tests/particle_data_builder_test.py`
  - Tests for valid builds, unit conversion, validation errors

- [ ] **E3-F1-P4**: Add conversion utilities
  - Issue: TBD | Size: M | Status: Not Started
  - Implement `from_representation()` in particle_data.py
  - Implement `to_representation()` in particle_data.py
  - Handle all distribution strategy types
  - Extend test file with conversion round-trip tests
  - Tests for each strategy type conversion

## Testing Strategy

### Unit Tests

Location: `particula/particles/tests/`

| Test File | Coverage Target |
|-----------|----------------|
| `particle_data_test.py` | ParticleData class, validation, properties |
| `particle_data_builder_test.py` | Builder validation, unit conversion |

### Test Cases

1. **Instantiation**: Valid inputs, shape validation failures
2. **Batch Dimension**: Single box (n_boxes=1), multi-box (n_boxes>1)
3. **Properties**: n_boxes, n_particles, n_species, radii, total_mass
4. **Edge Cases**: Zero mass particles, single species, single particle
5. **Builder**: Unit conversion, auto batch dim, validation
6. **Conversion**: Round-trip with ParticleRepresentation

## Dependencies

- `numpy>=2.0.0` (existing)
- `particula.util.convert` (existing unit conversion)
- `particula.particles.representation` (for conversion utilities)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Shape mismatches | Strict validation in `__post_init__`, clear error messages |
| Batch dim confusion | Clear docstrings, builder auto-adds dim if needed |
| Migration complexity | Conversion utilities enable gradual adoption |

## Success Criteria

1. `ParticleData` instantiation works with batch dimension
2. Validation catches shape mismatches with clear errors
3. Computed properties (radii, total_mass, etc.) match existing behavior
4. Builder converts units and validates inputs
5. Conversion to/from `ParticleRepresentation` is lossless
6. All tests pass with 80%+ coverage

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-19 | Initial feature document | ADW |
| 2026-01-19 | Simplified: dropped dataclass_array, added batch dimension, removed array ops | ADW |
