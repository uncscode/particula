# Feature E3-F2: Gas Data Container

**Parent Epic**: [E3: Data Representation Refactor](../epics/E3-data-representation-refactor.md)
**Status**: Planning
**Priority**: P1
**Start Date**: TBD
**Last Updated**: 2026-01-19

## Summary

Create a simple data container (`GasData`) for gas species that isolates data
from behavior (vapor pressure strategies). Uses batch dimension for multi-box
CFD support, matching the ParticleData convention.

## Goals

1. Define `GasData` dataclass with batch-aware array fields
2. Shape convention: `(n_boxes, n_species)` for concentrations
3. Provide builder with validation and unit conversion
4. Enable conversion to/from existing `GasSpecies`

## Non-Goals

- Array-like operations (slicing, stacking) - not needed
- GPU implementation (deferred to E3-F3)
- Changes to vapor pressure strategy pattern
- Breaking changes to existing APIs

## Design

### Core Data Container

```python
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class GasData:
    """Batched gas species data container for multi-box simulations.
    
    Simple data container with batch dimension built-in. Concentration
    arrays have shape (n_boxes, n_species) to support multi-box CFD.
    Single-box simulations use n_boxes=1.
    
    This is NOT a frozen dataclass - concentrations can be updated in place
    for performance. Use copy() if immutability needed.
    
    Attributes:
        name: Species names. List of strings, length n_species.
        molar_mass: Molar masses in kg/mol. Shape: (n_species,)
        concentration: Number concentrations in molecules/m^3.
            Shape: (n_boxes, n_species)
        partitioning: Whether each species can partition to particles.
            Shape: (n_species,) - shared across boxes
        
    Example:
        ```python
        # Single-box simulation (n_boxes=1)
        gas = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.array([[1e15, 1e12, 1e10]]),  # (1, 3)
            partitioning=np.array([True, True, True]),
        )
        
        # Multi-box CFD (100 boxes, 3 species)
        cfd_gas = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.zeros((100, 3)),  # Different per box
            partitioning=np.array([True, True, True]),
        )
        ```
    """
    
    name: list[str]
    molar_mass: NDArray[np.float64]
    concentration: NDArray[np.float64]
    partitioning: NDArray[np.bool_]
    
    def __post_init__(self):
        """Validate array shapes are consistent."""
        n_species = len(self.name)
        
        if self.molar_mass.shape != (n_species,):
            raise ValueError(
                f"molar_mass shape {self.molar_mass.shape} doesn't match "
                f"expected ({n_species},) from {n_species} species names"
            )
        if self.partitioning.shape != (n_species,):
            raise ValueError(
                f"partitioning shape {self.partitioning.shape} doesn't match "
                f"expected ({n_species},)"
            )
        if self.concentration.ndim != 2:
            raise ValueError(
                f"concentration must be 2D (n_boxes, n_species), got {self.concentration.ndim}D"
            )
        if self.concentration.shape[1] != n_species:
            raise ValueError(
                f"concentration n_species {self.concentration.shape[1]} doesn't match "
                f"{n_species} species names"
            )
    
    @property
    def n_boxes(self) -> int:
        """Number of simulation boxes (batch dimension)."""
        return self.concentration.shape[0]
    
    @property
    def n_species(self) -> int:
        """Number of gas species."""
        return len(self.name)
    
    def copy(self) -> "GasData":
        """Create a deep copy of this GasData."""
        return GasData(
            name=list(self.name),
            molar_mass=np.copy(self.molar_mass),
            concentration=np.copy(self.concentration),
            partitioning=np.copy(self.partitioning),
        )
```

### Builder Pattern

```python
class GasDataBuilder:
    """Builder for creating validated GasData instances.
    
    Provides:
    - Unit conversion for all inputs
    - Automatic batch dimension handling
    - Validation of array shapes and values
    
    Example:
        ```python
        # Single-box case
        builder = GasDataBuilder()
        builder.set_names(["Water", "Ammonia"])
        builder.set_molar_mass([18, 17], units="g/mol")
        builder.set_concentration([1e15, 1e12], units="1/m^3")
        builder.set_partitioning([True, True])
        gas = builder.build()
        
        # Multi-box case
        builder = GasDataBuilder()
        builder.set_n_boxes(100)
        builder.set_names(["Water", "Ammonia"])
        builder.set_molar_mass([18, 17], units="g/mol")
        builder.set_concentration(np.zeros((100, 2)))  # Already batched
        builder.set_partitioning([True, True])
        gas = builder.build()
        ```
    """
    
    def __init__(self):
        self._names = None
        self._molar_mass = None
        self._concentration = None
        self._partitioning = None
        self._n_boxes = None
    
    def set_names(self, names: list[str]) -> "GasDataBuilder":
        """Set species names."""
        ...
        return self
    
    def set_molar_mass(
        self, molar_mass, units: str = "kg/mol"
    ) -> "GasDataBuilder":
        """Set molar masses with unit conversion."""
        ...
        return self
    
    def set_concentration(
        self, concentration, units: str = "1/m^3"
    ) -> "GasDataBuilder":
        """Set concentrations with unit conversion.
        
        Args:
            concentration: Concentration values. If 1D (n_species,),
                batch dimension added. If 2D (n_boxes, n_species), used as-is.
            units: Input units (default "1/m^3" for number concentration)
        """
        ...
        return self
    
    def set_partitioning(
        self, partitioning: list[bool] | NDArray[np.bool_]
    ) -> "GasDataBuilder":
        """Set whether each species can partition."""
        ...
        return self
    
    def set_n_boxes(self, n_boxes: int) -> "GasDataBuilder":
        """Set number of boxes (for broadcasting 1D concentration)."""
        ...
        return self
    
    def build(self) -> GasData:
        """Build and return validated GasData.
        
        Raises:
            ValueError: If required fields are missing or invalid
        """
        ...
```

### Conversion Utilities

```python
def from_species(
    species: "GasSpecies",
    n_boxes: int = 1,
) -> GasData:
    """Convert existing GasSpecies to GasData.
    
    Args:
        species: Existing GasSpecies instance
        n_boxes: Number of boxes to replicate concentration into
    
    Returns:
        GasData with batch dimension
    """
    ...


def to_species(
    data: GasData,
    vapor_pressure_strategies: list,
    box_index: int = 0,
) -> "GasSpecies":
    """Convert GasData back to GasSpecies.
    
    Args:
        data: GasData instance
        vapor_pressure_strategies: Vapor pressure strategies for each species
        box_index: Which box to extract concentration from
    
    Returns:
        GasSpecies for single box
    """
    ...
```

## Phase Checklist

- [ ] **E3-F2-P1**: Define `GasData` dataclass with batched fields
  - Issue: TBD | Size: M | Status: Not Started
  - Create `particula/gas/gas_data.py`
  - Define dataclass with name, molar_mass, concentration, partitioning
  - Shape convention: (n_boxes, n_species) for concentration
  - Add `__post_init__` validation for shape consistency
  - Add n_boxes, n_species properties and copy() method
  - Write `particula/gas/tests/gas_data_test.py`
  - Tests for instantiation, validation errors, properties

- [ ] **E3-F2-P2**: Create `GasDataBuilder` with validation
  - Issue: TBD | Size: M | Status: Not Started
  - Create `particula/gas/gas_data_builder.py`
  - Implement setter methods with unit conversion
  - Auto-add batch dimension if 1D concentration provided
  - Add validation for positive molar mass, non-negative concentration
  - Write `particula/gas/tests/gas_data_builder_test.py`
  - Tests for valid builds, unit conversion, validation errors

- [ ] **E3-F2-P3**: Add conversion utilities
  - Issue: TBD | Size: M | Status: Not Started
  - Implement `from_species()` in gas_data.py
  - Implement `to_species()` in gas_data.py
  - Handle single and multi-species cases
  - Extend test file with conversion round-trip tests
  - Tests for single/multi species conversion, multi-box

## Testing Strategy

### Unit Tests

Location: `particula/gas/tests/`

| Test File | Coverage Target |
|-----------|----------------|
| `gas_data_test.py` | GasData class, validation, properties |
| `gas_data_builder_test.py` | Builder validation, unit conversion |

### Test Cases

1. **Instantiation**: Valid inputs, shape validation failures
2. **Batch Dimension**: Single box (n_boxes=1), multi-box (n_boxes>1)
3. **Properties**: n_boxes, n_species
4. **Builder**: Unit conversion, auto batch dim, validation
5. **Conversion**: Round-trip with GasSpecies

## Dependencies

- `numpy>=2.0.0` (existing)
- `particula.util.convert` (existing unit conversion)
- `particula.gas.species` (for conversion utilities)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Shape mismatches | Strict validation in `__post_init__` |
| Batch dim confusion | Clear docstrings, builder auto-adds dim |
| Migration complexity | Conversion utilities enable gradual adoption |

## Success Criteria

1. `GasData` instantiation works with batch dimension
2. Validation catches shape mismatches with clear errors
3. Builder converts units and validates inputs
4. Conversion to/from `GasSpecies` is lossless
5. All tests pass with 80%+ coverage

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-19 | Initial feature document | ADW |
| 2026-01-19 | Simplified: added batch dimension, removed array-like operations | ADW |
