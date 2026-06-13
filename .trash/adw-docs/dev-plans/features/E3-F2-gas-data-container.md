# Feature E3-F2: Gas Data Container

**Parent Epic**: [E3: Data Representation Refactor](../epics/E3-data-representation-refactor.md)
**Status**: In Progress
**Priority**: P1
**Start Date**: TBD
**Last Updated**: 2026-02-14

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
        concentration: Mass concentrations in kg/m^3.
            Shape: (n_boxes, n_species)
        partitioning: Whether each species can partition to particles.
            Shape: (n_species,) - shared across boxes
        
    Example:
        ```python
        # Single-box simulation (n_boxes=1)
        gas = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.array([[1e-3, 1e-6, 1e-8]]),  # kg/m^3, (1, 3)
            partitioning=np.array([True, True, True]),
        )
        
        # Multi-box CFD (100 boxes, 3 species)
        cfd_gas = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.zeros((100, 3)),  # kg/m^3, different per box
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
        builder.set_concentration([1e-3, 1e-6], units="kg/m^3")
        builder.set_partitioning([True, True])
        gas = builder.build()
        
        # Multi-box case
        builder = GasDataBuilder()
        builder.set_n_boxes(100)
        builder.set_names(["Water", "Ammonia"])
        builder.set_molar_mass([18, 17], units="g/mol")
        builder.set_concentration(np.zeros((100, 2)))  # kg/m^3, already batched
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
        self, concentration, units: str = "kg/m^3"
    ) -> "GasDataBuilder":
        """Set concentrations with unit conversion.
        
        Args:
            concentration: Concentration values. If 1D (n_species,),
                batch dimension added. If 2D (n_boxes, n_species), used as-is.
            units: Input units (default "kg/m^3" for mass concentration,
                matching GasSpecies convention)
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

- [x] **E3-F2-P1**: Define `GasData` dataclass with batched fields
  - Issue: TBD | Size: M | Status: Shipped
  - Create `particula/gas/gas_data.py`
  - Define dataclass with name, molar_mass, concentration, partitioning
  - Shape convention: (n_boxes, n_species) for concentration
  - Add `__post_init__` validation for shape consistency
  - Add n_boxes, n_species properties and copy() method
  - Write `particula/gas/tests/gas_data_test.py`
  - Tests for instantiation, validation errors, properties

- [x] **E3-F2-P2**: Create `GasDataBuilder` with validation
  - Issue: TBD | Size: M | Status: Shipped
  - Create `particula/gas/gas_data_builder.py`
  - Implement setter methods with unit conversion
  - Auto-add batch dimension if 1D concentration provided
  - Add validation for positive molar mass, non-negative concentration
  - Write `particula/gas/tests/gas_data_builder_test.py`
  - Tests for valid builds, unit conversion, validation errors

- [x] **E3-F2-P3**: Add conversion utilities
  - Issue: TBD | Size: M | Status: Shipped
  - Implement `from_species()` in gas_data.py
  - Implement `to_species()` in gas_data.py
  - Handle single and multi-species cases
  - Extend test file with conversion round-trip tests
  - Tests for single/multi species conversion, multi-box

- [ ] **E3-F2-P4**: Revise concentration units from molecules/m³ to kg/m³
  - Issue: TBD | Size: M | Status: Not Started
  - **Rationale**: Align `GasData.concentration` with `GasSpecies` convention
    (kg/m³). The current molecules/m³ introduces unnecessary Avogadro
    scaling in `from_species()`/`to_species()` and mismatches the rest
    of the codebase where gas concentration is universally kg/m³
    (e.g., `GasSpeciesBuilder.set_concentration(..., "kg/m^3")`).
  - **`particula/gas/gas_data.py`** — 5 changes:
    1. Remove `from particula.util.constants import AVOGADRO_NUMBER`
       (line 36 — no longer needed)
    2. Update `GasData` docstring (line 57): change
       `"Number concentrations in molecules/m^3"` →
       `"Mass concentrations in kg/m^3"`
    3. Update module-level examples (lines 16, 25): change `1e15, 1e12,
       1e10` to realistic kg/m³ values (e.g., `1e-3, 1e-6, 1e-8`)
    4. `from_species()` (lines 181–190): Remove Avogadro conversion.
       Currently does `concentration_molecules = (concentration_kg / molar_mass) * AVOGADRO_NUMBER`.
       Replace with direct passthrough: just reshape
       `species.get_concentration()` (already kg/m³) into (n_boxes, n_species).
       Update docstring (line 143–144) to say "direct copy" not "converted".
    5. `to_species()` (lines 275–281): Remove Avogadro back-conversion.
       Currently does `concentration_kg = (concentration_molecules * molar_mass) / AVOGADRO_NUMBER`.
       Replace with direct passthrough: `concentration_kg = data.concentration[box_index, :]`.
       Update docstring (line 221) to say "direct copy" not "converted".
  - **`particula/gas/gas_data_builder.py`** — 3 changes:
    1. Change `set_concentration()` default (line 108):
       `units: str = "1/m^3"` → `units: str = "kg/m^3"`
    2. Change conversion target (lines 131–133):
       `"1/m^3"` → `"kg/m^3"` in the `get_unit_conversion()` call
    3. Update module docstring examples (lines 16, 28):
       `units="1/m^3"` → `units="kg/m^3"` and values to kg/m³
  - **`particula/gas/__init__.py`** — 1 change:
    - Update module docstring example (line 12):
      `.set_concentration([1e15], units="1/m^3")` →
      `.set_concentration([1e-3], units="kg/m^3")`
  - **`particula/gas/tests/gas_data_test.py`** — 4 areas:
    1. Remove `from particula.util.constants import AVOGADRO_NUMBER` (line 11)
    2. `TestGasDataInstantiation` + `TestGasDataCopy` + `TestGasDataProperties`:
       Change concentration values from molecules/m³ (`1e15, 1e12, 1e10,
       1e20, 2e20, 3e20`) to kg/m³ (`1e-3, 1e-6, 1e-8`) — the exact values
       don't matter for shape/validation tests, just need to be realistic
    3. `TestFromSpecies.test_concentration_unit_conversion()` (lines 280–300):
       Remove Avogadro assertion — assert that `gas_data.concentration[0, 0]`
       equals `concentration_kg` directly (1e-6 kg/m³ in, 1e-6 kg/m³ out)
    4. `TestToSpecies` (lines 303–418): Change test values from molecules/m³
       (`1e20, 2e20, 3e20`) to kg/m³ (`1e-6, 2e-6, 3e-6`), remove Avogadro
       back-conversion expectations — assert species concentration equals
       the GasData value directly
  - **`particula/gas/tests/gas_data_builder_test.py`** — 2 areas:
    1. `TestGasDataBuilderBasics.test_build_valid_single_box()` (line 24):
       Change `units="1/m^3"` → `units="kg/m^3"` (or omit since it's default)
    2. `TestGasDataBuilderUnits.test_concentration_units()` (lines 90–109):
       Replace `"1/cm^3" → "1/m^3"` parametrized test with
       `"g/m^3" → "kg/m^3"` parametrized test
  - **`particula/gpu/warp_types.py`** — 1 change:
    - Update `WarpGasData.concentration` docstring (line 104):
      `"Number concentrations in molecules/m^3"` →
      `"Mass concentrations in kg/m^3"`
  - **Tests must pass**: All existing gas module tests plus updated tests
  - **No new files**: This is a revision of existing implementation

## Testing Strategy

### Unit Tests

Location: `particula/gas/tests/`

| Test File | Coverage Target |
|-----------|----------------|
| `gas_data_test.py` | GasData class, validation, properties, from/to_species |
| `gas_data_builder_test.py` | Builder validation, unit conversion (kg/m³ default) |

### Test Cases

1. **Instantiation**: Valid inputs with kg/m³ values, shape validation failures
2. **Batch Dimension**: Single box (n_boxes=1), multi-box (n_boxes>1)
3. **Properties**: n_boxes, n_species
4. **Builder**: Unit conversion (kg/m³ default, g/m³ supported), auto batch dim, validation
5. **Conversion**: Round-trip with GasSpecies (direct kg/m³ passthrough, no Avogadro scaling)

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
3. Builder converts units and validates inputs (kg/m³ default)
4. Conversion to/from `GasSpecies` is lossless (direct kg/m³, no Avogadro scaling)
5. `GasData.concentration` uses kg/m³ matching `GasSpecies` convention
6. All tests pass with 80%+ coverage

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-19 | Initial feature document | ADW |
| 2026-01-19 | Simplified: added batch dimension, removed array-like operations | ADW |
| 2026-02-14 | Marked P1–P3 shipped; added P4 revision to change concentration from molecules/m³ to kg/m³ | ADW |
| 2026-02-14 | P4 revised with exact line references from code review; added WarpGasData docstring fix | ADW |
