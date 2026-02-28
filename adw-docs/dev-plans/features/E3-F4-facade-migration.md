# Feature E3-F4: Facade and Migration

**Parent Epic**: [E3: Data Representation Refactor](../epics/E3-data-representation-refactor.md)
**Status**: Shipped
**Priority**: P1
**Start Date**: TBD
**Last Updated**: 2026-02-28

## Summary

Create facade layers that maintain backward compatibility for existing
`ParticleRepresentation` and `GasSpecies` APIs while using the new data
containers internally. Add deprecation warnings with clear migration guidance
and update dynamics modules to support both old and new data types.

## Goals

1. Create `ParticleRepresentation` facade over `ParticleData`
2. Create `GasSpecies` facade over `GasData`
3. Update `CondensationIsothermal` to accept both data types
4. Update `Coagulation` strategies to accept both data types
5. Provide documentation and migration guide

## Non-Goals

- Remove old APIs (deprecate only, keep functional)
- Force immediate migration (gradual adoption)
- Change external behavior (facade is transparent)

## Design

### Facade Pattern

The facade wraps the new data container while exposing the old API:

```python
import warnings
from particula.particles.particle_data import ParticleData

class ParticleRepresentation:
    """Particle representation with backward-compatible API.
    
    DEPRECATED: This class is deprecated in favor of ParticleData.
    Use ParticleData for new code. This facade will be removed in v1.0.
    
    The facade wraps a ParticleData instance internally while maintaining
    the existing API for backward compatibility.
    """
    
    def __init__(
        self,
        strategy: DistributionStrategy,
        activity: ActivityStrategy,
        surface: SurfaceStrategy,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
        concentration: NDArray[np.float64],
        charge: NDArray[np.float64],
        volume: float = 1,
    ):
        warnings.warn(
            "ParticleRepresentation is deprecated. Use ParticleData instead. "
            "See migration guide: https://particula.readthedocs.io/migration",
            DeprecationWarning,
            stacklevel=2,
        )
        
        # Store strategies (behavior)
        self.strategy = strategy
        self.activity = activity
        self.surface = surface
        
        # Create internal data container
        # Convert distribution to radii/masses based on strategy
        radii, masses = self._convert_distribution(
            distribution, density, strategy
        )
        self._data = ParticleData(
            radii=radii,
            masses=masses,
            concentration=concentration,
            density=density,
            charge=charge,
            volume=volume,
        )
    
    # Legacy property access (delegate to _data)
    @property
    def distribution(self) -> NDArray[np.float64]:
        """DEPRECATED: Access underlying data via get_distribution()."""
        return self._compute_distribution_from_data()
    
    @distribution.setter
    def distribution(self, value: NDArray[np.float64]) -> None:
        """DEPRECATED: Mutation is discouraged. Use functional updates."""
        warnings.warn(
            "Direct mutation of distribution is deprecated. "
            "Use ParticleData with functional updates instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Update internal data (creates new instance)
        radii, masses = self._convert_distribution(
            value, self._data.density, self.strategy
        )
        self._data = ParticleData(
            radii=radii,
            masses=masses,
            concentration=self._data.concentration,
            density=self._data.density,
            charge=self._data.charge,
            volume=self._data.volume,
        )
    
    # Method delegation
    def get_radius(self, clone: bool = False) -> NDArray[np.float64]:
        """Return particle radii (delegates to ParticleData)."""
        if clone:
            return np.copy(self._data.radii)
        return self._data.radii
    
    def get_mass(self, clone: bool = False) -> NDArray[np.float64]:
        """Return particle masses (delegates to ParticleData)."""
        total_mass = np.sum(self._data.masses, axis=-1)
        if clone:
            return np.copy(total_mass)
        return total_mass
    
    def add_mass(self, added_mass: NDArray[np.float64]) -> None:
        """Add mass to particles (functional update internally)."""
        from particula.particles.particle_operations import add_mass
        self._data = add_mass(self._data, added_mass)
    
    # New API access
    @property
    def data(self) -> ParticleData:
        """Access underlying ParticleData (preferred for new code)."""
        return self._data
    
    @classmethod
    def from_data(
        cls,
        data: ParticleData,
        strategy: DistributionStrategy,
        activity: ActivityStrategy,
        surface: SurfaceStrategy,
    ) -> "ParticleRepresentation":
        """Create from ParticleData without deprecation warning."""
        instance = object.__new__(cls)
        instance._data = data
        instance.strategy = strategy
        instance.activity = activity
        instance.surface = surface
        return instance
```

### Dual-Type Support in Dynamics

```python
from typing import Union, overload
from particula.particles import ParticleRepresentation, ParticleData

class CondensationIsothermal(CondensationStrategy):
    """Isothermal condensation supporting both data types."""
    
    @overload
    def step(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> tuple[ParticleRepresentation, GasSpecies]: ...
    
    @overload
    def step(
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> tuple[ParticleData, GasData]: ...
    
    def step(
        self,
        particle: Union[ParticleRepresentation, ParticleData],
        gas_species: Union[GasSpecies, GasData],
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> tuple:
        """Execute condensation step.
        
        Accepts both legacy types (ParticleRepresentation, GasSpecies) and
        new types (ParticleData, GasData). Returns same types as input.
        """
        # Convert to new types internally
        if isinstance(particle, ParticleRepresentation):
            particle_data = particle.data
            return_legacy = True
        else:
            particle_data = particle
            return_legacy = False
        
        if isinstance(gas_species, GasSpecies):
            gas_data = from_species(gas_species)
        else:
            gas_data = gas_species
        
        # Perform computation on new data types
        new_particle_data, new_gas_data = self._step_impl(
            particle_data, gas_data, temperature, pressure, time_step
        )
        
        # Convert back if needed
        if return_legacy:
            # Update facade's internal data
            particle._data = new_particle_data
            # Update gas species
            gas_species.concentration = new_gas_data.concentration
            return particle, gas_species
        else:
            return new_particle_data, new_gas_data
```

### Migration Guide Structure

````markdown
# Migration Guide: ParticleRepresentation to ParticleData

## Overview

In particula v0.3.0, we introduced `ParticleData` and `GasData` as pure
data containers, separating data from behavior. The legacy classes
`ParticleRepresentation` and `GasSpecies` are now facades that wrap these
new containers.

## Why Migrate?

1. **Extensibility**: Add custom fields by subclassing ParticleData
2. **GPU Acceleration**: ParticleData works with NVIDIA Warp backend
3. **Immutability**: Frozen dataclasses prevent accidental mutations
4. **Cleaner Code**: Separate data from behavior strategies

## Quick Migration

### Before (Legacy)
```python
from particula.particles import ParticleRepresentation

rep = ParticleRepresentation(
    strategy=strategy,
    activity=activity,
    surface=surface,
    distribution=radii,
    density=density,
    concentration=concentration,
    charge=charge,
)
rep.add_mass(added_mass)  # Mutates in place
````

### After (New)
```python
from particula.particles import ParticleData, add_mass

data = ParticleData(
    radii=radii,
    masses=compute_masses(radii, density),
    concentration=concentration,
    density=density,
    charge=charge,
)
data = add_mass(data, added_mass)  # Returns new instance
```

## Gradual Migration

You can migrate gradually by accessing the underlying data:

```python
# Existing code still works (with deprecation warning)
rep = ParticleRepresentation(...)

# Access new data container
data = rep.data  # ParticleData instance

# Use in new code
from particula.particles import add_mass
new_data = add_mass(data, added_mass)
```
```

## Phase Checklist

- [x] **E3-F4-P1**: Create `ParticleRepresentation` facade
  - Issue: #1068 | Size: L | Status: Shipped
  - Refactor `particula/particles/representation.py`
  - Wrap `ParticleData` internally
  - Delegate all methods to data container
  - Add deprecation warnings to constructor and mutation methods
  - Ensure all existing tests still pass
  - Write `particula/particles/tests/representation_facade_test.py`
  - Tests for facade behavior, deprecation warnings

- [x] **E3-F4-P2**: Create `GasSpecies` facade
  - Issue: #1069 | Size: M | Status: Shipped
  - Refactor `particula/gas/species.py`
  - Wrap `GasData` internally
  - Delegate all methods to data container
  - Add deprecation warnings
  - Ensure all existing tests still pass
  - Write `particula/gas/tests/species_facade_test.py`
  - Tests for facade behavior, deprecation warnings

- [x] **E3-F4-P3**: Update `CondensationIsothermal` for dual-type support
  - Issue: #1070 | Size: M | Status: Shipped
  - Update `particula/dynamics/condensation/condensation_strategies.py`
  - Add overloaded `step()` signatures
  - Implement type detection and conversion
  - Ensure existing tests pass
  - Add tests for new data type inputs
  - Write tests in `condensation_strategies_test.py`

- [x] **E3-F4-P4**: Update `Coagulation` for dual-type support
  - Issue: #1071 | Size: M | Status: Shipped
  - Update `particula/dynamics/coagulation/` strategies
  - Add overloaded signatures where appropriate
  - Implement type detection and conversion
  - Ensure existing tests pass
  - Add tests for new data type inputs
  - Write tests in coagulation test files

- [x] **E3-F4-P5**: Documentation and migration guide
  - Issue: #1072 | Size: M | Status: Shipped
  - Create `docs/Features/particle-data-migration.md`
  - Document new APIs in docstrings
  - Add examples showing both old and new patterns
  - Update `adw-docs/` with architecture changes
  - Update README with migration notes

## Testing Strategy

### Backward Compatibility Tests

All existing tests in these locations must continue passing:
- `particula/particles/tests/representation_test.py`
- `particula/gas/tests/species_test.py`
- `particula/dynamics/condensation/tests/`
- `particula/dynamics/coagulation/tests/`

### New Tests

| Test File | Coverage Target |
|-----------|----------------|
| `representation_facade_test.py` | Facade delegation, deprecation warnings |
| `species_facade_test.py` | Facade delegation, deprecation warnings |
| `*_strategies_test.py` | Dual-type support in dynamics |

### Integration Tests

- Verify full simulation workflow works with both old and new types
- Verify mixing old and new types in same simulation

## Dependencies

- E3-F1 (ParticleData) must be complete
- E3-F2 (GasData) must be complete

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing code | Extensive backward compatibility testing |
| Deprecation noise | Clear migration guide, silence option |
| Performance regression | Profile critical paths |
| Incomplete coverage | Track all public API usage |

## Success Criteria

1. All existing tests pass without modification
2. Deprecation warnings appear for legacy usage
3. New data types work in dynamics modules
4. Migration guide is complete and accurate
5. No performance regression in common workflows

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-19 | Initial feature document | ADW |
| 2026-02-28 | Marked phases shipped and updated status | ADW |
