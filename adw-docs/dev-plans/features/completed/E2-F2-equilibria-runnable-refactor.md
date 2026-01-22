# Feature E2-F2: Equilibria Runnable Refactor

**Status**: Completed
**Priority**: P2
**Parent Epic**: [E2 - Activity and Equilibria Strategy-Builder-Factory Refactor](../epics/completed/E2-activity-equilibria-refactor.md)
**Start Date**: 2026-01-07
**Completion Date**: 2026-01-21
**Last Updated**: 2026-01-21
**Size**: Medium (6 phases, ~500 LOC)

## Overview

Transform the `equilibria/` module into a Runnable pattern following the
`dynamics/` architecture. This enables equilibria calculations (liquid-vapor
partitioning) to be executed as system state transformations, composable with
other runnables in simulation pipelines.

The refactor introduces:

1. **`EquilibriaStrategy`**: Abstract base class for equilibrium solving strategies
2. **`LiquidVaporPartitioningStrategy`**: Concrete strategy for organic aerosol
   partitioning using activity coefficients
3. **`Equilibria`**: Runnable wrapper that executes strategies on aerosol state

## Acceptance Criteria

- [ ] `EquilibriaStrategy` ABC defines equilibrium solving interface
- [ ] `LiquidVaporPartitioningStrategy` wraps existing partitioning functions
- [ ] `LiquidVaporPartitioningBuilder` validates required parameters
- [ ] `EquilibriaFactory` supports `"liquid_vapor"` strategy type
- [ ] `Equilibria` runnable follows `Runnable` pattern from `dynamics/`
- [ ] Existing `equilibria/` tests continue passing
- [ ] New classes have 80%+ test coverage
- [ ] Docstrings follow Google-style format

## Technical Design

### EquilibriaStrategy ABC

```python
class EquilibriaStrategy(ABC):
    """Abstract base class for equilibria solving strategies.
    
    Equilibria strategies solve for thermodynamic equilibrium states given
    system conditions. Implementations may solve liquid-vapor partitioning,
    solid-liquid equilibria, or other phase equilibria.
    """
    
    @abstractmethod
    def solve(
        self,
        aerosol: Aerosol,
        temperature: float,
        pressure: float,
    ) -> EquilibriumResult:
        """Solve for equilibrium state.
        
        Args:
            aerosol: Current aerosol state.
            temperature: System temperature [K].
            pressure: System pressure [Pa].
            
        Returns:
            EquilibriumResult containing phase concentrations and activities.
        """
        ...
    
    def get_name(self) -> str:
        """Return strategy identifier."""
        return self.__class__.__name__
```

### LiquidVaporPartitioningStrategy

```python
class LiquidVaporPartitioningStrategy(EquilibriaStrategy):
    """Liquid-vapor partitioning equilibrium strategy.
    
    Solves for equilibrium partitioning between gas and condensed phases
    using activity coefficients from the BAT model. Supports phase separation
    into alpha (water-rich) and beta (organic-rich) phases.
    
    Attributes:
        activity_strategy: Activity strategy for computing activity coefficients.
        water_activity: Target water activity (relative humidity).
    
    References:
        Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
    """
    
    def __init__(
        self,
        activity_strategy: Optional[ActivityStrategy] = None,
        water_activity: float = 0.5,
    ) -> None: ...
    
    def solve(
        self,
        aerosol: Aerosol,
        temperature: float,
        pressure: float,
    ) -> EquilibriumResult:
        """Solve liquid-vapor partitioning equilibrium."""
        ...
```

### Equilibria Runnable

```python
class Equilibria(Runnable):
    """Runnable for equilibria calculations.
    
    Executes equilibria strategies on aerosol state, updating phase
    concentrations to equilibrium values.
    
    Examples:
        >>> import particula as par
        >>> strategy = par.equilibria.LiquidVaporPartitioningStrategy(
        ...     water_activity=0.75,
        ... )
        >>> equilibria = par.equilibria.Equilibria(strategy=strategy)
        >>> aerosol = equilibria.execute(aerosol, temperature=298, pressure=101325)
    """
    
    def __init__(self, strategy: EquilibriaStrategy) -> None: ...
    
    def execute(
        self,
        aerosol: Aerosol,
        temperature: float,
        pressure: float,
    ) -> Aerosol:
        """Execute equilibria calculation on aerosol."""
        ...
```

## Phases

### Phase E2-F2-P0: Equilibria Module Code Quality Improvements

**Issue**: TBD | **Size**: M | **Status**: Not Started

Improve docstrings, code logic, and function isolation in the existing
`equilibria/partitioning.py` to bring it up to particula standards before
adding new runnable patterns.

**Tasks**:
- Refactor `liquid_vapor_obj_function()` (currently 90+ lines, 7+ parameters):
  - Extract alpha phase calculation into `_calculate_alpha_phase()` helper
  - Extract beta phase calculation into `_calculate_beta_phase()` helper
  - Extract C* calculation into `_calculate_cstar()` helper
  - Reduce function to orchestration role (~30 lines)
- Add comprehensive Google-style docstrings:
  - Document all parameters with units and physical meaning
  - Add mathematical equations in LaTeX format where appropriate
  - Add "Examples" section with minimal working example
  - Add "Raises" section for error conditions
- Add `@validate_inputs` decorator to `liquid_vapor_partitioning()`:
  - Validate `c_star_j_dry` is non-negative
  - Validate `concentration_organic_matter` is non-negative
  - Validate array shape compatibility
- Improve `get_properties_for_liquid_vapor_partitioning()`:
  - Add explicit return type annotation
  - Document return tuple structure
  - Add validation for input array lengths match
- Add missing type hints throughout module
- Add unit tests for the new helper functions
- Run linters and fix all issues

**Files Modified**:
- `particula/equilibria/partitioning.py`
- `particula/equilibria/tests/partitioning_test.py` (add tests)
- `particula/equilibria/__init__.py` (add exports)

---

### Phase E2-F2-P1: Create EquilibriaStrategy ABC and LiquidVaporPartitioningStrategy

**Issue**: TBD | **Size**: M | **Status**: Not Started

Create the equilibria strategy abstraction and liquid-vapor partitioning
implementation.

**Tasks**:
- Create `equilibria/equilibria_strategies.py` with `EquilibriaStrategy` ABC
- Create `LiquidVaporPartitioningStrategy` wrapping existing functions
- Create `EquilibriumResult` dataclass for structured results
- Add comprehensive docstrings with references
- Add unit tests in `equilibria/tests/equilibria_strategies_test.py`

**Files Modified**:
- `particula/equilibria/equilibria_strategies.py` (new)
- `particula/equilibria/tests/equilibria_strategies_test.py` (new)

---

### Phase E2-F2-P2: Create LiquidVaporPartitioningBuilder

**Issue**: TBD | **Size**: S | **Status**: Not Started

Create the builder class for `LiquidVaporPartitioningStrategy` with
parameter validation.

**Tasks**:
- Create `equilibria/equilibria_builders.py`
- Create `LiquidVaporPartitioningBuilder` with fluent interface
- Add `set_activity_strategy()` for activity strategy injection
- Add `set_water_activity()` with validation (0-1 range)
- Implement `build()` with pre-build validation
- Add unit tests

**Files Modified**:
- `particula/equilibria/equilibria_builders.py` (new)
- `particula/equilibria/tests/equilibria_builders_test.py` (new)

---

### Phase E2-F2-P3: Create EquilibriaFactory

**Issue**: TBD | **Size**: S | **Status**: Not Started

Create factory for equilibria strategy selection.

**Tasks**:
- Create `equilibria/equilibria_factories.py`
- Create `EquilibriaFactory` extending `StrategyFactoryABC`
- Add `"liquid_vapor"` mapping to builder
- Add factory docstrings with usage examples
- Add factory dispatch tests

**Files Modified**:
- `particula/equilibria/equilibria_factories.py` (new)
- `particula/equilibria/tests/equilibria_factories_test.py` (new)

---

### Phase E2-F2-P4: Create Equilibria Runnable

**Issue**: TBD | **Size**: M | **Status**: Not Started

Create the `Equilibria` runnable following the `Runnable` pattern.

**Tasks**:
- Create `equilibria/equilibria.py` with `Equilibria` class
- Inherit from `Runnable` ABC (or implement compatible interface)
- Implement `execute()` method calling strategy's `solve()`
- Support pipe operator composition (`|`) with other runnables
- Add integration tests with activity strategies
- Update `equilibria/__init__.py` with exports

**Files Modified**:
- `particula/equilibria/equilibria.py` (new)
- `particula/equilibria/__init__.py`
- `particula/equilibria/tests/equilibria_test.py` (new)

---

### Phase E2-F2-P5: Equilibria Module Cleanup and Exports

**Issue**: TBD | **Size**: S | **Status**: Not Started

Final cleanup of equilibria module, ensuring proper exports and integration
with the rest of particula.

**Tasks**:
- Update `particula/equilibria/__init__.py` with all public exports:
  - `EquilibriaStrategy`
  - `LiquidVaporPartitioningStrategy`
  - `LiquidVaporPartitioningBuilder`
  - `EquilibriaFactory`
  - `Equilibria` (runnable)
  - `EquilibriumResult` (dataclass)
  - Original `liquid_vapor_partitioning()` (backward compatibility)
  - Original `get_properties_for_liquid_vapor_partitioning()` (backward compat)
- Add `__all__` list to `__init__.py` for explicit public API
- Verify all new classes follow particula naming conventions
- Add deprecation warnings for direct function imports (recommend strategy use)
- Run full test suite including integration tests
- Verify 80%+ coverage on new code

**Files Modified**:
- `particula/equilibria/__init__.py`
- `particula/equilibria/tests/` (integration tests)

## Dependencies

- **Depends On**: E2-F1 (Activity Strategy Refactor) for activity strategies
- **Internal**: `Runnable` ABC, `StrategyFactoryABC`, `BuilderABC`
- **External**: None

## Testing Strategy

### Unit Tests
- Test `LiquidVaporPartitioningStrategy.solve()` with known inputs
- Test builder validation for all parameters
- Test factory dispatch for strategy types

### Integration Tests
- Verify strategy integrates with different activity strategies
- Verify runnable works in simulation pipelines
- Verify results match existing `partitioning.py` function outputs

### Regression Tests
- Ensure existing `partitioning_test.py` continues passing
- Verify backward compatibility with direct function calls

## Data Structures

### EquilibriumResult

```python
@dataclass
class EquilibriumResult:
    """Result of equilibrium calculation.
    
    Attributes:
        alpha_phase: Concentrations in alpha (water-rich) phase.
        beta_phase: Concentrations in beta (organic-rich) phase.
        partition_coefficients: Species partition coefficients.
        water_content: Water content in each phase.
        error: Optimization error (convergence metric).
    """
    alpha_phase: PhaseConcentrations
    beta_phase: Optional[PhaseConcentrations]
    partition_coefficients: NDArray[np.float64]
    water_content: Tuple[float, float]
    error: float
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-07 | Initial feature creation | ADW |
| 2026-01-07 | Added P0 code quality phase and P5 cleanup phase | ADW |
| 2026-01-21 | Marked feature completed; updated status, dates, and parent epic link | ADW |

