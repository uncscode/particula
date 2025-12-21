# Architecture Guide

**Project:** particula
**Version:** 0.2.6
**Last Updated:** 2025-12-02

## Overview

This document provides comprehensive architectural guidance for the **particula** project—a simple, fast, and powerful particle simulator for aerosol science. It defines the principles, patterns, and practices that guide system design and implementation.

## Architectural Principles

### 1. Scientific Computing Focus

Particula is designed for high-performance scientific computing using NumPy vectorization.

**Rationale:** Aerosol simulations involve large arrays of particles and gas species. Vectorized operations provide orders of magnitude performance improvements over scalar operations.

**Examples:**
```python
# Good: Vectorized operation on all particles
densities = mass / volume  # NumPy arrays

# Avoid: Scalar loop over particles
for i in range(len(mass)):
    densities[i] = mass[i] / volume[i]
```

### 2. Separation of Concerns

Clear boundaries between gas phase, particle phase, and dynamics modules.

**Rationale:** Aerosol science naturally divides into distinct domains (gas properties, particle properties, dynamic processes). This separation makes the codebase easier to understand, test, and extend.

**Examples:**
```python
# Gas module: Handles vapor pressure, species, atmosphere
from particula.gas import Atmosphere, GasSpecies

# Particles module: Handles distributions, representations
from particula.particles import ParticleRepresentation

# Dynamics module: Handles coagulation, condensation, wall loss
from particula.dynamics import MassCondensation, Coagulation
```

### 3. Builder Pattern for Complex Objects

Use builders to construct complex objects with validation.

**Rationale:** Scientific objects like `Aerosol`, `ParticleRepresentation`, and `Atmosphere` have many parameters and interdependencies. Builders provide a fluent, validated way to construct them.

**Examples:**
```python
from particula.aerosol_builder import AerosolBuilder

aerosol = (
    AerosolBuilder()
    .set_atmosphere(atmosphere)
    .set_particles(particles)
    .build()  # Validates consistency before building
)
```

### 4. Strategy Pattern for Algorithms

Use strategy pattern to allow different calculation methods for the same physical process.

**Rationale:** Aerosol science has multiple valid approaches for many calculations (e.g., different coagulation kernels, vapor pressure models). The strategy pattern allows users to choose the most appropriate method without changing the interface.

**Examples:**
```python
# Different vapor pressure strategies
from particula.gas import (
    AntoineVaporPressureStrategy,
    ClausiusClapeyronStrategy,
    WaterBuckStrategy
)

# Different coagulation strategies
from particula.dynamics.coagulation import (
    BrownianCoagulationStrategy,
    ChargedCoagulationStrategy,
    TurbulentDNSCoagulationStrategy
)
```

## System Architecture

### High-Level Architecture

Particula follows a layered architecture where physical processes operate on aerosol state:

```
┌─────────────────────────────────────────────┐
│          User Interface / Examples          │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│         Aerosol (Central State)             │
│  ┌────────────┐        ┌─────────────────┐  │
│  │ Atmosphere │        │ Particles       │  │
│  │ (Gas)      │        │ (Representation)│  │
│  └────────────┘        └─────────────────┘  │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│          Dynamics Processes                 │
│  ┌─────────────┐  ┌──────────────┐          │
│  │ Condensation│  │ Coagulation  │          │
│  └─────────────┘  └──────────────┘          │
│  ┌─────────────┐  ┌──────────────┐          │
│  │ Wall Loss   │  │ Dilution     │          │
│  └─────────────┘  └──────────────┘          │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│    Core Utilities & Properties              │
│  - Constants      - Validation              │
│  - Unit Conversion - Properties Calculation │
└─────────────────────────────────────────────┘
```

### Module Organization

The project is organized into domain-specific modules:

```
particula/
├── gas/                    # Gas phase: species, vapor pressure, atmosphere
│   ├── properties/         # Gas property calculations
│   └── tests/              # Gas module tests
├── particles/              # Particle phase: distributions, representations
│   ├── distribution_strategies/
│   ├── properties/         # Particle property calculations
│   └── tests/              # Particle module tests
├── dynamics/               # Dynamic processes on aerosols
│   ├── coagulation/        # Particle coagulation algorithms
│   ├── condensation/       # Vapor condensation algorithms
│   └── properties/         # Dynamics property calculations
├── activity/               # Activity coefficients, phase separation
│   └── tests/              # Activity module tests
├── equilibria/             # Equilibrium partitioning
│   └── tests/              # Equilibria module tests
├── util/                   # Shared utilities
│   ├── chemical/           # Chemical property data
│   └── lf2013_coagulation/ # Specialized coagulation tables
├── aerosol.py              # Central Aerosol state class
├── aerosol_builder.py      # Builder for Aerosol
└── runnable.py             # Process chaining framework
```

#### gas/

**Purpose:** Represents the gas phase including vapor pressure, gas species, and atmospheric conditions.

**Key Components:**
- `GasSpecies`: Individual gas species with properties
- `Atmosphere`: Collection of gas species with state (T, P)
- `VaporPressureStrategy`: Strategy pattern for vapor pressure calculations
- `GasSpeciesBuilder`, `AtmosphereBuilder`: Builders for complex objects

**Dependencies:** util (constants, validation), NumPy, SciPy

#### particles/

**Purpose:** Represents particle distributions and properties.

**Key Components:**
- `ParticleRepresentation`: Main particle state representation
- Distribution strategies: `MassBasedMovingBin`, `RadiiBasedMovingBin`, `ParticleResolvedSpeciatedMass`
- `ActivityStrategy`: Calculates water activity using different models
- `SurfaceStrategy`: Calculates surface properties
- Builders and factories for all strategies

**Dependencies:** util (constants, validation), NumPy

#### dynamics/

**Purpose:** Implements dynamic processes that modify aerosol state over time.

**Key Components:**
- `MassCondensation`: Vapor condensation onto particles
- `Coagulation`: Particle-particle collisions
- `Coagulation strategies`: Brownian, charged, turbulent, sedimentation
- `wall_loss` package: Chamber wall loss coefficients, strategy-based wall loss models (`WallLossStrategy`, `SphericalWallLossStrategy`, `RectangularWallLossStrategy`), builders (`SphericalWallLossBuilder`, `RectangularWallLossBuilder`), and `WallLossFactory` exported via `particula.dynamics.wall_loss` and `particula.dynamics`
- `dilution`: Chamber dilution process

**Dependencies:** gas, particles, util, NumPy, SciPy

#### activity/

**Purpose:** Calculates activity coefficients and phase separation.

**Key Components:**
- `activity_coefficients.py`: BAT, UNIFAC-style activity models
- `phase_separation.py`: Liquid-liquid phase separation
- `water_activity.py`: Water activity calculations

**Dependencies:** util, NumPy

#### util/

**Purpose:** Shared utilities, constants, and validation.

**Key Components:**
- `constants.py`: Physical constants (R, kB, Na, etc.)
- `validate_inputs.py`: Decorator for input validation
- `convert_units.py`: Unit conversion functions
- `chemical/`: Chemical property databases

**Dependencies:** NumPy

## Design Patterns

### Builder Pattern

**When to Use:** Creating complex objects with many parameters and validation requirements.

**Implementation:**
```python
from particula.abc_builder import BuilderABC

class MyBuilder(BuilderABC):
    def __init__(self):
        required_parameters = ["param1", "param2"]
        super().__init__(required_parameters)
        self.param1 = None
        self.param2 = None
    
    def set_param1(self, value, units=None):
        self.param1 = value
        return self  # Enable chaining
    
    def set_param2(self, value, units=None):
        self.param2 = value
        return self
    
    def build(self):
        self.pre_build_check()  # Validates all required params set
        return MyObject(self.param1, self.param2)
```

**Examples in Codebase:**
- `AerosolBuilder` (builds `Aerosol`)
- `AtmosphereBuilder` (builds `Atmosphere`)
- `ParticleMassRepresentationBuilder` (builds `ParticleRepresentation`)
- `BrownianCoagulationBuilder` (builds `BrownianCoagulationStrategy`)
- `SphericalWallLossBuilder`, `RectangularWallLossBuilder` (build wall loss strategies using shared mixins for geometry and distribution)

### Strategy Pattern

**When to Use:** Multiple algorithms for the same interface (e.g., different vapor pressure models).

**Implementation:**
```python
from abc import ABC, abstractmethod

class VaporPressureStrategy(ABC):
    @abstractmethod
    def get_saturation_concentration(self, temperature):
        pass

class AntoineVaporPressureStrategy(VaporPressureStrategy):
    def get_saturation_concentration(self, temperature):
        # Antoine equation implementation
        pass

class ClausiusClapeyronStrategy(VaporPressureStrategy):
    def get_saturation_concentration(self, temperature):
        # Clausius-Clapeyron equation implementation
        pass
```

**Examples in Codebase:**
- Vapor pressure strategies: `AntoineVaporPressureStrategy`, `ClausiusClapeyronStrategy`, `WaterBuckStrategy`
- Coagulation strategies: `BrownianCoagulationStrategy`, `ChargedCoagulationStrategy`, `TurbulentDNSCoagulationStrategy`
- Activity strategies: `ActivityIdealMass`, `ActivityIdealMolar`, `ActivityKappaParameter`
- Surface strategies: `SurfaceStrategyVolume`, `SurfaceStrategyMass`, `SurfaceStrategyMolar`
- Wall loss strategies: `WallLossStrategy`, `SphericalWallLossStrategy`

### Factory Pattern

**When to Use:** Creating objects based on configuration or string identifiers.

**Implementation:**
```python
class VaporPressureFactory:
    @staticmethod
    def get_strategy(method: str, **kwargs):
        if method == "antoine":
            return AntoineVaporPressureStrategy(**kwargs)
        elif method == "clausius_clapeyron":
            return ClausiusClapeyronStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
```

**Examples in Codebase:**
- `VaporPressureFactory`
- `GasSpeciesFactory`
- `DistributionFactory`
- `ActivityFactory`
- `CoagulationFactory`
- `WallLossFactory`

### Runnable Pattern

**When to Use:** Chaining multiple processes that modify aerosol state.

**Implementation:**
```python
from particula.runnable import RunnableABC

class MyProcess(RunnableABC):
    def rate(self, aerosol):
        # Calculate process rate
        return calculated_rate
    
    def execute(self, aerosol, time_step, sub_steps=1):
        # Modify aerosol in-place
        # Return updated aerosol
        return aerosol

# Chain processes using | operator
process_sequence = condensation | coagulation | wall_loss
final_aerosol = process_sequence.execute(initial_aerosol, time_step=1.0)
```

**Examples in Codebase:**
- `MassCondensation` (implements `RunnableABC`)
- `Coagulation` (implements `RunnableABC`)
- `RunnableSequence` (chains multiple `Runnable` objects)

## Anti-Patterns

### Things to Avoid

1. **Scalar Loops Over Arrays**
   - Why: Orders of magnitude slower than vectorized operations
   - Instead: Use NumPy broadcasting and vectorization
   ```python
   # Bad
   for i in range(len(radius)):
       volume[i] = (4/3) * np.pi * radius[i]**3
   
   # Good
   volume = (4/3) * np.pi * radius**3
   ```

2. **Mutable Default Arguments**
   - Why: Python gotcha—shared mutable state across calls
   - Instead: Use `None` and initialize inside function
   ```python
   # Bad
   def process(data, cache={}):
       pass
   
   # Good
   def process(data, cache=None):
       if cache is None:
           cache = {}
   ```

3. **Hardcoded Physical Constants**
   - Why: Inconsistent values, hard to maintain
   - Instead: Use `particula.util.constants`
   ```python
   # Bad
   energy = 1.38e-23 * temperature
   
   # Good
   from particula.util.constants import BOLTZMANN_CONSTANT
   energy = BOLTZMANN_CONSTANT * temperature
   ```

4. **Missing Input Validation**
   - Why: Difficult to debug when bad inputs propagate
   - Instead: Use `@validate_inputs` decorator
   ```python
   from particula.util.validate_inputs import validate_inputs
   
   @validate_inputs({
       "mass": "positive",
       "volume": "positive"
   })
   def calculate_density(mass, volume):
       return mass / volume
   ```

## Data Flow

### Typical Simulation Workflow

```
1. Setup Phase
   ├─ Define gas species → GasSpecies objects
   ├─ Create atmosphere → Atmosphere object
   ├─ Define particle distribution → ParticleRepresentation
   └─ Build aerosol → Aerosol(atmosphere, particles)

2. Process Definition
   ├─ Create condensation process → MassCondensation
   ├─ Create coagulation process → Coagulation
   └─ Chain processes → process_chain

3. Time Evolution
   └─ For each time step:
       ├─ Calculate rates → process.rate(aerosol)
       ├─ Execute process → process.execute(aerosol, dt)
       └─ Update aerosol state → modified aerosol returned
```

### Aerosol State Updates

When a dynamic process executes:

```
Input: Aerosol (atmosphere + particles) + time_step
   │
   ├─> Calculate Process Rate
   │   └─> Uses particle and gas properties
   │
   ├─> Apply Process Over Time
   │   ├─> May use sub-stepping for stability
   │   └─> Modifies particle distribution
   │
   └─> Return Updated Aerosol
       └─> New state (gas concentrations, particle masses, etc.)
```

## Error Handling Strategy

Particula uses exceptions for error handling with input validation at function boundaries.

**Exception Hierarchy:**
```python
Exception
├── ValueError  # Invalid parameter values
├── TypeError   # Wrong type passed
└── KeyError    # Missing required configuration
```

**Validation Pattern:**
```python
from particula.util.validate_inputs import validate_inputs

@validate_inputs({
    "mass": "positive",          # Must be > 0
    "radius": "nonnegative",     # Must be >= 0
    "temperature": "finite"      # Must be finite (not inf/nan)
})
def my_function(mass, radius, temperature):
    # Inputs are guaranteed valid here
    pass
```

## Testing Architecture

### Unit Tests

- **Location**: Each module has a `tests/` directory
- **Naming**: `*_test.py` (e.g., `activity_coefficients_test.py`)
- **Coverage Target**: >80%
- **Framework**: pytest

### Integration Tests

- **Location**: `particula/integration_tests/`
- **Naming**: `*_integration_test.py`
- **Scope**: Test interactions between modules (e.g., condensation + coagulation)

### Test Organization

```
particula/
├── activity/
│   ├── tests/
│   │   ├── activity_coefficients_test.py
│   │   └── phase_separation_test.py
│   └── activity_coefficients.py
├── gas/
│   ├── tests/
│   │   ├── species_test.py
│   │   └── atmosphere_test.py
│   └── species.py
└── integration_tests/
    ├── condensation_particle_resolved_test.py
    └── coagulation_integration_test.py
```

## Performance Considerations

### NumPy Vectorization

Particula relies heavily on NumPy vectorization for performance.

**Guidelines:**
- Always prefer array operations over loops
- Use NumPy broadcasting for element-wise operations
- Avoid repeatedly creating small arrays in loops
- Pre-allocate arrays when size is known

**Example:**
```python
# Efficient: Single vectorized operation
particle_volumes = (4/3) * np.pi * radii**3

# Inefficient: Loop with repeated operations
particle_volumes = np.zeros_like(radii)
for i in range(len(radii)):
    particle_volumes[i] = (4/3) * np.pi * radii[i]**3
```

### Memory Management

- Use appropriate NumPy dtypes (`float64` for scientific computing)
- Avoid unnecessary array copies (use views when possible)
- Be mindful of memory for large particle distributions

## Security Considerations

### Input Validation

All public functions validate inputs using the `@validate_inputs` decorator.

**Guidelines:**
- Validate all user-provided numerical inputs
- Check for finite values (no inf/nan)
- Verify array shapes match expected dimensions
- Validate physical reasonableness (e.g., positive mass)

### Safe Dependencies

- Pin dependency versions in `pyproject.toml`
- Rely on well-established scientific Python packages (NumPy, SciPy)
- Minimize external dependencies

## Deployment Architecture

Particula is deployed as a Python package via PyPI.

```
Development → Testing → PyPI Release → User Installation
     │           │            │              │
     ├─ ruff     ├─ pytest    ├─ GitHub      ├─ pip install
     ├─ mypy     └─ coverage  │   Actions    └─ conda install
     └─ pytest                └─ twine
```

**CI/CD Pipeline:**
1. Lint checks (ruff, mypy)
2. Unit tests (pytest)
3. Integration tests
4. Build package
5. Publish to PyPI (on release)

## Decision Records

Major architectural decisions will be documented in [Architecture Decision Records (ADRs)](decisions/).

Current ADRs include:
- [ADR-001: Strategy-based wall loss subsystem and `wall_loss` package refactor](decisions/ADR-001-strategy-based-wall-loss-subsystem.md)

Examples of decisions that would warrant additional ADRs:
- Adopting a new design pattern
- Choosing between competing algorithm approaches
- Major refactoring of module structure
- Adding new external dependencies

See the [ADR template](decisions/template.md) for creating new decision records.

## Migration Guidelines

### Adding New Modules

1. Create module directory under `particula/`
2. Add `__init__.py` with public API exports
3. Create `tests/` subdirectory
4. Add module documentation (docstrings)
5. Update this architecture guide
6. Add integration tests if module interacts with others

### Adding New Strategies

1. Define strategy interface (ABC with abstract methods)
2. Implement concrete strategy classes
3. Create builder for strategy configuration
4. Add factory method for string-based creation
5. Write unit tests for each strategy
6. Update module `__init__.py` to export new strategies

### Deprecating Modules

1. Mark as deprecated in docstring with removal version
2. Add deprecation warning using `warnings.warn()`
3. Update documentation to recommend alternative
4. Keep deprecated code for at least 2 minor versions
5. Create ADR documenting deprecation decision

## References

- [Architecture Outline](architecture_outline.md): High-level overview
- [Code Style Guide](../code_style.md): Coding conventions
- [Testing Guide](../testing_guide.md): Testing standards
- [Review Guide](../review_guide.md): Architecture review criteria

## Contributing

When making architectural changes:

1. Review this guide and ensure alignment with principles
2. Create an ADR for significant decisions (see [template](decisions/template.md))
3. Update relevant documentation (this guide, outline, code style)
4. Ensure new patterns are consistent with existing codebase
5. Get architectural review before implementation
6. Update this guide if introducing new patterns or modules
