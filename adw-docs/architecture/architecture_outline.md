# Architecture Outline

**Project:** particula
**Version:** 0.2.6
**Last Updated:** 2026-03-02

## Purpose

This document provides a high-level overview of the **particula** architecture—a simple, fast, and powerful particle simulator for aerosol science. For detailed information, see the [Architecture Guide](architecture_guide.md).

## System Overview

Particula is a scientific computing package for simulating aerosol dynamics. It provides:

- **Gas Phase Modeling**: Vapor pressure, gas species, atmospheric conditions
- **Particle Representations**: Multiple distribution strategies (mass-based, radius-based, particle-resolved)
- **Dynamic Processes**: Condensation, coagulation, wall loss, dilution
- **Activity Calculations**: Activity coefficients, phase separation, water activity
- **Equilibrium Partitioning**: Gas-particle equilibrium calculations

The system is built on **NumPy** for high-performance vectorized operations and uses design patterns (Builder, Strategy, Factory) to provide flexibility for different scientific approaches.

## Core Components

### Aerosol (Central State)

**Responsibility:** Central container holding the complete aerosol state (gas + particles)

**Location:** `particula/aerosol.py`

**Key Interfaces:**
- `Aerosol(atmosphere, particles)`: Constructor
- `replace_atmosphere(atmosphere)`: Update gas phase
- `replace_particles(particles)`: Update particle phase

**Dependencies:**
- `particula.gas.Atmosphere`
- `particula.particles.ParticleRepresentation`

---

### Atmosphere (Gas Phase)

**Responsibility:** Represents atmospheric conditions and gas species concentrations

**Location:** `particula/gas/atmosphere.py`

**Key Interfaces:**
- `Atmosphere`: Collection of gas species with temperature and pressure
- `GasSpecies`: Individual gas species with vapor pressure strategy
- `VaporPressureStrategy`: Calculate saturation concentrations

**Dependencies:**
- `particula.util.constants` (physical constants)
- NumPy for array operations

---

### ParticleRepresentation (Particle Phase)

**Responsibility:** Represents particle size distributions and chemical composition

**Location:** `particula/particles/representation.py`

**Key Interfaces:**
- `ParticleRepresentation`: Main particle state container
- Distribution strategies: `MassBasedMovingBin`, `RadiiBasedMovingBin`, `ParticleResolvedSpeciatedMass`
- `ActivityStrategy`: Water activity calculations
- `SurfaceStrategy`: Surface property calculations

**Dependencies:**
- `particula.util` (validation, constants)
- NumPy for array operations

---

### Dynamics Processes

**Responsibility:** Modify aerosol state over time (condensation, coagulation, etc.)

**Location:** `particula/dynamics/`

**Key Interfaces:**
- `MassCondensation`: Vapor condensation onto particles
- `Coagulation`: Particle-particle collisions
- `WallLossStrategy`, `SphericalWallLossStrategy`: Strategy-based wall loss for chamber geometries, provided by the `particula.dynamics.wall_loss` package
- `SphericalWallLossBuilder`, `RectangularWallLossBuilder`, `WallLossFactory`: Builder/factory support for wall loss strategies exported via `particula.dynamics.wall_loss` and `particula.dynamics`
- `RunnableABC`: Abstract base for chainable processes
- `RunnableSequence`: Chain multiple processes with `|` operator

**Dependencies:**
- `particula.aerosol.Aerosol`
- `particula.gas` and `particula.particles` modules
- NumPy, SciPy

---

### Builders

**Responsibility:** Construct complex objects with validation using fluent interface

**Location:** Multiple modules (`*_builder.py` files)

**Key Interfaces:**
- `AerosolBuilder`: Build `Aerosol` objects
- `AtmosphereBuilder`: Build `Atmosphere` objects
- `ParticleMassRepresentationBuilder`: Build `ParticleRepresentation` objects
- `BrownianCoagulationBuilder`: Build coagulation strategies
- `SphericalWallLossBuilder`, `RectangularWallLossBuilder`: Build wall loss strategies (used by `WallLossFactory`)

**Dependencies:**
- `particula.abc_builder.BuilderABC` (base class)

---

## Component Relationships

```
┌────────────────────────────────────────────────────────┐
│                    User Code                           │
│  (Examples, Scripts, Jupyter Notebooks)                │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │    Aerosol Builder            │
         └───────────────────────────────┘
                         │
         ┌───────────────┴────────────────┐
         ▼                                ▼
┌─────────────────┐          ┌────────────────────────┐
│   Atmosphere    │          │ ParticleRepresentation │
│   (Gas Phase)   │          │  (Particle Phase)      │
└─────────────────┘          └────────────────────────┘
         │                                │
         └────────────┬───────────────────┘
                      ▼
         ┌────────────────────────┐
         │      Aerosol           │
         │  (Combined State)      │
         └────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
┌──────────────┐ ┌──────────┐ ┌──────────┐
│ Condensation │ │Coagulation│ │Wall Loss │
│  Process     │ │ Process   │ │ Process  │
└──────────────┘ └──────────┘ └──────────┘
         │            │            │
         └────────────┼────────────┘
                      ▼
         ┌────────────────────────┐
         │  Updated Aerosol       │
         └────────────────────────┘
```

**Key Interactions:**

1. **Construction**: Builders create `Atmosphere` and `ParticleRepresentation`, which are combined into an `Aerosol`
2. **Dynamics**: Processes (implementing `RunnableABC`) operate on `Aerosol` state, returning updated state
3. **Chaining**: Multiple processes can be chained using `|` operator to create complex simulations

## Module Structure

```
particula/
├── gas/                    # Gas phase: vapor pressure, species, atmosphere
│   ├── properties/         # Gas property calculations (viscosity, diffusion, etc.)
│   ├── tests/              # Test coverage
│   ├── atmosphere.py       # Atmosphere class
│   ├── atmosphere_builders.py
│   ├── species.py          # GasSpecies class
│   ├── species_builders.py
│   ├── species_factories.py
│   ├── latent_heat_strategies.py    # Strategy pattern for latent heat
│   ├── vapor_pressure_strategies.py  # Strategy pattern for vapor pressure
│   ├── vapor_pressure_builders.py
│   └── vapor_pressure_factories.py
├── particles/              # Particle phase: distributions, representations
│   ├── distribution_strategies/      # Different distribution types
│   ├── properties/                   # Particle property calculations
│   ├── tests/                        # Test coverage
│   ├── representation.py             # ParticleRepresentation class
│   ├── representation_builders.py
│   ├── representation_factories.py
│   ├── activity_strategies.py        # Water activity models
│   ├── activity_builders.py
│   ├── activity_factories.py
│   ├── surface_strategies.py         # Surface property models
│   ├── surface_builders.py
│   └── surface_factories.py
├── dynamics/               # Dynamic processes that modify aerosol state
│   ├── coagulation/        # Particle coagulation (multiple kernels)
│   │   ├── coagulation_strategy/     # Strategy implementations
│   │   ├── coagulation_builder/      # Builders for strategies
│   │   ├── particle_resolved_step/   # Particle-resolved algorithms
│   │   └── turbulent_dns_kernel/     # DNS turbulence kernels
│   ├── condensation/       # Vapor condensation onto particles
│   │   ├── condensation_builder/
│   │   └── condensation_strategies.py
│   ├── properties/         # Wall loss coefficients, etc.
│   ├── tests/              # Test coverage
│   ├── particle_process.py # MassCondensation, Coagulation classes
│   ├── wall_loss/          # Chamber wall loss (package)
│   │   ├── wall_loss_strategies.py  # Strategy-based wall loss models
│   │   ├── wall_loss_builders.py    # Builders for wall loss strategies (spherical, rectangular)
│   │   ├── wall_loss_factories.py   # Factory for wall loss strategies
│   │   ├── rate.py                 # Legacy wall loss rate functions
│   │   └── tests/                  # Test coverage for wall loss strategies, builders, factories
│   └── dilution.py         # Chamber dilution
├── activity/               # Activity coefficients, phase separation
│   ├── tests/              # Test coverage
│   ├── activity_coefficients.py  # BAT, UNIFAC-style models
│   ├── phase_separation.py        # Liquid-liquid phase separation
│   └── water_activity.py          # Water activity calculations
├── equilibria/             # Gas-particle equilibrium partitioning
│   ├── tests/              # Test coverage
│   └── partitioning.py     # Partitioning calculations
├── util/                   # Shared utilities, constants, validation
│   ├── chemical/           # Chemical property databases
│   ├── lf2013_coagulation/ # Specialized coagulation tables
│   ├── tests/              # Test coverage
│   ├── constants.py        # Physical constants (R, kB, Na, etc.)
│   ├── validate_inputs.py  # Input validation decorator
│   ├── convert_units.py    # Unit conversion functions
│   └── colors.py           # Plotting utilities
├── integration_tests/      # Integration tests across modules
│   ├── coagulation_integration_test.py
│   ├── condensation_particle_resolved_test.py
│   └── quick_start_test.py
├── tests/                  # Top-level tests
│   ├── aerosol_test.py
│   ├── aerosol_builder_test.py
│   └── runnable_test.py
├── aerosol.py              # Aerosol class (central state)
├── aerosol_builder.py      # Builder for Aerosol
├── abc_builder.py          # Abstract base class for all builders
├── abc_factory.py          # Abstract base class for factories
├── runnable.py             # RunnableABC, RunnableSequence (process chaining)
└── logger_setup.py         # Logging configuration
```

## Technology Stack

### Core Technologies

- **Language:** Python (3.12+)
- **Build System:** setuptools (via pyproject.toml)
- **Package Manager:** pip, uv
- **Testing Framework:** pytest

### Key Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| numpy | >= 2.0.0 | Array operations, vectorized scientific computing |
| scipy | >= 1.12 | Scientific algorithms (interpolation, integration) |
| pytest | (dev) | Unit and integration testing |
| ruff | (dev) | Linting and code formatting |
| mypy | (dev) | Static type checking (optional) |
| mkdocs-material | (dev) | Documentation generation |

## Design Principles (Quick Reference)

1. **Scientific Computing Focus**: Use NumPy vectorization for performance
2. **Separation of Concerns**: Clear boundaries between gas, particles, and dynamics
3. **Builder Pattern**: Construct complex objects with validation
4. **Strategy Pattern**: Multiple algorithms for the same physical process
5. **Input Validation**: All public functions validate inputs using `@validate_inputs`
6. **Type Hints**: All public APIs have type hints for clarity
7. **Google-Style Docstrings**: Comprehensive documentation with examples

## Common Patterns

- **Builder Pattern**: Used for `Aerosol`, `Atmosphere`, `ParticleRepresentation`, and all strategy objects
- **Strategy Pattern**: Used for vapor pressure models, coagulation kernels, activity models, surface calculations
- **Factory Pattern**: Used for creating strategies from configuration (`VaporPressureFactory`, `CoagulationFactory`, etc.)
- **Runnable Pattern**: Used for chainable dynamic processes (`MassCondensation`, `Coagulation`, etc.)

## Data Flow (High-Level)

```
Setup Phase:
  User → Builders → Atmosphere + ParticleRepresentation → AerosolBuilder → Aerosol

Simulation Phase:
  Aerosol → Process.rate() → Calculate rates
         → Process.execute() → Update state
         → Updated Aerosol → Next time step

Output Phase:
  Aerosol.atmosphere → Gas concentrations, temperature, pressure
  Aerosol.particles → Particle masses, radii, composition
```

1. **Setup**: Use builders to create `Atmosphere` and `ParticleRepresentation`, then combine into `Aerosol`
2. **Process Definition**: Create dynamic processes (`MassCondensation`, `Coagulation`, etc.) with desired strategies
3. **Time Evolution**: Execute processes on aerosol, updating state at each time step
4. **Analysis**: Extract results from `Aerosol.atmosphere` and `Aerosol.particles` for analysis

## Extension Points

Areas designed for extension:

1. **New Vapor Pressure Models**
   - Interface: `VaporPressureStrategy` (abstract class)
   - Example: Create new strategy class, add builder, register in factory
   - Location: `particula/gas/vapor_pressure_strategies.py`

2. **New Coagulation Kernels**
   - Interface: `CoagulationStrategyABC`
   - Example: Implement new kernel calculation, create builder
   - Location: `particula/dynamics/coagulation/coagulation_strategy/`

3. **New Particle Distributions**
   - Interface: Distribution strategy (mass-based, radius-based, etc.)
   - Example: Create new distribution type with appropriate binning
   - Location: `particula/particles/distribution_strategies/`

4. **New Dynamic Processes**
   - Interface: `RunnableABC` (requires `rate()` and `execute()` methods)
   - Example: Create new process (e.g., nucleation, deposition), make it chainable
   - Location: `particula/dynamics/`

5. **New Activity Models**
   - Interface: `ActivityStrategy`
   - Example: Implement UNIFAC, AIOMFAC, or other activity coefficient models
   - Location: `particula/particles/activity_strategies.py`

## Quick Start for Developers

### Understanding the Codebase

1. Start with `aerosol.py` - The central state container
2. Review `aerosol_builder.py` - Shows the builder pattern in action
3. Explore `gas/species.py` and `particles/representation.py` - Core domain objects
4. Look at `dynamics/particle_process.py` - How processes modify state
5. Check `examples/` in `docs/Examples/` - Real-world usage patterns

### Making Changes

1. **New Features**: 
   - Determine which module (gas, particles, dynamics, activity, equilibria, util)
   - Follow existing patterns (Builder, Strategy, Factory as appropriate)
   - Add tests in module's `tests/` directory
   
2. **Bug Fixes**: 
   - Identify affected module(s) in structure above
   - Add regression test before fixing
   - Ensure all existing tests pass

3. **Refactoring**: 
   - Consult [Architecture Guide](architecture_guide.md) for patterns
   - Maintain backward compatibility when possible
   - Update documentation and tests

### Architecture Review

For changes affecting multiple modules or introducing new patterns, request an architecture review or consult the architecture guide.

## References

- **[Architecture Guide](architecture_guide.md)**: Detailed architectural documentation
- **[Decision Records](decisions/README.md)**: Architectural decision history (future ADRs)
- **[Code Style Guide](../code_style.md)**: Coding standards and conventions
- **[Testing Guide](../testing_guide.md)**: Testing conventions and pytest usage
- **[Docstring Guide](../docstring_guide.md)**: Google-style docstring format

## Updates

When updating the architecture:

1. Update this outline for high-level changes (new modules, major refactoring)
2. Update the [Architecture Guide](architecture_guide.md) for detailed pattern changes
3. Create an [ADR](decisions/template.md) for significant architectural decisions
4. Update affected module documentation (docstrings, examples)
5. Ensure CI passes (linting, tests)
