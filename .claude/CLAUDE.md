# Particula Codebase Architecture Guide

## Project Overview

**Particula** is a Python-based aerosol particle simulator for modeling atmospheric aerosol systems including both gas and particle phases. It's designed to be simple, fast, and extensible for scientific research.

- **Repository**: https://github.com/uncscode/particula
- **Version**: 0.2.6
- **Python**: >= 3.9
- **Core Dependencies**: NumPy (>=2.0.0), SciPy (>=1.12)
- **LOC**: ~10,000 lines of Python code
- **License**: MIT

## High-Level Architecture

Particula follows a **modular, strategy-based architecture** using design patterns (Builder, Factory, Strategy) to provide flexibility in particle and gas phase modeling.

### Core Concepts

1. **Aerosol**: The central container combining atmosphere (gas phase) and particles (particle phase)
2. **Runnable Processes**: Chainable processes (condensation, coagulation, etc.) that modify aerosols
3. **Strategies**: Pluggable implementations for distribution types, activity models, coagulation kernels, etc.
4. **Builders**: Fluent interfaces for constructing complex objects with unit conversion support

## Main Package Structure

```
particula/
├── __init__.py                # Main entry point, version, logger setup
├── aerosol.py                 # Core Aerosol class (gas + particles container)
├── aerosol_builder.py         # Builder pattern for Aerosol construction
├── runnable.py                # Abstract base for chainable processes
├── abc_builder.py             # Base class for all builders
├── abc_factory.py             # Base class for all factories
├── builder_mixin.py           # Mixins for builder unit conversion (dense, 769 lines)
│
├── gas/                        # Gas phase modeling
│   ├── atmosphere.py          # Atmosphere class (temperature, pressure, species)
│   ├── atmosphere_builders.py # Builder for Atmosphere
│   ├── species.py             # GasSpecies dataclass
│   ├── species_builders.py    # Builders for GasSpecies
│   ├── vapor_pressure_*.py    # Vapor pressure strategy implementations
│   └── properties/            # Gas property calculations (6 modules)
│       ├── mean_free_path.py
│       ├── dynamic_viscosity.py
│       ├── thermal_conductivity.py
│       ├── fluid_rms_velocity.py
│       └── ... (kolmogorov, integral scale, etc.)
│
├── particles/                  # Particle phase modeling
│   ├── representation.py       # ParticleRepresentation (central class, 553 lines)
│   ├── representation_builders.py
│   ├── distribution_strategies/ # How to represent particle distributions
│   │   ├── base.py            # DistributionStrategy ABC
│   │   ├── mass_based_moving_bin.py
│   │   ├── radii_based_moving_bin.py
│   │   ├── speciated_mass_moving_bin.py
│   │   └── particle_resolved_speciated_mass.py
│   │
│   ├── activity_strategies.py  # Activity coefficient strategies
│   ├── surface_strategies.py   # Surface tension and Kelvin effect
│   ├── properties/             # 30+ particle property modules
│   │   ├── kelvin_effect_module.py
│   │   ├── diffusion_coefficient.py
│   │   ├── knudsen_number_module.py
│   │   ├── slip_correction_module.py
│   │   ├── settling_velocity.py
│   │   ├── coulomb_enhancement.py
│   │   └── ... (many more physical calculations)
│   │
│   └── distribution_builders.py
│
├── dynamics/                   # Process implementations (Runnable)
│   ├── particle_process.py     # MassCondensation, Coagulation classes (222 lines)
│   ├── wall_loss.py            # Wall loss process
│   ├── dilution.py             # Dilution process
│   │
│   ├── condensation/           # Mass transfer and condensation
│   │   ├── condensation_strategies.py
│   │   ├── mass_transfer.py
│   │   └── condensation_builder/
│   │
│   ├── coagulation/            # Particle collision and coalescence
│   │   ├── coagulation_*.py    # Coagulation kernel implementations
│   │   ├── coagulation_strategy/ # Strategy pattern implementations
│   │   ├── particle_resolved_step/
│   │   └── turbulent_dns_kernel/
│   │
│   └── properties/
│       └── wall_loss_coefficient.py
│
├── activity/                   # Activity coefficient models
│   ├── activity_coefficients.py
│   ├── gibbs.py, gibbs_mixing.py
│   ├── bat_coefficients.py
│   └── ... (7 modules total)
│
├── equilibria/                 # Gas-particle partitioning
│   └── partitioning.py
│
├── util/                       # Utility functions and constants
│   ├── constants.py            # Physics constants (scipy-based)
│   ├── convert_units.py        # Unit conversion system
│   ├── validate_inputs.py      # Input validation decorators
│   ├── convert_dtypes.py       # Type conversion utilities
│   ├── chemical/               # Chemical property lookups
│   │   ├── chemical_properties.py
│   │   ├── chemical_search.py
│   │   └── thermo_import.py
│   └── lf2013_coagulation/     # Legacy coagulation reference
│
├── logger_setup.py             # Logging configuration
├── tests/                      # Root test directory
├── integration_tests/          # End-to-end integration tests
└── docs/                       # Documentation (MkDocs)
```

## Key Architectural Patterns

### 1. Builder Pattern (Fluent Interface)

Used extensively for creating complex objects with unit conversion and validation:

```python
# Example from abc_builder.py
class BuilderABC:
    def set_parameters(self, parameters: dict[str, Any]):
        """Set multiple parameters at once with unit handling."""
        self.check_keys(parameters)
        for key in self.required_parameters:
            unit_key = f"{key}_units"
            getattr(self, f"set_{key}")(
                parameters[key], 
                parameters.get(unit_key)
            )
        return self

# Usage in quick_start_test.py
aerosol = (
    AerosolBuilder()
    .set_atmosphere(atmosphere)
    .set_particles(particle)
    .build()
)
```

**Key Files**:
- `/home/kyle/Code/particula/particula/abc_builder.py` (168 lines)
- `/home/kyle/Code/particula/particula/builder_mixin.py` (769 lines - contains 20+ mixin classes)
- All `*_builder.py` files throughout the codebase

### 2. Strategy Pattern

Pluggable implementations for different physics models:

- **Distribution Strategies**: How particle distributions are represented (mass-based, radii-based, particle-resolved, etc.)
- **Activity Strategies**: Activity coefficient calculations (Gibbs, BAT, etc.)
- **Surface Strategies**: Surface tension and Kelvin effect models
- **Coagulation Strategies**: Brownian, charged, sedimentation, turbulent, DNS
- **Condensation Strategies**: Isothermal, etc.
- **Vapor Pressure Strategies**: Antoine, constant, lookup table

All strategies follow an ABC (Abstract Base Class) pattern:

```python
# From particles/distribution_strategies/base.py
class DistributionStrategy(ABC):
    @abstractmethod
    def get_mass(...) -> NDArray[np.float64]: ...
    @abstractmethod
    def get_radius(...) -> NDArray[np.float64]: ...
    @abstractmethod
    def add_mass(...) -> tuple: ...
```

### 3. Factory Pattern

Used to instantiate strategies from configuration dictionaries:

```python
# From abc_factory.py
class StrategyFactoryABC(ABC, Generic[BuilderT, StrategyT]):
    def get_strategy(
        self, 
        strategy_type: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> StrategyT:
        builder = self.get_builders()[strategy_type.lower()]
        if parameters:
            builder.set_parameters(parameters)
        return builder.build()
```

**Key Factory Files**:
- `gas/vapor_pressure_factories.py`
- `particles/distribution_factories.py`
- `particles/activity_factories.py`
- `dynamics/coagulation/coagulation_factories.py`
- `dynamics/condensation/condensation_factories.py`

### 4. Runnable/Process Chain Pattern

Processes are chainable and composable:

```python
# From runnable.py
class RunnableABC(ABC):
    @abstractmethod
    def rate(self, aerosol: Aerosol) -> Any: ...
    @abstractmethod
    def execute(self, aerosol: Aerosol, time_step: float, sub_steps: int = 1) -> Aerosol: ...
    
    def __or__(self, other: "RunnableABC"):
        """Chain with | operator"""
        sequence = RunnableSequence()
        sequence.add_process(self)
        sequence.add_process(other)
        return sequence

# Usage (from integration tests)
result = (process1 | process2 | process3).execute(aerosol, time_step=10.0)
```

## Core Classes and Their Roles

### 1. **Aerosol** (`aerosol.py`, 97 lines)
Central container holding:
- `atmosphere`: An Atmosphere instance (gas phase)
- `particles`: A ParticleRepresentation instance (particle phase)

Methods: `replace_atmosphere()`, `replace_particles()`

### 2. **Atmosphere** (`gas/atmosphere.py`, 73 lines)
Dataclass representing gas mixture:
- `temperature`: float (K)
- `total_pressure`: float (Pa)
- `partitioning_species`: list of GasSpecies (can partition to particles)
- `gas_only_species`: list of GasSpecies (inert)

### 3. **ParticleRepresentation** (`particles/representation.py`, 553 lines)
Central class for particle state:
- `strategy`: DistributionStrategy (how particles are represented)
- `activity`: ActivityStrategy (partial pressure calculations)
- `surface`: SurfaceStrategy (surface tension, Kelvin effect)
- `distribution`: NDArray (particle size/mass distribution)
- `density`: NDArray (particle density, kg/m³)
- `concentration`: NDArray (particle number concentration, #/m³)
- `charge`: NDArray (charge per particle, elementary charges)
- `volume`: float (air volume for particle-resolved sims, m³)

Methods: 30+ getters (get_mass, get_radius, get_concentration, etc.) and setters (add_mass, add_concentration, collide_pairs)

### 4. **Process Classes** (`dynamics/particle_process.py`, 222 lines)

#### MassCondensation (Runnable)
- Handles vapor condensation/evaporation
- Updates particle mass and gas species concentration
- Uses CondensationStrategy for mass transfer calculations

#### Coagulation (Runnable)
- Handles particle collision and coalescence
- Uses CoagulationStrategyABC for collision kernel calculations

Both implement: `rate(aerosol)` and `execute(aerosol, time_step, sub_steps)`

## Testing Structure and Patterns

### Test Organization

Tests are located alongside source code with `*_test.py` naming:
```
particula/
├── activity/tests/
├── dynamics/coagulation/tests/
├── dynamics/condensation/tests/
├── gas/tests/
├── particles/tests/
├── util/tests/
└── integration_tests/
```

**Test Count**: 200+ test files

### Testing Patterns

1. **Unit Tests**: Test individual functions/classes
   - Located in `tests/` subdirectories
   - Use pytest framework
   - Naming: `function_name_test.py`

2. **Integration Tests**: Test complete workflows
   - Located in `/particula/integration_tests/`
   - Files: `quick_start_test.py`, `coagulation_integration_test.py`, `condensation_particle_resolved_test.py`

3. **Example from** `particula/activity/tests/activity_coefficients_test.py`:
```python
def test_activity_coefficients():
    """Test for activity_coefficients function."""
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.linspace(0.1, 1, 10)
    oxygen2carbon = 0.3
    density = 2500

    activity_coefficients = bat_activity_coefficients(...)
    assert np.all(activity_coefficients[0] >= 0)  # Validation
```

### Test Execution

- **Workflow**: `.github/workflows/test.yml`
- **Framework**: pytest
- **Coverage**: Comprehensive across all modules
- **Linting**: ruff (configured in pyproject.toml)
  - Line length: 80 characters
  - Docstring convention: Google style
  - Selective ignores for test files and specific rules

## Configuration Files

### Development Configuration

**`pyproject.toml`** (86 lines):
- Project metadata (name, description, version, authors, URLs)
- Dependencies: numpy>=2.0.0, scipy>=1.12
- Optional dependencies:
  - **dev**: pylint, pytest, autopep8, jupyterlab, ruff, flake8, mkdocs-material
  - **extra**: matplotlib, pandas, pint, thermo, tqdm
- Flit build system (dynamic versioning from `particula.__version__`)
- **Ruff Configuration**:
  - Line length: 80
  - Selected rules: E, F, W, C90, D, ANN, B, S, N, I
  - Google-style docstrings
  - Per-file ignores for test files (S101, E721, B008)

**`.github/workflows/`**:
- `test.yml`: Run pytest on push/PR
- `lint.yml`: Run ruff linting
- `mkdocs.yml`: Build documentation with MkDocs Material theme
- `pypi.yml`: Publish to PyPI on release
- `AIdocs.yml`: Auto-generate docs (AI-assisted)
- `stale.yml`: Manage stale issues

### Documentation Configuration

**`mkdocs.yml`** (112 lines):
- Theme: Material with custom styling
- Plugins: search, social, mkdocs-jupyter (execute: false)
- Markdown extensions: pymdown, superfences (for Mermaid diagrams), MathJax
- Site: https://uncscode.github.io/particula

## Key Utilities and Patterns

### Unit Conversion System (`util/convert_units.py`)

Central system for handling unit conversions:
```python
get_unit_conversion(from_units: str, to_units: str, value=1.0) -> float
```

Integrated into builder mixins for transparent unit handling in all `.set_*()` methods.

### Constants System (`util/constants.py`)

All physics constants sourced from scipy.constants:
- BOLTZMANN_CONSTANT
- AVOGADRO_NUMBER
- GAS_CONSTANT
- ELEMENTARY_CHARGE_VALUE
- RELATIVE_PERMITTIVITY_AIR
- REF_VISCOSITY_AIR_STP, etc.

### Input Validation (`util/validate_inputs.py`)

Decorator-based validation:
```python
@validate_inputs({"density": "positive"})
def set_density(self, density, density_units: str): ...
```

Validation types: "positive", "nonnegative", "finite"

## Development Workflow

### Code Style and Quality

1. **Ruff**: Fast Python linter and formatter (configured in pyproject.toml)
   - Format: 80-character lines
   - Docstring convention: Google style
   - Automatically applied via pre-commit or manual run

2. **Testing**: pytest framework
   - Run: `pytest particula/`
   - Coverage tracking via CI/CD

3. **Documentation**: MkDocs with Material theme
   - Build: `mkdocs build`
   - Deploy to GitHub Pages automatically

### Git Workflow

- **Main Branch**: `main` (stable releases)
- **Feature Branches**: Feature branches for development
- **CI/CD**: GitHub Actions workflows for testing, linting, building docs, and PyPI publishing

## Important Patterns and Conventions

### 1. Method Chaining
All builder methods return `self` for fluent interface:
```python
builder.set_a(val).set_b(val).set_c(val).build()
```

### 2. Unit-Aware Builders
All setters accept optional `*_units` parameter:
```python
.set_temperature(25, "degC")  # Auto-converts to Kelvin
.set_density(1000, "g/m^3")   # Auto-converts to kg/m³
```

### 3. Strategy-Based Extensibility
New physics models are added as Strategy subclasses without modifying core classes.

### 4. Numpy Array Usage
Extensive use of numpy arrays for vectorized calculations:
- `get_radius()` → NDArray[np.float64]
- `get_mass()` → NDArray[np.float64]
- All shape operations preserved for compatibility with different distribution types

### 5. Logger Integration
```python
logger = logging.getLogger("particula")
```
All modules use the same logger for unified output.

### 6. Clone Pattern
Many getters accept `clone: bool` parameter:
```python
distribution = particle.get_distribution(clone=True)  # Deep copy
distribution = particle.get_distribution(clone=False) # Reference (default)
```

## Entry Points and Examples

### Basic Usage Flow
1. Create GasSpecies with builders
2. Build Atmosphere with species
3. Build ParticleRepresentation with distribution/activity/surface strategies
4. Create Aerosol from atmosphere + particles
5. Create process (Condensation, Coagulation, etc.)
6. Execute process: `aerosol = process.execute(aerosol, time_step=10.0)`

### Key Integration Test
`/home/kyle/Code/particula/particula/integration_tests/quick_start_test.py` - Complete end-to-end example

## Performance Considerations

1. **Vectorization**: Uses numpy for fast array operations
2. **Strategy Pattern**: Allows algorithm selection without inheritance overhead
3. **Optional Dependencies**: Extra features (visualization, units) optional
4. **Sub-stepping**: Processes support internal sub-steps for stability/accuracy trade-offs

## Future Extension Points

1. **New Distribution Types**: Implement DistributionStrategy subclass
2. **New Activity Models**: Implement ActivityStrategy subclass
3. **New Coagulation Kernels**: Implement CoagulationStrategyABC subclass
4. **New Processes**: Implement RunnableABC subclass and chain with `|` operator
5. **New Vapor Pressure Models**: Implement vapor pressure strategy
