# Particula Documentation Index

## Quick Start for Claude Code

### For First-Time Onboarding
1. Read **ARCHITECTURE_SUMMARY.txt** (3-5 minutes)
   - Overview of all components
   - Main design patterns
   - Extension points

2. Read **CLAUDE.md** relevant sections (10-15 minutes)
   - Section 1-3 for high-level architecture
   - Section specific to your task area

3. Review example code:
   - `/particula/integration_tests/quick_start_test.py` - Complete workflow
   - Relevant test files in `/particula/[module]/tests/` directory

---

## CLAUDE.md Sections

### Introduction
- Project Overview
- High-Level Architecture
- Core Concepts
- Main Package Structure

### Architecture Patterns (with code examples)
1. **Builder Pattern (Fluent Interface)**
   - Key files: abc_builder.py, builder_mixin.py
   - Usage: `.set_parameter(value, units).build()`

2. **Strategy Pattern**
   - Distribution, Activity, Surface, Coagulation, Condensation, Vapor Pressure strategies
   - All follow ABC (Abstract Base Class) pattern

3. **Factory Pattern**
   - Configuration-driven instantiation
   - Located in multiple `*_factories.py` files

4. **Runnable/Process Chain Pattern**
   - Chainable processes using `|` operator
   - Located in dynamics/ module

### Core Classes Reference
1. **Aerosol** - Central container (97 lines)
2. **Atmosphere** - Gas phase dataclass (73 lines)
3. **ParticleRepresentation** - Particle state (553 lines)
4. **Process Classes** - MassCondensation, Coagulation (222 lines)

### Testing
- Test organization and location
- Testing patterns and conventions
- How to run tests and linting

### Configuration Files
- pyproject.toml: Project metadata, dependencies, ruff config
- mkdocs.yml: Documentation setup
- GitHub workflows: CI/CD pipeline

### Key Utilities
- Unit Conversion System
- Constants System (scipy-based)
- Input Validation (decorator-based)

### Development Workflow
- Code style (ruff, 80 chars, Google docstrings)
- Testing framework (pytest)
- Documentation (MkDocs)
- Git workflow (main branch stable)

### Important Patterns and Conventions
1. Method chaining (all builders)
2. Unit-aware builders (automatic conversion)
3. Strategy-based extensibility
4. Numpy array vectorization
5. Logger integration
6. Clone pattern (deep copy on demand)

### Entry Points and Examples
- Basic usage flow
- Key integration test
- Complete example in quick_start_test.py

### Performance Considerations
- Vectorization with numpy
- Strategy pattern efficiency
- Optional dependencies
- Sub-stepping support

### Future Extension Points
- New Distribution Types
- New Activity Models
- New Coagulation Kernels
- New Processes
- New Vapor Pressure Models

---

## Directory Structure Reference

```
/home/kyle/Code/particula/
├── CLAUDE.md                          # Full architecture guide
├── ARCHITECTURE_SUMMARY.txt           # Quick reference
├── DOCUMENTATION_INDEX.md             # This file
│
├── particula/
│   ├── __init__.py                   # Main entry, version 0.2.6
│   ├── aerosol.py                    # Core Aerosol class
│   ├── aerosol_builder.py            # Aerosol builder
│   ├── runnable.py                   # Process base classes
│   ├── abc_builder.py                # Builder base
│   ├── abc_factory.py                # Factory base
│   ├── builder_mixin.py              # 20+ setter mixins (769 lines!)
│   │
│   ├── gas/                          # Gas phase modeling
│   ├── particles/                    # Particle phase modeling
│   ├── dynamics/                     # Processes (condensation, coagulation)
│   ├── activity/                     # Activity coefficient models
│   ├── equilibria/                   # Gas-particle partitioning
│   ├── util/                         # Constants, unit conversion, validation
│   │
│   ├── integration_tests/
│   │   ├── quick_start_test.py       # *** START HERE FOR USAGE EXAMPLE
│   │   ├── coagulation_integration_test.py
│   │   └── condensation_particle_resolved_test.py
│   │
│   └── [module]/tests/               # Unit tests everywhere (*_test.py)
│
├── pyproject.toml                     # Build config, ruff settings
├── mkdocs.yml                         # Documentation config
├── .github/workflows/                 # CI/CD pipelines
│   ├── test.yml                      # pytest
│   ├── lint.yml                      # ruff
│   ├── mkdocs.yml                    # Build docs
│   ├── pypi.yml                      # Publish to PyPI
│   ├── AIdocs.yml                    # AI doc generation
│   └── stale.yml                     # Stale issue management
│
└── docs/                             # MkDocs documentation
```

---

## Module Purpose Guide

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `gas/` | Atmosphere/vapor pressure/gas properties | atmosphere.py, species.py, properties/ |
| `particles/` | Particle distribution/activity/surface | representation.py, *_strategies.py, properties/ |
| `dynamics/` | Processes for particle evolution | particle_process.py, condensation/, coagulation/ |
| `activity/` | Activity coefficient models | activity_coefficients.py, gibbs.py, bat_*.py |
| `equilibria/` | Gas-particle partitioning | partitioning.py |
| `util/` | Constants, unit conversion, validation | constants.py, convert_units.py, validate_inputs.py |

---

## Common Tasks and Where to Look

### Task: Add a New Condensation Model
1. Read: `CLAUDE.md` → "Runnable/Process Chain Pattern"
2. Create: New CondensationStrategy subclass in `dynamics/condensation/`
3. Register: In `dynamics/condensation/condensation_factories.py`
4. Test: Create `*_test.py` alongside implementation

### Task: Add a New Coagulation Kernel
1. Read: `CLAUDE.md` → "Strategy Pattern"
2. Create: New CoagulationStrategy subclass in `dynamics/coagulation/coagulation_strategy/`
3. Kernel: In `dynamics/coagulation/` as `*_kernel.py`
4. Register: In `dynamics/coagulation/coagulation_factories.py`
5. Test: In `dynamics/coagulation/tests/`

### Task: Add a New Particle Distribution Strategy
1. Read: `CLAUDE.md` → "Strategy Pattern" section
2. Create: Implement DistributionStrategy in `particles/distribution_strategies/`
3. Register: In `particles/distribution_factories.py`
4. Test: In `particles/distribution_strategies/tests/`

### Task: Modify Builder Interface
1. Location: `particula/builder_mixin.py` - likely add/modify mixin class
2. Or: Specific builder like `gas/atmosphere_builders.py`
3. Pattern: Follow fluent interface (return self)
4. Units: Support auto-conversion in setter

### Task: Understand Test Patterns
1. Start: `particula/integration_tests/quick_start_test.py` - complete example
2. Examples: `particula/activity/tests/activity_coefficients_test.py`
3. Run: `pytest particula/` from repo root

---

## Important Conventions

1. **All paths in CLAUDE.md are absolute**: `/home/kyle/Code/particula/...`
2. **Unit-aware design**: Accept ANY unit in builders, convert to SI internally
3. **Strategy extensibility**: Add new physics without modifying core classes
4. **Method chaining**: All builder methods return `self`
5. **Numpy arrays**: Most getters accept `clone: bool` parameter
6. **Logger**: Use `logging.getLogger("particula")` for unified logging
7. **Docstrings**: Google style (enforced by ruff)
8. **Test naming**: `function_name_test.py` pattern
9. **Process composition**: Use `|` operator to chain: `process1 | process2 | process3`

---

## Useful Commands

```bash
# Run all tests
pytest /home/kyle/Code/particula/

# Run specific test file
pytest /home/kyle/Code/particula/particula/activity/tests/activity_coefficients_test.py

# Check linting
ruff check /home/kyle/Code/particula/particula/

# Fix linting issues
ruff check --fix /home/kyle/Code/particula/particula/

# Build documentation
mkdocs build

# Quick syntax check
python -m py_compile /home/kyle/Code/particula/particula/aerosol.py
```

---

## Quick Reference: Key Classes

### Aerosol
- File: `/home/kyle/Code/particula/particula/aerosol.py`
- Central container for atmosphere + particles
- Methods: `replace_atmosphere()`, `replace_particles()`

### ParticleRepresentation
- File: `/home/kyle/Code/particula/particula/particles/representation.py`
- Orchestrates particle state with strategies
- 30+ getter/setter methods

### Builder Classes
- Base: `/home/kyle/Code/particula/particula/abc_builder.py`
- Mixins: `/home/kyle/Code/particula/particula/builder_mixin.py`
- All return self for chaining

### Process Classes
- Base: `/home/kyle/Code/particula/particula/runnable.py`
- Implementations: MassCondensation, Coagulation in `/dynamics/particle_process.py`
- Chain with `|` operator

---

## Contact/Documentation

- **Project**: https://github.com/uncscode/particula
- **Docs**: https://uncscode.github.io/particula
- **PyPI**: https://pypi.org/project/particula/
- **Citation**: See `/home/kyle/Code/particula/citation` file

---

*Last Updated: 2025-10-18*
*For questions, review CLAUDE.md or check related test files*
