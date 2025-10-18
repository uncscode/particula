# Particula Architecture Documentation

This directory now contains comprehensive architecture documentation for future Claude Code instances working with the Particula codebase.

## Getting Started

### For Claude Code Users

Start here in this order:

1. **[ARCHITECTURE_SUMMARY.txt](ARCHITECTURE_SUMMARY.txt)** (5 min read)
   - Quick overview of the entire system
   - Component relationships
   - Key design patterns

2. **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** (5 min read)
   - Navigation guide through CLAUDE.md
   - Common tasks and where to find help
   - Quick reference tables

3. **[CLAUDE.md](CLAUDE.md)** (Read sections relevant to your task)
   - Deep dive into specific areas
   - Design patterns with code examples
   - Extension points and examples

4. **Integration Test Example**
   - Read: `/particula/integration_tests/quick_start_test.py`
   - See: Complete workflow from start to finish
   - Learn: How to use builders and processes

## Documentation Files

| File | Size | Purpose |
|------|------|---------|
| CLAUDE.md | 17 KB | Complete architecture reference |
| ARCHITECTURE_SUMMARY.txt | 3.7 KB | One-page quick reference |
| DOCUMENTATION_INDEX.md | 9.2 KB | Navigation and task guide |
| README_ARCHITECTURE_DOCS.md | This file | Documentation overview |

## What's Documented

### 1. CLAUDE.md
- Project overview and scope
- High-level architecture and core concepts
- Complete package structure
- 4 major design patterns with code examples
- Core classes and their relationships
- Testing structure and patterns
- Configuration files (pyproject.toml, mkdocs.yml, workflows)
- Key utilities and systems
- Development workflow
- Important conventions and patterns
- Extension points and examples

### 2. ARCHITECTURE_SUMMARY.txt
- Project scope (one screen)
- Core design pattern overview
- Main components diagram
- Key files and locations
- Testing approach
- Development tools
- Usage pattern flow
- Extension points

### 3. DOCUMENTATION_INDEX.md
- Onboarding checklist
- CLAUDE.md section guide
- Directory structure with annotations
- Module purpose reference table
- Common task workflows
- Important conventions
- Useful commands
- Quick reference for key classes

## Key Architectural Insights

### The Big Picture

Particula is built on a **multi-pattern architecture**:

```
Aerosol (Central Hub)
  ├─ Atmosphere (Gas Phase)
  │  ├─ Temperature, Pressure
  │  ├─ Partitioning Species
  │  └─ Gas-Only Species
  │
  └─ ParticleRepresentation (Particle Phase)
     ├─ Distribution Strategy (how represented)
     ├─ Activity Strategy (partial pressure)
     ├─ Surface Strategy (Kelvin effect)
     └─ Particle Data (distributions, density, etc.)

Processes (Chainable with |)
  ├─ MassCondensation
  ├─ Coagulation
  ├─ WallLoss
  └─ Dilution

Strategies (Pluggable Physics)
  ├─ Distribution Strategies (10+ types)
  ├─ Activity Strategies (5+ models)
  ├─ Coagulation Strategies (7+ kernels)
  ├─ Surface Strategies
  ├─ Condensation Strategies
  └─ Vapor Pressure Strategies

Builders (Fluent Constructors)
  └─ 20+ Mixin Classes for Unit Conversion
```

### Design Patterns Used

1. **Builder Pattern**: Fluent interfaces with automatic unit conversion
2. **Strategy Pattern**: Pluggable physics models without core modification
3. **Factory Pattern**: Configuration-driven strategy instantiation
4. **Process Chain**: Composable operations with | operator
5. **Mixin Pattern**: Reusable setter functionality across builders

### Key Insights

- **Builder Mixin System (769 lines)**: Core innovation enabling unit-aware construction
- **ParticleRepresentation (553 lines)**: Orchestrates particle state elegantly
- **Process Chaining**: Simple syntax for complex simulations
- **Strategy Extensibility**: Add new physics without modifying core
- **Complete Unit Support**: Accept ANY unit, convert to SI internally

## Common Tasks

### Add a New Physics Model
1. Identify which Strategy ABC to extend
2. Implement in appropriate module
3. Register in factory file
4. Add tests
5. See DOCUMENTATION_INDEX.md for specific guidance

### Modify Builders
1. Either extend BuilderABC + mixins
2. Or add new mixin to builder_mixin.py
3. Follow fluent interface pattern (return self)
4. Support unit conversion in setters

### Create New Process
1. Implement RunnableABC
2. Implement rate() and execute() methods
3. Chain with | operator: `process1 | process2`
4. Add tests in appropriate tests/ directory

### Run Tests
```bash
pytest /home/kyle/Code/particula/
```

### Check Code Style
```bash
ruff check /home/kyle/Code/particula/particula/
```

## Quick References

### Key Files
- `/particula/aerosol.py` - Central Aerosol class
- `/particula/particles/representation.py` - Particle state (553 lines)
- `/particula/builder_mixin.py` - Unit conversion mixins (769 lines!)
- `/particula/dynamics/particle_process.py` - Main processes
- `/particula/util/constants.py` - Physics constants

### Key Patterns
- Builders: `.set_param(value, "units").build()`
- Processes: `process1 | process2 | process3.execute(aerosol, time_step)`
- Strategies: Implement ABC, register in factory
- Constants: All from scipy.constants
- Logging: `logging.getLogger("particula")`

### Important Conventions
1. All builders return `self` for method chaining
2. All paths in docs use absolute paths
3. Unit conversion is automatic and transparent
4. Tests use pytest and *_test.py naming
5. Docstrings follow Google style
6. Code formatted to 80 characters
7. New physics added via Strategy pattern
8. Arrays use `.get_X(clone=True)` for deep copy

## For Future Claude Code Instances

This documentation was created with the following goals:

1. **Reduce onboarding time**: Start with 5-minute overview
2. **Provide clear navigation**: DOCUMENTATION_INDEX.md guides you
3. **Explain architecture deeply**: CLAUDE.md has complete details
4. **Show examples**: Code examples in each pattern section
5. **Enable extensions**: Clear extension points identified

Use this documentation to:
- Understand the existing architecture quickly
- Navigate the codebase efficiently
- Know where to add new features
- Follow established patterns and conventions
- Write code that fits the project style

## Contact and Resources

- **Project**: https://github.com/uncscode/particula
- **Documentation**: https://uncscode.github.io/particula
- **PyPI**: https://pypi.org/project/particula/
- **Citation**: See /citation file

## Documentation Status

- Created: 2025-10-18
- Analysis Type: Very Thorough
- Coverage: Complete architecture with patterns, components, testing, and extensions
- Quality: Production-ready for Claude Code usage

---

*This documentation set provides everything needed for Claude Code instances to quickly understand and work effectively with the Particula codebase.*
