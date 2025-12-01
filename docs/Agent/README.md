# Agent Documentation for particula

**Version:** 0.2.6
**Last Updated:** 2025-11-30

## Overview

This directory contains documentation guides for ADW (AI Developer Workflow) agents working on the **particula** repository. These guides define coding standards, testing requirements, linting rules, and development workflows specific to particula.

## Quick Navigation

### Core Development Guides

| Guide | Purpose | Key Info |
|-------|---------|----------|
| **[Testing Guide](testing_guide.md)** | Test framework and requirements | pytest, 500+ tests, *_test.py pattern |
| **[Linting Guide](linting_guide.md)** | Code quality tools | ruff (check + format), mypy |
| **[Code Style Guide](code_style.md)** | Naming and formatting conventions | snake_case functions, 80-char lines |
| **[Docstring Guide](docstring_guide.md)** | Documentation standards | Google-style, type hints required |

### Workflow Guides

| Guide | Purpose | Key Info |
|-------|---------|----------|
| **[Commit Conventions](commit_conventions.md)** | Commit message format | Conventional Commits, imperative mood |
| **[PR Conventions](pr_conventions.md)** | Pull request standards | GitHub, CI checks, ~100 lines |
| **[Review Guide](review_guide.md)** | Code review criteria | Scientific correctness, test coverage |
| **[Code Culture](code_culture.md)** | Development philosophy | 100-line rule, "smooth is safe, safe is fast" |

### Reference Guides

| Guide | Purpose | Key Info |
|-------|---------|----------|
| **[Documentation Guide](documentation_guide.md)** | Documentation practices | MkDocs, Markdown, auto-generated API docs |
| **[Architecture Reference](architecture_reference.md)** | Architecture documentation | Module structure, design decisions |
| **[Issue Interpret Guide](issue_interpret_guide.md)** | Issue interpretation | Convert user requests to structured issues |

### Specialized Guides

| Guide | Purpose |
|-------|---------|
| **[Workflow Engine](workflow-engine.md)** | ADW workflow system |
| **[Workflow Conditionals](workflow-conditionals.md)** | Conditional workflow logic |
| **[Workflow JSON Schema](workflow-json-schema.md)** | Workflow configuration format |
| **[Workflow Examples](workflow-examples.md)** | Example workflows |
| **[Workflow Migration Guide](workflow-migration-guide.md)** | Migration from Python workflows |
| **[OpenCode Advanced Features](opencode-advanced-features.md)** | Advanced ADW features |
| **[Conditional Docs](conditional_docs.md)** | Conditional documentation updates |
| **[Agent Fallback System](agent_fallback_system.md)** | Agent error handling |

## Agent Subdirectories

### agents/

Agent-specific configuration and documentation for each ADW agent type:

- `adw-default.md` - Default ADW agent configuration
- `architecture-subagent.md` - Architecture documentation agent
- `build-subagent.md` - Build/implementation agent
- `docs-feature-subagent.md` - Feature documentation agent
- `docs-maintenance-subagent.md` - Maintenance documentation agent
- `docs-subagent.md` - General documentation agent
- `examples-subagent.md` - Examples documentation agent
- `lint-subagent.md` - Linting agent
- `review-subagent.md` - Code review agent
- `test-subagent.md` - Testing agent
- `theory-subagent.md` - Theory documentation agent

### architecture/

Architecture documentation and design decisions:

- `decisions/` - Architecture Decision Records (ADRs)
  - `adr-001-workflow-engine.md` - Workflow engine design
  - `adr-002-conditional-docs.md` - Conditional documentation system
- Project structure and module organization
- Design patterns and conventions

### feature/

Feature development tracking:

- `README.md` - Feature development process
- `template.md` - Feature documentation template

### maintenance/

Maintenance task tracking:

- `README.md` - Maintenance process
- `template.md` - Maintenance task template

## Repository Information

### Project Details

- **Package Name**: particula
- **Version**: 0.2.6
- **Language**: Python 3.9+
- **Repository**: https://github.com/uncscode/particula.git
- **Main Branch**: main
- **Description**: A simple, fast, and powerful particle simulator

### Development Stack

- **Testing**: pytest (minimum 500 tests, currently 711)
- **Linting**: ruff check + format, mypy (optional)
- **Documentation**: MkDocs with Material theme
- **CI/CD**: GitHub Actions (test, lint, docs build)
- **Package Manager**: pip, uv, or conda
- **Build System**: flit

### Code Quality Standards

- **Line Length**: 80 characters
- **Docstring Style**: Google-style (configured in pyproject.toml)
- **Test Pattern**: `*_test.py`
- **Type Hints**: Required for all public functions
- **Test Coverage**: ≥90% for new code

## Getting Started for ADW Agents

### 1. Read Core Guides First

Start with these essential guides:

1. [Code Style Guide](code_style.md) - Naming conventions and formatting
2. [Testing Guide](testing_guide.md) - How to write and run tests
3. [Linting Guide](linting_guide.md) - Code quality requirements
4. [Docstring Guide](docstring_guide.md) - Documentation standards

### 2. Understand Workflows

Review workflow guides to understand the development process:

1. [Commit Conventions](commit_conventions.md) - How to write commit messages
2. [PR Conventions](pr_conventions.md) - How to create pull requests
3. [Review Guide](review_guide.md) - What reviewers check
4. [Code Culture](code_culture.md) - Development philosophy

### 3. Check Agent-Specific Docs

If you're a specialized agent, check `agents/` for your configuration:

- Build agents → `agents/build-subagent.md`
- Test agents → `agents/test-subagent.md`
- Documentation agents → `agents/docs-subagent.md`
- etc.

### 4. Reference as Needed

Keep these handy for reference:

- [Architecture Reference](architecture_reference.md) - Module structure
- [Documentation Guide](documentation_guide.md) - MkDocs and Markdown
- [Workflow Examples](workflow-examples.md) - Example workflows

## Quick Reference Commands

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=particula --cov-report=term-missing

# Using ADW tool
.opencode/tool/run_pytest.py
```

### Linting
```bash
# Run linters (auto-fix, format, check)
ruff check particula/ --fix
ruff format particula/
ruff check particula/

# Using ADW tool
.opencode/tool/run_linters.py

# Optional: Type checking
mypy particula/ --ignore-missing-imports
```

### Documentation
```bash
# Build documentation
mkdocs build

# Serve with live reload
mkdocs serve

# Access at http://localhost:8000
```

## Particula-Specific Conventions

### Scientific Computing

particula is a **scientific computing package** for atmospheric particle simulations. Special considerations:

1. **Physical Units**: Always document units in docstrings and comments
2. **Numerical Stability**: Consider edge cases (zero, negative, very large/small values)
3. **Vectorization**: Use NumPy for array operations
4. **Citations**: Include paper references in module docstrings
5. **Validation**: Use `@validate_inputs` decorator for public functions

### Module Structure

```
particula/
├── activity/          # Activity coefficients, phase separation
├── dynamics/          # Particle dynamics (coagulation, condensation)
│   ├── coagulation/
│   ├── condensation/
│   └── properties/
├── equilibria/        # Thermodynamic equilibrium
├── gas/               # Gas phase, species, vapor pressure
│   └── properties/
├── particles/         # Particle distributions and representations
│   ├── distribution_strategies/
│   └── properties/
└── util/              # Utilities, constants, validation
```

### Common Patterns

**Input Validation:**
```python
from particula.util.validate_inputs import validate_inputs

@validate_inputs({
    "temperature": "positive",
    "pressure": "positive",
})
def calculate_property(temperature: float, pressure: float) -> float:
    """Calculate property from temperature and pressure."""
    pass
```

**Type Hints for Scientific Functions:**
```python
from typing import Union
from numpy.typing import NDArray
import numpy as np

def calculate_density(
    mass: Union[float, NDArray[np.float64]],
    volume: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate density (supports scalars and arrays)."""
    return mass / volume
```

## Integration with particula Project

These guides integrate with particula's existing documentation:

- **Contributing Guide**: `docs/contribute/CONTRIBUTING.md` - General contribution workflow
- **Code Specifications**: `docs/contribute/Code_Specifications/` - Detailed code standards
- **Feature Workflow**: `docs/contribute/Feature_Workflow/` - Feature development process
- **Examples**: `docs/Examples/` - Tutorial notebooks and examples
- **Theory**: `docs/Theory/` - Theoretical background

## Support and Questions

### For Development Questions

- Review existing code in `particula/` for examples
- Check `docs/contribute/CONTRIBUTING.md` for general guidelines
- See `docs/Examples/` for usage patterns

### For ADW Questions

- Check `docs/Agent/` guides (this directory)
- Review agent-specific docs in `agents/`
- See workflow guides for process questions

### For Scientific Questions

- Check module docstrings for paper references
- See `docs/Theory/` for theoretical background
- Review implementation in source code

## Version History

- **0.2.6** (2025-11-30) - Updated all Agent guides with particula-specific information
- **0.2.5** - Added testing, linting, and code style guides
- **0.2.4** - Initial ADW integration

## See Also

- **[AGENTS.md](../../AGENTS.md)** - Quick reference for particula development
- **[Contributing Guide](../contribute/CONTRIBUTING.md)** - Full contribution workflow
- **[Project README](../../readme.md)** - Main project documentation
- **[MkDocs Site](https://uncscode.github.io/particula)** - Online documentation
