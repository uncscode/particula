# Agent Documentation for particula

**Version:** 0.2.6
**Last Updated:** 2025-12-21

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
| **[Architecture Reference](architecture_reference.md)** | Architecture documentation | Module structure, design decisions |
| **[Conditional Docs](conditional_docs.md)** | Task-driven documentation | When to read each guide |

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
- **Linting**: ruff check + format, mypy
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

## Quick Reference Commands

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=particula --cov-report=term-missing

# Using ADW tool
.opencode/tool/run_pytest.py

# Slow + performance benchmarks (excluded from CI)
pytest particula/dynamics/condensation/tests/staggered_performance_test.py -v -m "slow and performance"
```
Targets: <2x overhead vs simultaneous, ~O(n) scaling at 1k/10k/100k, and theta-mode
comparisons (half/random/batch) with deterministic seeds.

### Linting
```bash
# Run linters (auto-fix, format, check)
ruff check particula/ --fix
ruff format particula/
ruff check particula/

# Using ADW tool
.opencode/tool/run_linters.py

# Type checking
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

## Development Plans

Active development plans are tracked in the [dev-plans/](dev-plans/) directory:

- **Features**: [dev-plans/features/](dev-plans/features/)
- **Maintenance**: [dev-plans/maintenance/](dev-plans/maintenance/)
- **Epics**: [dev-plans/epics/](dev-plans/epics/)

See [dev-plans/README.md](dev-plans/README.md) for current plans.

## Architecture Documentation

Architecture decisions and guides are in the [architecture/](architecture/) directory:

- **[Architecture Guide](architecture/architecture_guide.md)**: Patterns and principles
- **[Architecture Outline](architecture/architecture_outline.md)**: System overview
- **[Decision Records](architecture/decisions/)**: ADRs documenting key decisions

## See Also

- **[AGENTS.md](../AGENTS.md)** - Quick reference for particula development
- **[Contributing Guide](../docs/contribute/CONTRIBUTING.md)** - Full contribution workflow
- **[Project README](../readme.md)** - Main project documentation
- **[MkDocs Site](https://uncscode.github.io/particula)** - Online documentation
