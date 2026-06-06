# Developer Documentation

**Project:** particula  
**Version:** 0.2.6  
**Last Updated:** 2026-06-06

This directory contains agent-facing guides for contributors working on the
particula particle-simulation codebase. User-facing project documentation lives
under `docs/` and is built by MkDocs.

## Core Guides

- **[Code Style Guide](code_style.md)**: Python standards, naming, formatting, type hints, and scientific-code patterns.
- **[Testing Guide](testing_guide.md)**: pytest commands, file naming, warnings-as-errors, and performance benchmark guidance.
- **[Linting Guide](linting_guide.md)**: ruff and mypy configuration, commands, and CI expectations.
- **[Docstring Guide](docstring_guide.md)**: Google-style docstrings and module documentation.
- **[Documentation Guide](documentation_guide.md)**: MkDocs and notebook documentation workflow.
- **[Review Guide](review_guide.md)**: Review criteria for correctness, tests, scientific validity, and maintainability.

## Architecture

- **[Architecture Outline](architecture/architecture_outline.md)**: High-level module map.
- **[Architecture Guide](architecture/architecture_guide.md)**: Design patterns and architecture rules.
- **[Architecture Reference](architecture_reference.md)**: Quick reference for module boundaries.
- **[Decision Records](architecture/decisions/README.md)**: ADRs for important design choices.

## Process Guides

- **[Commit Conventions](commit_conventions.md)**: Conventional commit format.
- **[PR Conventions](pr_conventions.md)**: Pull request structure and expectations.
- **[Code Culture](code_culture.md)**: Small, safe, clear development practices.
- **[Conditional Docs](conditional_docs.md)**: Which guide to read for different task types.
- **[Notebook Validation Guide](notebook_validation_guide.md)**: Jupytext validation and execution details.

## Repository Quick Reference

```bash
# Install for development
pip install -e .[dev]

# Test
pytest
pytest --cov=particula --cov-report=term-missing
pytest -Werror

# Lint and type check
ruff check particula/ --fix
ruff format particula/
ruff check particula/
mypy particula/ --ignore-missing-imports

# Documentation
mkdocs build
```

## Project Structure

```text
particula/
├── activity/          # Activity coefficients and phase separation
├── dynamics/          # Coagulation, condensation, wall loss
├── equilibria/        # Partitioning calculations
├── gas/               # Gas phase, species, vapor pressure
├── particles/         # Particle distributions and representations
├── util/              # Constants, validation, chemistry utilities
└── integration_tests/ # Cross-module integration tests
```

## Key Standards

- Use Python 3.12+ syntax.
- Keep formatted lines to 80 characters.
- Use `snake_case` for functions, variables, and modules.
- Use `PascalCase` for classes and builders.
- Use `UPPER_CASE` for constants.
- Put tests in module-local `tests/` directories and name files `*_test.py`.
- Prefer NumPy vectorization for scientific calculations.
- Use `particula.util.validate_inputs.validate_inputs` for public input validation.
- Cite scientific sources in module docstrings or comments when implementing equations.

## Additional Resources

- **[AGENTS.md](../../AGENTS.md)**: Repository quick reference for agents.
- **[Project README](../../readme.md)**: Main package overview.
- **[Contributing Guide](../../docs/contribute/CONTRIBUTING.md)**: Contributor workflow.
