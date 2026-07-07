# Agent Documentation for particula

**Project:** particula  
**Version:** 0.2.6  
**Last Updated:** 2026-06-06

`.opencode/guides/` is the agent-facing documentation directory for the
particula repository. These guides capture repository-specific conventions for
coding style, tests, linting, review, documentation, and architecture.

## Quick Navigation

### Core Development Guides

| Guide | Purpose | Key Info |
|-------|---------|----------|
| **[Testing Guide](testing_guide.md)** | Test framework and requirements | pytest, `*_test.py`, warning-free CI |
| **[Linting Guide](linting_guide.md)** | Code quality tools | ruff check + format, mypy |
| **[Code Style Guide](code_style.md)** | Naming and formatting conventions | Python 3.12+, 80-char lines |
| **[Docstring Guide](docstring_guide.md)** | Docstring format standards | Google-style docstrings |
| **[Documentation Guide](documentation_guide.md)** | Documentation workflows | MkDocs, examples, Jupytext notebooks |

### Workflow Guides

| Guide | Purpose | Key Info |
|-------|---------|----------|
| **[Commit Conventions](commit_conventions.md)** | Commit message format | Conventional Commits, imperative mood |
| **[PR Conventions](pr_conventions.md)** | Pull request standards | GitHub, CI checks, small focused PRs |
| **[Review Guide](review_guide.md)** | Code review criteria | Scientific correctness, tests, maintainability |
| **[Code Culture](code_culture.md)** | Development philosophy | Small changes, safety, clarity |

### Reference Guides

| Guide | Purpose | Key Info |
|-------|---------|----------|
| **[Architecture Guide](architecture/architecture_guide.md)** | Architecture documentation | Module structure and design decisions |
| **[Architecture Reference](architecture_reference.md)** | Architecture quick reference | Repository module map |
| **[Notebook Validation Guide](notebook_validation_guide.md)** | Notebook tooling | Jupytext sync, validation, execution |
| **[Conditional Docs](conditional_docs.md)** | Task-driven documentation | Which guides to read for each task |

## Repository Information

- **Package Name:** `particula`
- **Language:** Python 3.12+
- **Description:** A simple, fast, and powerful particle simulator
- **Repository:** `https://github.com/Gorkowski/particula.git`
- **Main Branch:** `main`

## Development Stack

- **Testing:** pytest with pytest-cov
- **Linting:** ruff check, ruff format, mypy
- **Documentation:** MkDocs with Material theme
- **Build System:** flit
- **Package Managers:** pip, uv, or conda

## Code Quality Standards

- **Line Length:** 80 characters
- **Docstring Style:** Google-style
- **Test Pattern:** `*_test.py`
- **Type Hints:** Required for public APIs where practical
- **Test Coverage:** Add focused tests for new behavior
- **Scientific Code:** Prefer vectorized NumPy operations and explicit units

## Quick Reference Commands

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=particula --cov-report=term-missing

# Match warning behavior used by CI
pytest -Werror

# Slow + performance benchmarks, excluded from normal CI
pytest particula/dynamics/condensation/tests/staggered_performance_test.py -v -m "slow and performance"
```

The staggered condensation performance benchmarks verify O(n) scaling at
1k/10k/100k particles, theta-mode comparisons (`half`, `random`, `batch`), and
deterministic seeds. Staggered stepping is Gauss-Seidel and sequential, so high
overhead relative to simultaneous vectorized stepping is expected.

### Linting

```bash
# Run linters in CI order
ruff check particula/ --fix
ruff format particula/
ruff check particula/
mypy particula/ --ignore-missing-imports
```

### Documentation

```bash
# Build documentation
mkdocs build

# Serve locally
mkdocs serve
```

## Example Source Workflow

`docs/Examples/` can contain runnable `.py` examples, paired notebooks, or
both. When an example is notebook-backed, edit the `.py` percent file, not the
`.ipynb` directly.

```bash
ruff check docs/Examples/path/to/file.py --fix
ruff format docs/Examples/path/to/file.py
python3 .opencode/tools/validate_notebook.py docs/Examples/path/to/file.ipynb --sync
python3 .opencode/tools/run_notebook.py docs/Examples/path/to/file.ipynb
```

Commit both the `.py` and `.ipynb` files after sync and execution when the
example ships as a notebook.

## Architecture Documentation

Architecture decisions and guides live under [architecture/](architecture/):

- **[Architecture Guide](architecture/architecture_guide.md)**: Patterns and principles
- **[Architecture Outline](architecture/architecture_outline.md)**: System overview
- **[Decision Records](architecture/decisions/)**: ADRs documenting key decisions

## See Also

- **[AGENTS.md](../../AGENTS.md)**: Quick reference for particula development
- **[Project README](../../readme.md)**: Main project documentation
- **[Contributing Guide](../../docs/contribute/CONTRIBUTING.md)**: Contribution workflow
