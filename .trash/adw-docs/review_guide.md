# Code Review Guide

**Version:** 0.2.6
**Last Updated:** 2025-11-30

## Overview

This guide documents code review standards and criteria for the **particula** repository. It defines what reviewers should check, how to provide feedback, and what standards must be met before code can be merged.

> **See Also:** [Code Culture](code_culture.md) - Development philosophy, the 100-line rule, and review culture principles.

### Repository Structure

```
particula/
├── particula/              # Source code (production)
│   ├── activity/          # Activity coefficients, phase separation
│   ├── dynamics/          # Coagulation, condensation, wall loss, dilution
│   ├── equilibria/        # Thermodynamic equilibrium calculations
│   ├── gas/               # Gas phase, species, vapor pressure
│   ├── particles/         # Particle distributions and properties
│   ├── util/              # Utility functions, constants, validation
│   └── integration_tests/ # Integration tests
├── docs/                   # Documentation (MkDocs)
│   ├── Agent/             # ADW agent documentation
│   ├── Examples/          # Example notebooks and tutorials
│   ├── Theory/            # Theoretical background
│   └── contribute/        # Contribution guides
├── .github/                # GitHub Actions CI/CD workflows
├── pyproject.toml          # Package configuration, ruff, pytest
├── mkdocs.yml              # Documentation configuration
└── readme.md               # Project README
```

### Validation Commands

Before submitting code for review, authors must run:

```bash
# Run tests with coverage
pytest --cov=particula --cov-report=term-missing

# Run linters (auto-fix, format, final check)
ruff check particula/ --fix
ruff format particula/
ruff check particula/

# Optional: Run type checker
mypy particula/ --ignore-missing-imports

# Optional: Build documentation locally
mkdocs build
```

All commands should complete without errors before requesting review.

### Review Checklist

Reviewers should verify:

- [ ] **Tests pass**: `pytest --cov=particula`
  - Minimum 500 tests must run
  - New code has ≥90% coverage
  - Tests are meaningful (not just coverage targets)
- [ ] **Linting passes**: Ruff check + format
  - No ruff errors
  - Code follows 80-character line limit
  - Google-style docstrings for all public APIs
- [ ] **Type hints present**: All function signatures have type hints
  - Use `Union[float, NDArray[np.float64]]` for scientific functions
  - Import from `typing` and `numpy.typing`
- [ ] **Documentation updated**:
  - Module docstrings present
  - Function/class docstrings complete
  - Examples updated if behavior changes
  - Citations included for scientific algorithms
- [ ] **Scientific correctness** (if applicable):
  - Algorithms match cited papers
  - Numerical methods are appropriate
  - Units are consistent and documented
  - Edge cases are handled
- [ ] **Code quality**:
  - Names are descriptive (`snake_case` for functions/variables)
  - No magic numbers (use named constants)
  - Input validation for public functions (`@validate_inputs`)
  - No code duplication
- [ ] **PR size**: ~100 lines of production code (excluding tests/docs)
  - Large PRs should be split into phases
  - See [Code Culture](code_culture.md) for rationale

## Review Focus Areas

### 1. Scientific Correctness

For scientific code (most of particula), reviewers must verify:

**Algorithm Implementation**
- Matches description in cited paper
- Handles units correctly (SI units preferred)
- Numerical methods are appropriate for the problem
- Boundary conditions are correct
- Convergence criteria are reasonable

**Physical Validity**
- Results are physically plausible
- Conservation laws are respected (mass, energy, etc.)
- Dimensionality is correct (scalars vs arrays)
- Edge cases don't produce NaN or Inf

**Examples to Check:**
```python
# Good: Clear units, validation, physical limits
@validate_inputs({"temperature": "positive"})
def vapor_pressure(temperature: float) -> float:
    """Calculate saturation vapor pressure using Clausius-Clapeyron.
    
    Args:
        temperature: Temperature in Kelvin. Must be positive.
    
    Returns:
        Vapor pressure in Pascals.
    """
    if temperature < 173.15 or temperature > 373.15:
        raise ValueError("Temperature outside valid range (173-373 K)")
    # Implementation...

# Bad: No validation, unclear units, no bounds checking
def vapor_pressure(temp):
    """Calculate vapor pressure."""
    return 611 * np.exp(5423 * (1/273 - 1/temp))
```

### 2. Code Quality

**Readability**
- Variable names are descriptive and follow conventions
- Complex calculations have explanatory comments
- Functions are focused (single responsibility)
- Magic numbers are replaced with named constants

**Maintainability**
- No code duplication (DRY principle)
- Dependencies are explicit (imports at top)
- Side effects are minimized
- Functions are testable in isolation

**Performance**
- NumPy vectorization used appropriately
- No premature optimization
- Performance-critical code has benchmarks

### 3. Testing Quality

**Coverage**
- New code has ≥90% test coverage
- Both success and failure paths tested
- Edge cases covered (zero, negative, very large/small values)
- Array and scalar inputs tested (for scientific functions)

**Test Quality**
- Tests are independent (can run in any order)
- Tests are deterministic (no random failures)
- Test names are descriptive
- Assertions are meaningful

**Example:**
```python
# Good test
def test_density_calculation_scalar():
    """Test density calculation with scalar inputs."""
    mass = 10.0  # kg
    volume = 2.0  # m³
    expected_density = 5.0  # kg/m³
    
    result = calculate_density(mass, volume)
    
    assert result == expected_density

def test_density_calculation_array():
    """Test density calculation with array inputs."""
    masses = np.array([10.0, 20.0, 30.0])  # kg
    volumes = np.array([2.0, 4.0, 6.0])  # m³
    expected = np.array([5.0, 5.0, 5.0])  # kg/m³
    
    result = calculate_density(masses, volumes)
    
    np.testing.assert_array_equal(result, expected)

def test_density_raises_on_zero_volume():
    """Test that zero volume raises ValueError."""
    with pytest.raises(ValueError, match="Volume must be positive"):
        calculate_density(10.0, 0.0)
```

### 4. Documentation Quality

**Docstrings**
- Present for all public functions and classes
- Follow Google style (configured in `pyproject.toml`)
- Include Args, Returns, Raises, Examples sections
- Examples are valid (can be tested with doctest)
- Scientific citations included in module docstrings

**Code Comments**
- Explain **why**, not **what**
- Complex algorithms have references
- Units are specified for physical quantities
- Assumptions are documented

## Issue Severity Levels

When providing feedback, categorize issues by severity:

### Blocker (Must Fix Before Merge)
- Tests failing
- Linting errors
- Security vulnerabilities
- Scientific errors (incorrect algorithms)
- Breaking changes to public API without migration path
- Missing documentation for public APIs

### Major (Should Fix Before Merge)
- Insufficient test coverage (<90% for new code)
- Missing type hints
- Unclear variable names
- Significant performance issues
- Missing scientific citations
- Inconsistent coding style

### Minor (Nice to Have)
- Additional test cases for edge cases
- Improved comments
- Refactoring opportunities
- Non-critical performance improvements
- Documentation enhancements

## Review Culture

### For Reviewers

**Be Constructive:**
- Praise good work
- Ask questions before making statements
- Suggest improvements, don't demand perfection
- Explain **why** changes are needed
- Assume good intent

**Be Thorough:**
- Review logic, not just style
- Check scientific correctness
- Verify tests are meaningful
- Consider edge cases
- Test locally if needed

**Be Timely:**
- Review small PRs (~100 lines) within 4 hours
- Larger PRs within 24 hours
- Provide initial feedback quickly
- Unblock teammates promptly

### For Authors

**Before Requesting Review:**
- Run all validation commands locally
- Self-review your changes
- Provide context in PR description
- Highlight areas needing special attention
- Ensure PR is reasonably sized (~100 lines)

**During Review:**
- Respond to comments promptly
- Ask for clarification if feedback is unclear
- Don't take criticism personally
- Push fixes quickly
- Thank reviewers for their time

## Integration with ADW

ADW review commands use this guide to validate:
- Code passes all validation commands
- Scientific correctness for algorithm implementations
- Test coverage meets minimum requirements (≥90%)
- Documentation is complete and properly formatted
- Issue severity is appropriately assessed

## See Also

- **[Testing Guide](testing_guide.md)**: Test framework, commands, and requirements
- **[Linting Guide](linting_guide.md)**: Linting tools and standards
- **[Code Style Guide](code_style.md)**: Naming conventions and formatting
- **[Code Culture](code_culture.md)**: Development philosophy and the 100-line rule
- **[Docstring Guide](docstring_guide.md)**: Documentation standards
