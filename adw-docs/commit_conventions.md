# Commit Conventions

**Version:** 0.2.6
**Last Updated:** 2025-11-30

## Overview

This guide documents commit message format for the **particula** repository. Consistent commit messages help maintain a clear project history and make it easier to understand changes, generate changelogs, and track features and fixes.

### Commit Message Format

particula uses **Conventional Commits** format with imperative mood.

**Structure:**
```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Example:**
```
feat(dynamics): add coagulation kernel for Brownian motion

Implements the Fuchs form of the Brownian coagulation kernel
for spherical particles. Includes slip correction factor.

Closes #123
```

### Commit Types

particula uses the following commit types:

- **feat**: New feature or functionality
  - Example: `feat(gas): add vapor pressure calculation for organics`
- **fix**: Bug fix
  - Example: `fix(particles): correct mass conservation in condensation`
- **docs**: Documentation changes only
  - Example: `docs(examples): add coagulation simulation notebook`
- **test**: Adding or updating tests
  - Example: `test(activity): add tests for BAT model edge cases`
- **refactor**: Code changes that neither fix bugs nor add features
  - Example: `refactor(util): simplify input validation decorator`
- **perf**: Performance improvements
  - Example: `perf(coagulation): vectorize kernel calculations`
- **style**: Code style changes (formatting, missing semicolons, etc.)
  - Example: `style: apply ruff formatting to all modules`
- **build**: Changes to build system or dependencies
  - Example: `build: update numpy to >=2.0.0`
- **ci**: Changes to CI/CD configuration
  - Example: `ci: add mypy type checking to GitHub Actions`
- **chore**: Other changes that don't modify src or test files
  - Example: `chore: update .gitignore for VSCode files`

### Scope

The scope should indicate which part of the codebase is affected. Common scopes in particula:

- `activity` - Activity coefficients and phase separation
- `dynamics` - Particle dynamics (coagulation, condensation, dilution, wall loss)
- `equilibria` - Thermodynamic equilibrium calculations
- `gas` - Gas phase and species
- `particles` - Particle distributions and properties
- `util` - Utility functions and constants
- `tests` - Test-related changes
- `docs` - Documentation
- `ci` - Continuous integration
- `build` - Build system and dependencies

**Examples:**
```
feat(particles): add lognormal distribution builder
fix(dynamics): correct time step handling in condensation
docs(examples): add chamber wall loss simulation
test(gas): add vapor pressure calculation tests
```

### Message Tense

**Tense**: Use **imperative mood** (present tense command form)

Write commit messages as if giving an order or instruction.

**Examples:**
- ✓ Correct: `add coagulation kernel for Brownian motion`
- ✓ Correct: `fix density calculation in particle class`
- ✓ Correct: `update documentation for BAT model`
- ✗ Wrong: `added coagulation kernel` (past tense)
- ✗ Wrong: `adding coagulation kernel` (present continuous)
- ✗ Wrong: `adds coagulation kernel` (present tense)

### Length Limits

- **Subject line**: 72 characters maximum (enforced in code review)
- **Body**: 72 characters per line (wrap text)
- **First line**: Brief, complete summary of the change

### Issue Linking

Link commits to GitHub issues using keywords in the commit body or footer.

**Format**: 
```
<type>(<scope>): <description>

<optional body>

Closes #<issue-number>
```

**Closing Keywords**: `Closes`, `Fixes`, `Resolves` (automatically close issues)

**Reference Keywords**: `Refs`, `See`, `Related to` (just reference, don't close)

**Examples**:
```
fix(particles): correct surface area calculation

The surface area calculation was using diameter instead of radius.
This affected all particle property calculations.

Fixes #456
```

```
feat(dynamics): add new coagulation kernel

Implements Fuchs form for Brownian coagulation with slip correction.

Related to #789
Refs #123
```

## Scientific Computing Context

For scientific computing changes in particula, provide additional context:

### Algorithm Changes

When implementing or modifying scientific algorithms:

```
feat(activity): implement BAT activity coefficient model

Implements the Binary Activity Thermodynamics (BAT) model
for calculating activity coefficients in organic-water mixtures.
Based on Gorkowski et al. (2019) ACP.

Key features:
- Supports multiple functional groups
- Handles phase separation
- Vectorized for performance

References:
https://doi.org/10.5194/acp-19-13383-2019

Closes #234
```

### Numerical Changes

When fixing numerical issues:

```
fix(dynamics): improve numerical stability in condensation solver

Replace explicit Euler with implicit scheme to prevent
negative concentrations at large time steps.

Tested with dt up to 1000s without instability.

Fixes #345
```

### Performance Improvements

When optimizing scientific code:

```
perf(coagulation): vectorize kernel matrix computation

Replace nested loops with NumPy broadcasting, reducing
computation time from 5s to 0.2s for 100-bin distribution.

Benchmark results included in tests.

Related to #567
```

## Integration with ADW

ADW commit commands use this guide to:
- Format commit messages correctly
- Apply appropriate commit types
- Include scientific context for algorithm changes
- Link commits to issues using proper keywords

## See Also

- **[PR Conventions](pr_conventions.md)**: Pull request format
- **[Review Guide](review_guide.md)**: Code review process
- **[Contributing Guide](../contribute/CONTRIBUTING.md)**: Contribution workflow
