# Documentation Guide

**Version:** 0.2.6
**Last Updated:** 2026-01-30

## Overview

This guide documents the documentation standards, conventions, and workflows for the **particula** repository. It serves as the primary reference for ADW (AI Developer Workflow) agents and human contributors when creating, updating, or reviewing documentation.

### Documentation Philosophy

Particula follows a **layered documentation** approach:

1. **Code-level**: Docstrings in Python code (Google-style)
2. **Developer guides**: ADW documentation in `adw-docs/`
3. **User documentation**: MkDocs site in `docs/`
4. **Architecture records**: ADRs in `adw-docs/architecture/decisions/`

### Integration with ADW

This guide is referenced by the ADW documentation primary agent to coordinate specialized subagents. Each documentation layer has a dedicated subagent:

| Layer | Subagent | Scope |
|-------|----------|-------|
| Docstrings | `docstring` | `*.py` files |
| Developer guides | `docs` | `adw-docs/*.md`, `README.md` |
| Feature docs | `docs-feature` | `adw-docs/dev-plans/features/*.md` |
| Maintenance docs | `docs-maintenance` | `adw-docs/dev-plans/maintenance/*.md` |
| User examples | `examples` | `docs/Examples/*.md`, `.py`, `.ipynb` |
| Architecture | `architecture` | `adw-docs/architecture/*.md` |
| Theory | `theory` | `docs/Theory/*.md` |
| Features | `features` | `docs/Features/*.md` |
| Validation | `docs-validator` | All docs (read-only) |

## Documentation Structure

### Repository Layout

```
particula/
├── AGENTS.md                    # Quick reference for ADW agents
├── README.md                    # Project README (PyPI, GitHub)
├── readme.md                    # Alternative README
├── adw-docs/                    # Developer documentation
│   ├── README.md                # ADW docs index
│   ├── code_style.md            # Coding standards
│   ├── docstring_guide.md       # Docstring format
│   ├── testing_guide.md         # Testing standards
│   ├── linting_guide.md         # Linting tools
│   ├── commit_conventions.md    # Commit message format
│   ├── pr_conventions.md        # Pull request standards
│   ├── review_guide.md          # Code review criteria
│   ├── code_culture.md          # Development philosophy
│   ├── documentation_guide.md   # THIS FILE
│   ├── architecture_reference.md # Architecture entry point
│   ├── architecture/            # Architecture documentation
│   │   ├── architecture_guide.md
│   │   ├── architecture_outline.md
│   │   └── decisions/           # ADRs
│   └── dev-plans/               # Development plans
│       ├── features/            # Feature documentation
│       ├── maintenance/         # Maintenance docs
│       └── epics/               # Epic tracking
├── docs/                        # User documentation (MkDocs)
│   ├── index.md                 # Home page
│   ├── Examples/                # Tutorials and examples
│   │   ├── index.md
│   │   ├── Setup_Particula/
│   │   ├── Aerosol/
│   │   ├── Dynamics/
│   │   ├── Gas_Phase/
│   │   ├── Particle_Phase/
│   │   ├── Chamber_Wall_Loss/
│   │   ├── Equilibria/
│   │   ├── Nucleation/
│   │   ├── Activity/
│   │   └── Simulations/
│   ├── Features/                # High-level feature docs
│   │   ├── index.md
│   │   ├── activity_system.md
│   │   ├── condensation_strategy_system.md
│   │   ├── coagulation_strategy_system.md
│   │   └── wall_loss_strategy_system.md
│   ├── Theory/                  # Theoretical background
│   │   ├── index.md
│   │   ├── Activity_Calculations/
│   │   ├── Technical/
│   │   └── Code_Concepts/
│   └── contribute/              # Contribution guides
│       ├── CONTRIBUTING.md
│       ├── CODE_OF_CONDUCT.md
│       ├── Feature_Workflow/
│       └── Code_Specifications/
└── particula/                   # Source code with docstrings
    ├── activity/
    ├── dynamics/
    ├── gas/
    ├── particles/
    ├── equilibria/
    └── util/
```

### Documentation Types

| Type | Location | Purpose | Audience |
|------|----------|---------|----------|
| **Docstrings** | `*.py` files | API reference, function/class docs | Developers |
| **Developer guides** | `adw-docs/` | Coding standards, workflows | Contributors |
| **Feature docs** | `adw-docs/dev-plans/features/` | Feature planning, specs | Developers |
| **User examples** | `docs/Examples/` | Tutorials, how-to guides | Users |
| **Feature overviews** | `docs/Features/` | High-level feature docs | Users |
| **Theory** | `docs/Theory/` | Scientific background | Researchers |
| **Architecture** | `adw-docs/architecture/` | System design, ADRs | Architects |
| **README** | Root directory | Project overview | Everyone |

## Docstring Standards

### Format: Google-Style

All Python code uses **Google-style** docstrings as configured in `pyproject.toml`:

```toml
[tool.ruff.lint.pydocstyle]
convention = "google"
```

### Function Docstring Template

```python
from typing import Union
from numpy.typing import NDArray
import numpy as np

def calculate_density(
    mass: Union[float, NDArray[np.float64]],
    volume: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate density from mass and volume.
    
    Computes density using the fundamental relationship: density = mass / volume.
    Supports both scalar and array inputs for vectorized operations.
    
    Args:
        mass: Mass of the object in kilograms. Can be scalar or array.
        volume: Volume of the object in cubic meters. Can be scalar or array.
    
    Returns:
        Density in kg/m^3. Returns scalar if both inputs are scalars,
        otherwise returns array matching the broadcast shape.
    
    Raises:
        ValueError: If volume is zero or negative.
    
    Examples:
        >>> calculate_density(10.0, 2.0)
        5.0
        
        >>> import numpy as np
        >>> masses = np.array([10.0, 20.0, 30.0])
        >>> volumes = np.array([2.0, 4.0, 6.0])
        >>> calculate_density(masses, volumes)
        array([5., 5., 5.])
    """
    if np.any(volume <= 0):
        raise ValueError("Volume must be positive")
    return mass / volume
```

### Class Docstring Template

```python
class ParticleDistribution:
    """Represents a particle size distribution.
    
    Stores particle radii, masses, and concentrations for aerosol
    calculations. Supports discrete, continuous PDF, and particle-resolved
    distribution types.
    
    Attributes:
        radii: Particle radii in meters.
        masses: Particle masses in kilograms.
        concentrations: Number concentrations in 1/m^3.
        distribution_type: Type of distribution ("discrete", 
            "continuous_pdf", "particle_resolved").
    
    Examples:
        >>> dist = ParticleDistribution(radii=np.array([1e-7, 1e-6]))
        >>> dist.radii
        array([1.e-07, 1.e-06])
    """
    
    def __init__(
        self,
        radii: NDArray[np.float64],
        distribution_type: str = "discrete",
    ):
        """Initialize particle distribution.
        
        Args:
            radii: Particle radii in meters.
            distribution_type: Type of distribution.
        """
        self.radii = radii
        self.distribution_type = distribution_type
```

### Module Docstring Template

```python
"""Activity coefficients for organic-water mixtures.

This module implements the Binary Activity Thermodynamic (BAT) model
for calculating activity coefficients in organic-water aerosol systems.

Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""

import numpy as np
# ... rest of module
```

### Docstring Requirements

| Section | Required | Description |
|---------|----------|-------------|
| Brief description | Yes | One-line summary (imperative mood) |
| Extended description | Optional | Detailed explanation, equations |
| Args | Yes (if params) | All parameters documented |
| Returns | Yes (if returns) | Return value description |
| Raises | If applicable | Exceptions that can be raised |
| Examples | Recommended | Usage examples with expected output |
| References | If applicable | Scientific citations |

### Type Hints vs Docstrings

Use **type hints** in function signatures. Do **NOT** duplicate types in docstrings.

**Good:**
```python
def process(mass: float, volume: float) -> float:
    """Process mass and volume.
    
    Args:
        mass: Mass in kilograms.
        volume: Volume in cubic meters.
    
    Returns:
        Processed value.
    """
```

**Bad (redundant):**
```python
def process(mass, volume):
    """Process mass and volume.
    
    Args:
        mass (float): Mass in kilograms.
        volume (float): Volume in cubic meters.
    
    Returns:
        float: Processed value.
    """
```

## User Documentation (MkDocs)

### MkDocs Configuration

The user documentation is built with **MkDocs** and the **Material** theme:

```bash
# Build documentation
mkdocs build

# Serve locally with live reload
mkdocs serve

# Access at http://localhost:8000
```

### Examples Structure

Examples follow a consistent structure:

```
docs/Examples/
├── index.md                     # Examples gallery
├── Setup_Particula/
│   ├── index.md                 # Setup overview
│   └── Details/
│       ├── Setup_PIP.md
│       ├── Setup_Conda.md
│       └── Setup_UV.md
├── Dynamics/
│   ├── index.md                 # Dynamics overview
│   └── Notebooks/
│       └── *.ipynb              # Jupyter notebooks
└── Simulations/
    ├── index.md                 # Simulations overview
    └── Notebooks/
        └── *.ipynb              # End-to-end simulations
```

### Jupyter Notebook Guidelines

1. **Location**: Place notebooks in `docs/Examples/<category>/Notebooks/`
2. **Naming**: Use `Snake_Case_Title.ipynb` format
3. **Structure**:
   - Title cell (markdown, H1)
   - Overview/objectives
   - Prerequisites/imports
   - Step-by-step code cells
   - Results/visualization
   - Summary/next steps

4. **Validation**: Notebooks must execute without errors
5. **Output**: Include cell outputs for documentation build

```python
# Example notebook structure
# Cell 1: Title
"""
# Wall Loss Tutorial

Learn how to use wall loss strategies in particula.
"""

# Cell 2: Imports
import numpy as np
import particula as par

# Cell 3+: Tutorial content
...
```

### Jupytext Paired Sync Workflow

Notebooks in `docs/Examples/` use **Jupytext paired sync** for LLM-friendly
editing. Each `.ipynb` file has a corresponding `.py:percent` file that stays
in sync.

#### Why Paired Sync?

| Problem with Direct `.ipynb` | Solution with Paired Sync |
|------------------------------|---------------------------|
| JSON structure errors | Edit plain `.py` files |
| Cannot lint notebooks | Run ruff/mypy on `.py` |
| Unreadable PR diffs | Clean Python diffs |
| Merge conflicts nightmare | Standard git merge |
| No type checking | mypy works on `.py` |

#### File Structure

```
docs/Examples/Dynamics/Coagulation/
├── Coagulation_Tutorial.ipynb    # For users/MkDocs (JSON)
└── Coagulation_Tutorial.py       # For development (percent format)
```

Most notebooks in `docs/Examples/` have been migrated to Jupytext paired sync
format. To find all paired notebooks, look for `.py` files alongside `.ipynb`
files in the Examples directories:

```bash
# List all paired .py files
find docs/Examples -name "*.py" -type f | head -20
```

Categories with paired notebooks include:
- `Activity/` - Activity coefficient tutorials
- `Aerosol/` - Aerosol building tutorials
- `Chamber_Wall_Loss/Notebooks/` - Wall loss strategy examples
- `Dynamics/Coagulation/` - Coagulation tutorials and functional examples
- `Equilibria/Notebooks/` - Equilibria and activity examples
- `Gas_Phase/Notebooks/` - Gas species and atmosphere tutorials
- `Nucleation/Notebooks/` - Nucleation tutorials
- `Particle_Phase/Notebooks/` - Particle representation tutorials
- `Simulations/Notebooks/` - End-to-end simulation notebooks

All paired files use the `.py:percent` format with `# %%` cell markers.

#### LLM Editing Workflow

When editing notebooks, LLMs should follow this workflow:

```
1. EDIT the .py file (percent format)
   - Plain text, lintable, easy diffs
   - Uses # %% cell markers

2. LINT the .py file (catch syntax errors early)
   ruff check docs/Examples/path/to/file.py --fix
   ruff format docs/Examples/path/to/file.py

3. SYNC to regenerate .ipynb from edited .py
   validate_notebook({notebookPath: 'path/to/file.ipynb', sync: true})
   - The .py is newer, so it overwrites the .ipynb

4. EXECUTE .ipynb to validate code works AND generate outputs
   run_notebook({notebookPath: 'path/to/file.ipynb'})
   - If execution FAILS: the .py edit broke something, fix it
   - If execution PASSES: outputs (graphs, tables) are now in .ipynb
   - Creates .ipynb.bak backup automatically

5. COMMIT both files
   - .py for development (source of truth for code)
   - .ipynb for users/docs (with outputs for website)
```

#### Why This Order Matters

| Step | Purpose |
|------|---------|
| Lint first | Catches syntax errors before sync |
| Sync before execute | Transfers .py edits into .ipynb |
| Execute after sync | Validates code AND generates website outputs |

If you execute before syncing, you're testing the OLD code, not your edits!

#### Why Execution is Required

MkDocs renders notebooks with `execute: False`, meaning outputs must be
pre-stored in the `.ipynb` file. Without execution:
- Graphs and plots won't appear on the website
- Print statements and table outputs will be missing
- Users see empty code cells

#### Execution Also Validates Your Code

The `run_notebook` execution serves two purposes:
1. **Validation**: If execution fails, your `.py` edit broke something
2. **Output generation**: Successful execution stores graphs/tables in `.ipynb`

This means you don't need a separate "test the .py file" step - executing the
synced `.ipynb` tests it for you.

#### Percent Format Example

The `.py` file uses Jupytext percent format:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial Title
#
# This notebook demonstrates...

# %%
import numpy as np
import particula as par

# %% [markdown]
# ## Section 1
#
# Explanation text...

# %%
# Code cell
result = par.some_function()
print(result)
```

#### ADW Tool Commands

```python
# Check sync status (read-only, CI-friendly)
validate_notebook({
    "notebookPath": "docs/Examples",
    "recursive": True,
    "checkSync": True
})

# Sync all notebooks (bidirectional, newer wins)
validate_notebook({
    "notebookPath": "docs/Examples",
    "recursive": True,
    "sync": True
})

# Convert single notebook to .py
validate_notebook({
    "notebookPath": "notebook.ipynb",
    "convertToPy": True
})

# Execute notebook (overwrites with outputs, creates .bak backup)
run_notebook({
    "notebookPath": "docs/Examples/Activity/activity_tutorial.ipynb"
})

# Execute without overwriting (validation only)
run_notebook({
    "notebookPath": "notebook.ipynb",
    "noOverwrite": True
})

# Execute all notebooks in directory
run_notebook({
    "notebookPath": "docs/Examples/Activity",
    "recursive": True
})
```

#### CLI Commands

```bash
# Check sync status
python3 .opencode/tool/validate_notebook.py docs/Examples --recursive --check-sync

# Sync all notebooks
python3 .opencode/tool/validate_notebook.py docs/Examples --recursive --sync

# Convert to .py:percent
python3 .opencode/tool/validate_notebook.py notebook.ipynb --convert-to-py

# Execute notebook (overwrites with outputs)
python3 .opencode/tool/run_notebook.py docs/Examples/Activity/activity_tutorial.ipynb

# Execute without overwriting (validation only)
python3 .opencode/tool/run_notebook.py notebook.ipynb --no-overwrite

# Execute all notebooks in directory
python3 .opencode/tool/run_notebook.py docs/Examples/Activity --recursive

# Lint example Python files
ruff check docs/Examples/ --fix
ruff format docs/Examples/
```

#### Best Practices

1. **Always edit `.py` files** for code changes
2. **Sync after editing** to update the `.ipynb`
3. **Lint before committing** to catch style issues
4. **Commit both files** to keep them paired
5. **Use `--check-sync` in CI** to catch drift

#### Coagulation tutorial pairing (M4-P6)

- The coagulation tutorials were migrated to Jupytext percent format in the
  M4-P6 maintenance task. Keep the paired files in sync:
  - `docs/Examples/Dynamics/Coagulation/Coagulation_1_PMF_Pattern.py` /
    `.ipynb`
  - `docs/Examples/Dynamics/Coagulation/Coagulation_3_Particle_Resolved_Pattern.py`
    / `.ipynb`
  - `docs/Examples/Dynamics/Coagulation/Coagulation_4_Compared.py` / `.ipynb`
- Preserve the `particula_dev312` kernelspec in the notebook metadata after
  syncing.
- After edits: lint → sync → execute → check-sync to confirm the `.py` and
  `.ipynb` stay aligned and outputs are refreshed.

See [Maintenance Plan M3](dev-plans/maintenance/M3-jupytext-notebook-sync.md)
for migration details.

### Feature Documentation Structure

Feature docs in `docs/Features/` follow this template:

```markdown
# Feature Name

> Brief tagline describing the feature.

## Overview

What the feature does and why it exists.

## Key Benefits

- Benefit 1
- Benefit 2
- Benefit 3

## Who It's For

Target audience and use cases.

## Capabilities

### Capability 1

Description and code examples.

### Capability 2

Description and code examples.

## Getting Started

### Quick Start

Minimal example to get started.

## Prerequisites

- Required version
- Dependencies
- Prior knowledge

## Typical Workflows

### Workflow 1

Step-by-step guide.

## Use Cases

### Use Case 1: Title

**Scenario:** Description
**Solution:** Code example

## Configuration

| Option | Description | Default |
|--------|-------------|---------|
| option1 | Description | value |

## Best Practices

1. Practice 1
2. Practice 2

## Limitations

- Limitation 1
- Limitation 2

## Related Documentation

- Links to related docs

## FAQ

### Question 1?

Answer.

## See Also

- Related links
```

## Developer Documentation (adw-docs)

### Guide Structure

Developer guides in `adw-docs/` follow this structure:

```markdown
# Guide Title

**Version:** 0.2.6
**Last Updated:** YYYY-MM-DD

## Overview

What this guide covers and why it's important.

### Key Points

Summary of main takeaways.

### Integration with ADW

How ADW uses this guide.

## Main Content

### Section 1

Content with code examples.

### Section 2

Content with examples.

## Commands

### Basic Commands

```bash
command --option
```

### Advanced Commands

```bash
command --advanced-option
```

## Configuration

Configuration details and `pyproject.toml` excerpts.

## Best Practices

1. Practice 1
2. Practice 2

## Troubleshooting

### Common Issue 1

**Problem:** Description
**Solution:** Fix

## Summary

**Key Requirements:**
1. Requirement 1
2. Requirement 2

**Quick Reference:**
```bash
# Common commands
```
```

### Development Plans

Development plans in `adw-docs/dev-plans/` use templates:

| Template | Purpose | Location |
|----------|---------|----------|
| `template-feature.md` | Feature specifications | `features/` |
| `template-maintenance.md` | Maintenance tasks | `maintenance/` |
| `template-epic.md` | Epic tracking | `epics/` |

### Architecture Decision Records (ADRs)

ADRs document significant architectural decisions:

```markdown
# ADR-XXX: Title

**Status:** Proposed | Accepted | Deprecated | Superseded
**Date:** YYYY-MM-DD
**Authors:** Name

## Context

Why this decision is needed.

## Decision

What was decided.

## Alternatives Considered

Other options and why they weren't chosen.

## Consequences

### Positive

- Benefit 1
- Benefit 2

### Negative

- Trade-off 1
- Trade-off 2

## Implementation

How to implement the decision.

## References

- Related documents
```

## Markdown Standards

### Line Length

**Maximum**: 80 characters for body text, no limit for URLs.

### Headings

Use ATX-style headings with proper hierarchy:

```markdown
# Document Title (H1 - only one per document)

## Major Section (H2)

### Subsection (H3)

#### Minor Section (H4 - use sparingly)
```

### Code Blocks

Always specify language for syntax highlighting:

````markdown
```python
import particula as par
```

```bash
pytest --cov=particula
```

```toml
[tool.ruff]
line-length = 80
```
````

### Tables

Use consistent table formatting:

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |
```

### Links

Use relative links for internal references:

```markdown
<!-- Good: Relative links -->
See [Code Style Guide](code_style.md) for details.
See [Architecture](architecture/architecture_guide.md).

<!-- Bad: Absolute paths -->
See [Code Style Guide](code_style.md).
```

### Admonitions (MkDocs)

Use Material for MkDocs admonitions:

```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.

!!! tip
    This is a tip.

!!! example
    This is an example.
```

## Documentation Workflows

### When to Update Documentation

| Change Type | Documentation to Update |
|-------------|-------------------------|
| New function/class | Docstrings |
| API change | Docstrings, README if public |
| New feature | `docs/Features/`, examples, docstrings |
| Bug fix | Release notes in `maintenance/` |
| New module | Architecture outline, docstrings |
| Design decision | ADR in `architecture/decisions/` |
| New pattern | Theory docs, architecture guide |
| Breaking change | Migration guide, README |

### Documentation Checklist

Before committing documentation changes:

- [ ] Docstrings follow Google-style format
- [ ] Type hints in signatures, not in docstrings
- [ ] Line lengths <= 80 characters
- [ ] All markdown links are valid (relative)
- [ ] Code examples are runnable
- [ ] Notebooks execute without errors
- [ ] Spelling and grammar checked
- [ ] Index files updated if new pages added

### Validation

The `docs-validator` subagent checks:

1. **Markdown links**: All internal links resolve
2. **Code blocks**: Proper language specification
3. **Formatting**: Consistent heading hierarchy
4. **Cross-references**: Valid between documents

## Quick Reference

### Documentation Locations

| What | Where |
|------|-------|
| Python docstrings | `particula/**/*.py` |
| Developer guides | `adw-docs/*.md` |
| Architecture | `adw-docs/architecture/` |
| ADRs | `adw-docs/architecture/decisions/` |
| Feature plans | `adw-docs/dev-plans/features/` |
| User examples | `docs/Examples/` |
| Feature docs | `docs/Features/` |
| Theory | `docs/Theory/` |
| Contributing | `docs/contribute/` |

### Key Commands

```bash
# Build MkDocs site
mkdocs build

# Serve locally
mkdocs serve

# Check docstrings (via linting)
ruff check particula/ --select=D

# Validate notebooks
.opencode/tool/run_notebook.py --validate
```

### Key Files

| File | Purpose |
|------|---------|
| `AGENTS.md` | Quick reference for ADW agents |
| `README.md` | Project overview |
| `mkdocs.yml` | MkDocs configuration |
| `adw-docs/README.md` | Developer docs index |
| `docs/index.md` | User docs home page |

## See Also

- **[Docstring Guide](docstring_guide.md)**: Detailed docstring format
- **[Code Style Guide](code_style.md)**: Coding conventions
- **[Architecture Reference](architecture_reference.md)**: Architecture docs
- **[Testing Guide](testing_guide.md)**: Test documentation
- **[Contributing Guide](../docs/contribute/CONTRIBUTING.md)**: Contribution workflow
