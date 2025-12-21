# Documentation Guide

**Version:** 0.2.6
**Last Updated:** 2025-11-30

## Overview

This guide documents documentation format and standards for the **particula** project. particula uses MkDocs with Material theme for building comprehensive documentation from Markdown files.

### Documentation Format

particula uses **Markdown (.md)** for all documentation.

### Documentation Tools

particula uses **MkDocs** with the Material theme for documentation generation.

- **MkDocs**: Static site generator optimized for project documentation
- **Material for MkDocs**: Modern, responsive theme with advanced features
- **mkdocstrings**: Automatic API documentation from Python docstrings
- **Build command**: `mkdocs build`
- **Serve command**: `mkdocs serve` (live preview at http://localhost:8000)
- **Deploy command**: Automated via GitHub Actions to GitHub Pages

### Documentation Configuration

Documentation is configured in `mkdocs.yml`:

```yaml
site_name: Particula
site_url: https://uncscode.github.io/particula
repo_url: https://github.com/uncscode/particula

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.tracking
    - search.suggest
    - content.code.copy
    
plugins:
  - search
  - mkdocstrings:  # Auto-generate API docs from docstrings
      handlers:
        python:
          docstring_style: google
  - mkdocs-jupyter:  # Include Jupyter notebooks
      execute: False
```

### Documentation Structure

```
docs/
├── index.md                # Landing page
├── Agent/                  # ADW agent documentation
│   ├── README.md          # Agent guide index
│   ├── testing_guide.md   # Test framework and conventions
│   ├── linting_guide.md   # Linting tools and standards
│   ├── code_style.md      # Code conventions
│   ├── docstring_guide.md # Documentation standards
│   └── ...                # Other agent guides
├── Examples/               # Tutorial notebooks and examples
│   ├── index.md
│   ├── Aerosol/
│   ├── Dynamics/
│   ├── Gas_Phase/
│   └── ...
├── Theory/                 # Theoretical background
│   ├── index.md
│   ├── Technical/
│   └── ...
├── contribute/             # Contribution guidelines
│   ├── CONTRIBUTING.md
│   ├── Code_Specifications/
│   └── Feature_Workflow/
└── images/                 # Images and assets
```

### Required Sections for Documentation

All documentation types have specific requirements:

#### Feature/Design Documents

All feature and design documents must include:

1. **Overview**: Brief summary of the feature/design
2. **Motivation**: Why this feature/design is needed
3. **Specification**: Detailed technical specification
   - API design
   - Algorithm description
   - Implementation approach
4. **Testing**: How the feature will be tested
5. **Examples**: Usage examples
6. **References**: Scientific papers or external resources (if applicable)

#### API Documentation

Generated automatically from docstrings using mkdocstrings:

1. **Module docstring**: Purpose and optional scientific citation
2. **Class docstrings**: Description, attributes
3. **Function docstrings**: Description, Args, Returns, Raises, Examples
4. **Type hints**: Must be present for API documentation generation

#### Tutorial Notebooks

Jupyter notebooks in `docs/Examples/`:

1. **Title and Description**: What the notebook demonstrates
2. **Setup**: Import statements and configuration
3. **Step-by-Step**: Clear progression of concepts
4. **Visualization**: Plots and results
5. **Summary**: Key takeaways
6. **References**: Related documentation or papers

### File Naming

**Pattern**: `snake_case.md` for Markdown files

**Examples:**
- `testing_guide.md` (guides)
- `code_style.md` (guides)
- `brownian_coagulation.md` (technical docs)
- `Quick_Start.ipynb` (notebooks - can use Title_Case)

**Consistency:**
- Use lowercase with underscores for guide files
- Use descriptive names that match content
- Group related files in subdirectories

### Asset Handling

**Images and Figures:**
- **Location**: Store in `docs/images/`
- **Format**: PNG, SVG (vector graphics preferred for diagrams)
- **Naming**: `descriptive_name.png` (snake_case)
- **Resolution**: High enough for clarity (typically 1200px wide for screenshots)

**Jupyter Notebooks:**
- **Location**: Store in appropriate `docs/Examples/` subdirectory
- **Execution**: Set `execute: False` in mkdocs.yml (don't run during build)
- **Outputs**: Include outputs in committed notebooks for preview

**Data Files:**
- Small example data: Include in repository
- Large datasets: Link to external sources (DOI, GitHub releases)

## Building Documentation Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install -e .[dev]
# or
uv pip install -e .[dev]
```

This installs:
- mkdocs
- mkdocs-material[imaging]
- mkdocs-jupyter
- mkdocstrings[python]
- mkdocs-gen-files
- mkdocs-literate-nav

### Build and Serve

```bash
# Build documentation
mkdocs build

# Serve with live reload (http://localhost:8000)
mkdocs serve

# Build for deployment
mkdocs build --clean
```

### Automatic API Documentation

API documentation is generated automatically from Python docstrings using mkdocstrings.

**Configuration in mkdocs.yml:**
```yaml
plugins:
  - mkdocstrings:
      handlers:
        python:
          paths: [particula]
          options:
            show_source: true
            docstring_style: google
            merge_init_into_class: true
            show_signature_annotations: true
```

**Auto-generation script:** `docs/.assets/mk_generator.py`

This script scans `particula/` and generates API documentation pages automatically.

## Writing Good Documentation

### Code Documentation (Docstrings)

- Use Google-style format (enforced by ruff)
- Include type hints in signatures (not in docstrings)
- Provide examples for non-trivial functions
- Include scientific citations in module docstrings
- Document units for physical quantities

See [Docstring Guide](docstring_guide.md) for details.

### Tutorial Documentation

- Start with motivation (why learn this?)
- Build progressively (simple to complex)
- Include complete, runnable examples
- Show visualizations of results
- Link to relevant API documentation
- Provide references for further reading

### Theory Documentation

- Explain concepts clearly for target audience
- Use equations (MathJax/LaTeX supported)
- Include diagrams and figures
- Reference original papers
- Provide implementation notes linking theory to code

### Contributing Documentation

- Update relevant docs when changing code
- Test documentation builds locally
- Check for broken links
- Ensure examples are current
- Follow existing structure and style

## Integration with ADW

ADW documentation commands use this guide for:
- Determining documentation format (Markdown)
- Knowing documentation build tools (MkDocs)
- Understanding required sections for different doc types
- Locating documentation files (docs/ directory)
- Building and validating documentation

## See Also

- **[Docstring Guide](docstring_guide.md)**: Code documentation style (Google-style)
- **[Code Style Guide](code_style.md)**: General coding standards
- **[Contributing Guide](../contribute/CONTRIBUTING.md)**: Contribution workflow
- **MkDocs Documentation**: https://www.mkdocs.org/
- **Material for MkDocs**: https://squidfunk.github.io/mkdocs-material/
