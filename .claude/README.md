# Claude Code Setup for Particula

This directory contains configuration and documentation for using Claude Code with the Particula project.

## Files in this directory

- **CLAUDE.md** - Comprehensive codebase architecture guide that Claude uses to understand the project structure, patterns, and conventions
- **ARCHITECTURE_SUMMARY.txt** - High-level architecture summary and key patterns
- **DOCUMENTATION_INDEX.md** - Index of all documentation files and their purposes
- **README_ARCHITECTURE_DOCS.md** - Guide to the architecture documentation structure
- **TESTING_AND_LINTING.md** - Complete guide for pytest testing and ruff linting/formatting
- **README.md** - This file, containing setup instructions

## Setting up Claude Code

### 1. Install Claude Code

If you haven't already, install Claude Code:
```bash
npm install -g @anthropic-ai/claude-code
```

Or follow the installation instructions at: https://docs.claude.com/claude-code

### 2. Configure the project

Claude Code will automatically detect the `CLAUDE.md` file in this directory and use it to understand the project structure.

The `CLAUDE.md` file is referenced in your Claude Code configuration and provides:
- High-level architecture overview
- Module and package structure
- Design patterns used throughout the codebase
- Testing patterns and conventions
- Development workflow guidelines
- Key utilities and patterns

### 3. Usage

When you start Claude Code in this project directory, it will:
1. Read the `CLAUDE.md` file to understand the codebase
2. Use this context to provide more accurate suggestions and help
3. Follow the project's patterns and conventions

### 4. Updating the Architecture Guide

If the project structure changes significantly, update the `CLAUDE.md` file to reflect:
- New modules or packages
- Changed design patterns
- Updated testing strategies
- New utilities or conventions

## Tips for working with Claude Code

1. **Reference the architecture**: The CLAUDE.md file is comprehensive - Claude will use it to understand your code
2. **Follow patterns**: Claude will suggest code that follows the patterns documented in CLAUDE.md
3. **Ask about structure**: You can ask Claude about the codebase architecture and it will reference CLAUDE.md
4. **Keep it updated**: As the project evolves, update CLAUDE.md so Claude stays in sync
5. **Testing and linting**: See TESTING_AND_LINTING.md for complete pytest and ruff usage

## Development Workflow

### Quick Start
```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Make your changes

# 4. Run tests and linting
uv run ruff format . && uv run ruff check --fix . && uv run pytest

# 5. Commit changes
git add .
git commit -m "Your commit message"
```

See **TESTING_AND_LINTING.md** for detailed instructions on:
- Running pytest tests with various options
- Using ruff for linting and formatting
- Test writing conventions and best practices
- Code style rules and configuration
- CI/CD integration

## Documentation Build Setup

The project uses MkDocs with several plugins for documentation:

### Required plugins
- `mkdocs-material` - Material theme
- `mkdocs-jupyter` - Jupyter notebook support
- `mkdocstrings[python]` - Python API documentation
- `mkdocs-gen-files` - Dynamic file generation
- `mkdocs-literate-nav` - Navigation from SUMMARY.md

### API Documentation Generator
- Located in: `docs/.assets/mk_generator.py`
- Automatically generates API documentation from Python source files
- Creates organized navigation with clean module names
- Shows parent folder context (e.g., `properties/activity_module`)

### Building docs locally
```bash
# Install dependencies
uv sync

# Live preview
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

## Project Structure Reference

See `CLAUDE.md` for the complete project structure, but here are the key directories:

```
particula/
├── .claude/              # Claude Code configuration (this directory)
├── particula/            # Main package source code
│   ├── activity/         # Activity coefficient models
│   ├── dynamics/         # Process implementations (coagulation, condensation)
│   ├── gas/              # Gas phase modeling
│   ├── particles/        # Particle phase modeling
│   └── util/             # Utility functions and constants
├── docs/                 # Documentation source
│   ├── .assets/          # Documentation build scripts
│   ├── API/              # Generated API docs (do not edit manually)
│   ├── Examples/         # Example notebooks and tutorials
│   └── Theory/           # Theoretical documentation
└── .github/workflows/    # CI/CD pipelines
```

## Getting Help

- For Claude Code documentation: https://docs.claude.com/claude-code
- For project-specific questions: See CLAUDE.md in this directory
- For Particula documentation: https://uncscode.github.io/particula