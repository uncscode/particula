---
description: "Install & Setup"
---

# Install & Setup

## Overview
Sets up the repo development environment using uv virtual environment.

## Steps

1. **Review Dependencies**
   - Read `pyproject.toml` to understand project dependencies
   - Ensure torch is removed from dependencies (no longer used)

2. **Create Virtual Environment**
   - Run `uv venv .venv` to create a virtual environment

3. **Install Package**
   - Run `uv pip install -e ".[dev]"` to install repo in editable mode with dev dependencies
   - This installs all required packages

4. **Configure Repository Specifics**
   - Ask the user about their codebase setup with questions about:
     - Primary programming language (Python, C++, JavaScript, etc.)
     - Language version
     - Build system (setuptools, cmake, npm, etc.)
     - Package manager (pip, npm, cargo, etc.)
     - Test framework (pytest, jest, googletest, etc.)
     - Test locations and naming conventions
   - Auto-fill values from codebase analysis:
     - Language extension (e.g., "py" for Python, "js" for JavaScript)
     - Test file patterns from test framework config (e.g., pytest.ini, jest.config.js)
     - Test command from package manager and test framework
     - Test timeout (reasonable default based on codebase size)
     - Example test names and purposes from actual test files
     - Documentation format (detect from existing docs: Markdown, RST, etc.)
     - Documentation tooling (detect: Sphinx, MkDocs, Docusaurus, or None)
     - Screenshot/asset directories (detect or use sensible defaults)
   - Ask the user for optional documentation preferences:
     - Documentation tool (if not auto-detected: Sphinx, MkDocs, Docusaurus, None)
     - Documentation build/serve commands (if applicable)
     - Required documentation sections for features (default: Overview, Implementation, Testing)
     - Asset naming conventions (if different from detected pattern)

5. **Generate Architecture and Development Documentation**
   - Ask the user if they want to generate comprehensive development documentation (default: yes)
   - If yes, analyze the codebase and ask clarifying questions about:
     - System purpose and high-level overview
     - Core modules/components (3-5 main components)
     - For each component: purpose, location, key interfaces
     - Key design principles (3-4 principles)
     - Main data flows through the system
     - Technology stack and key dependencies
     - Common design patterns used
     - Testing strategy and coverage goals
     - Linting tools and code quality standards
   - After gathering information:
     - Analyze the codebase structure using file exploration tools
     - Read key entry point files (CLI, main modules, etc.)
     - Read core modules to understand component relationships
     - Read pyproject.toml or equivalent for test/lint configuration
     - Generate `docs/Agent/architecture/architecture_outline.md` with:
       - Component overview with responsibilities
       - Module structure diagram
       - Technology stack table
       - High-level data flow
       - Quick start guide for developers
     - Generate `docs/Agent/architecture/architecture_guide.md` with:
       - Detailed component descriptions
       - Design principles with examples
       - Common patterns with code examples
       - Data flow scenarios
       - Extension points for new features
       - Architecture evolution and future directions
     - Update `docs/Agent/testing_guide.md` with actual values from pyproject.toml:
       - Test framework and version
       - Test command and options
       - Test file patterns
       - Coverage commands and thresholds
       - Test execution examples from the actual codebase
     - Update `docs/Agent/linting_guide.md` with actual values from pyproject.toml:
       - Linter tools and versions
       - Linting commands
       - Line length and formatting rules
       - Configuration file locations
       - Auto-fix capabilities
     - Generate `docs/Agent/README.md` with:
       - Project overview and quick links
       - Architecture summary
       - Development guide quick reference
       - Testing and linting command summary
       - Links to all documentation guides

6. **Validate Installation**
   - Run /test and /lint to validate repo install.

## Report

After installation, inform the user:
- Virtual environment created at `.venv`
- Development documentation generated:
  - `docs/Agent/architecture/architecture_outline.md` - High-level architecture overview
  - `docs/Agent/architecture/architecture_guide.md` - Detailed architecture guide with patterns
  - `docs/Agent/testing_guide.md` - Updated with actual test configuration and commands
  - `docs/Agent/linting_guide.md` - Updated with actual linting tools and standards
  - `docs/Agent/README.md` - Documentation hub with quick reference links
- All tests passing successfully (286/286 tests)
- All linting checks passed
- To activate the environment: `source .venv/bin/activate`
- Next steps: Review generated documentation and customize as needed

