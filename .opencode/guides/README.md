# AI Developer Documentation

**Project:** adw
**Version:** 2.1.0
**Last Updated:** 2026-05-07

> **Directory purpose:** `.opencode/guides/` is the **agent-facing** documentation directory — it contains
> internal guides, architecture references, and development standards consumed by ADW agents and
> contributors working on the system itself. Files here are not part of the built documentation
> site; they are readable directly on GitHub.
>
> **User-facing documentation** (examples, tutorials, theory, and feature guides intended for
> end-users of ADW) lives in **`docs/`** at the repository root. `docs/` is the MkDocs `docs_dir`
> (configured in `mkdocs.yml`) — only files there are included in the built documentation site.
>
> If you are adding user-facing content (e.g. `docs/Examples/`, `docs/Theory/`,
> `docs/Features/`), place it under `docs/`, not here.

Welcome to the ADW (Agent Developer Workflow) documentation hub. This directory contains comprehensive documentation for developers working with or extending the ADW system.

CLI help and references use the configured documentation directory (`DOCS_DIR`, canonical default `.opencode/guides`).
Links in this guide assume the canonical layout.

## Quick Links

### 📚 Getting Started
- **[Installation Guide](../../README.md#quick-start)** - Set up ADW in your environment
- **[Quick Start](../../README.md#quick-start)** - Run your first workflow
- **[Architecture Outline](architecture/architecture_outline.md)** - High-level system overview

### 🔧 Setup & Configuration
- **[Setup Guide](setup_guide.md)** - Complete installation and configuration walkthrough
- **[Troubleshooting Setup](troubleshooting_setup.md)** - Common setup issues and solutions
- **[Backend Configuration](backend_configuration.md)** - Advanced platform configuration

### 🏗️ Architecture
- **[Architecture Outline](architecture/architecture_outline.md)** - Components, modules, and relationships
- **[Architecture Guide](architecture/architecture_guide.md)** - Design principles, patterns, and examples
- **[Decision Records](architecture/decisions/)** - Architectural decision history (ADRs)

### 💻 Development
- **[Code Style Guide](code_style.md)** - Coding standards and conventions
- **[Testing Guide](testing_guide.md)** - Testing framework, commands, and strategies
- **[Linting Guide](linting_guide.md)** - Code quality tools and standards
- **[Review Guide](review_guide.md)** - Code review criteria and process

### 📖 Additional Guides
> **Archived Docs:** Legacy migration guides were moved to `.trash/adw-docs/` for reference.
- **CLI Migration: gh → adw** (archived; see `.trash/adw-docs/cli-migration-guide.md`) - Map common gh/git commands to adw platform/git
- **[Documentation Guide](documentation_guide.md)** - Documentation standards
- **[Docstring Guide](docstring_guide.md)** - Docstring format and examples
- **[Commit Conventions](commit_conventions.md)** - Commit message format
- **[PR Conventions](pr_conventions.md)** - Pull request format and process
- **[Code Culture](code_culture.md)** - Team culture and best practices

### 🛠️ Development Plans
- `.opencode/plans/` - Canonical plan records, sections, templates, and generated plan views
- `.opencode/plans/templates/epic/` - Epic plan section templates
- `.opencode/plans/templates/feature/` - Feature plan section templates
- `.opencode/plans/templates/maintenance/` - Maintenance plan section templates
- `.opencode/plans/templates/{feature,maintenance}/issue_template.json` - Runtime-loaded issue body templates
  for plan-driven issue generation with path confinement, deterministic fallback,
  and file-size guardrails
- `adw/plans/issue_contracts.py` - Shared fallback section/review contracts used when runtime
  template metadata is unavailable
- `.opencode/plans/generated/` - Generated plan indexes and rendered markdown output

`adw spec batch init --plan-type <type>` loads runtime issue-template metadata,
persisting `batch_meta.section_names` and `batch_meta.role_index` for runtime
CLI section validation and role-aware review flows.

## Architecture Summary

ADW is an AI-powered development workflow automation system built on four core principles:

1. **Isolated Execution**: Each workflow runs in an isolated git worktree, enabling safe parallel execution
2. **State-Driven Architecture**: Workflow state persists to JSON files for resumption and debugging
3. **Phase-Based Orchestration**: Workflows decompose into discrete phases (plan → build → test → review → document → ship)
4. **AI-Augmented Automation**: AI agents (via OpenCode) handle complex tasks requiring intelligence and reasoning

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **CLI & Entry Points** | Command-line interface for workflow execution | `adw/cli.py` |
| **Workflow Orchestration** | Coordinates multi-phase workflows | `adw/workflows/` |
| **Core Models & Types** | Data structures and type definitions | `adw/core/` |
| **GitHub Integration** | GitHub API operations | `adw/github/` |
| **Git Operations** | Git and worktree management | `adw/git/` |
| **State Management** | Persistent workflow state | `adw/state/` |
| **Agent Interface** | OpenCode agent execution | `adw/core/agent.py` |
| **Utilities** | Logging, metrics, helpers | `adw/utils/` |
| **Trigger Scheduler** | Cron-only poller that replaces the webhook stack | `adw/triggers/trigger_cron_unified.py` |

For detailed architecture documentation, see the [Architecture Outline](architecture/architecture_outline.md) and [Architecture Guide](architecture/architecture_guide.md).

## Development Quick Reference

### Setup and Installation

```bash
# Create virtual environment
uv venv .venv

# Install in editable mode with dev dependencies (includes ast-grep CLI)
uv pip install -e ".[dev]"

# Activate environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Trigger Operations
```bash
uv run adw launch        # Start the cron poller (webhook server removed)
uv run adw stop          # Stop cron + dispatcher processes
```

`adw launch` always runs the cron trigger now; the `--trigger-type`, `--host`,
`--port`, and `INTERNAL_TRIGGER_TOKEN` options were removed along with the
webhook server.

### Setup Command Scaffold
```bash
adw setup             # Launch the environment setup wizard (default)
adw setup env         # Launch the environment setup wizard (alias)
adw setup env --with-templates  # Force template bootstrap (no prompt, uses defaults)
adw setup env --skip-templates  # Skip template bootstrap prompt
adw setup template init [--yes]  # Init template manifest (.opencode/.adw-template-manifest.yaml)
                                 # Loads placeholder defaults; template/live modes; --yes skips
                                 # prompts and overwrites
adw setup template apply [--dry-run|--check|--yes]
                                 # Template → live substitution (manifest mode='template')
                                 # Copies missing files, substitutes placeholders; check mode is read-only
# `adw setup template extract` has been removed; the documentation directory (`DOCS_DIR`,
# default `.opencode/guides`) is the source of truth and placeholder tokenization now lives in docs
# stubs.
adw setup validate    # Run full setup validation (returns non-zero on failures)
adw setup check       # Run fast preflight checks
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest adw/tests/cli_test.py

# Run with coverage
python -m pytest --cov=adw --cov-report=term-missing

# Run specific test
python -m pytest adw/tests/cli_test.py::TestCompleteCommand::test_complete_success -v
```

**Test Framework:** pytest
**Test Pattern:** `*_test.py`
**Coverage Target:** 80%+
**Test Location:** `adw/tests/`, `adw/*/tests/`

See [Testing Guide](testing_guide.md) for detailed information.

### Linting and Formatting

```bash
# Run linter
ruff check adw

# Run linter with auto-fix
ruff check adw --fix

# Run formatter
ruff format adw

# Check syntax
find adw -name "*.py" -exec python -m py_compile {} \;
```

**Linter:** ruff
**Line Length:** 100 characters
**Rules:** E, F, W, I, N (errors, warnings, imports, naming)

See [Linting Guide](linting_guide.md) for detailed information.

### Common Development Tasks

**Add a New Workflow:**
1. Create workflow function in `adw/workflows/workflow_name.py`
2. Follow pattern: `(ctx: WorkflowContext) -> WorkflowResult`
3. Register CLI command in `adw/cli.py`
4. Add tests in `adw/workflows/tests/`
5. Update documentation

**Add a New Slash Command:**
1. Create markdown file in `.opencode/command/command_name.md`
2. Update `adw/core/models.py` SlashCommand type
3. Add to SLASH_COMMAND_MODEL_MAP for model selection
4. Test command execution

**Fix a Bug:**
1. Write failing test that reproduces the bug
2. Identify affected module(s) from architecture outline
3. Implement fix
4. Verify test passes
5. Run full test suite
6. Update documentation if behavior changed

**Refactor Code:**
1. Ensure existing tests pass
2. Make incremental changes
3. Run tests after each change
4. Follow patterns from [Architecture Guide](architecture/architecture_guide.md)
5. Update architecture docs if structure changed

## Documentation Standards

### Documentation Types

| Type | Purpose | Location | Format |
|------|---------|----------|--------|
| **README** | Project overview | Root, module roots | Markdown |
| **Architecture Docs** | System design | `.opencode/guides/architecture/` | Markdown |
| **API Docs** | Module/function reference | Generated from docstrings | Google-style |
| **Decision Records** | Design decisions | `.opencode/guides/architecture/decisions/` | ADR format |
| **Guides** | Development guides | `.opencode/guides/` | Markdown |
| **Development Plans** | Roadmaps, epics, and maintenance plans | `.opencode/plans/` | JSON + Markdown |

### Docstring Format

ADW uses **Google-style docstrings** for Python code:

```python
def workflow_function(ctx: WorkflowContext) -> WorkflowResult:
    """Execute workflow with validation.

    This function orchestrates a multi-phase workflow including
    planning, implementation, testing, and shipping.

    Args:
        ctx: Workflow context containing issue number, ADW ID, and model set

    Returns:
        WorkflowResult with success status, errors, and metadata

    Raises:
        WorkflowError: If critical failure occurs during execution

    Example:
        >>> ctx = WorkflowContext(issue_number=123, adw_id="abc12345")
        >>> result = workflow_function(ctx)
        >>> print(result.success)
        True
    """
```

See [Docstring Guide](docstring_guide.md) for complete format specification.

## Testing Strategy

ADW uses a comprehensive testing strategy:

### Test Categories

1. **Unit Tests**: Test individual functions and classes in isolation
   - Location: Co-located with source code in `*/tests/` directories
   - Pattern: `*_test.py`
   - Focus: Single function/class behavior

2. **Integration Tests**: Test component interactions
   - Location: `adw/tests/integration_workflow_test.py`
   - Focus: Workflow orchestration, state management, GitHub integration

3. **CLI Tests**: Test command-line interface
   - Location: `adw/tests/cli_test.py`
   - Focus: CLI commands, argument parsing, user interaction

### Test Execution

```bash
# All tests
python -m pytest                    # 286 tests

# Specific categories
python -m pytest adw/tests/cli_test.py                       # CLI tests
python -m pytest adw/tests/integration_workflow_test.py      # Integration tests
python -m pytest adw/workflows/tests/                        # Workflow tests

# With coverage
python -m pytest --cov=adw --cov-report=html
# View: coverage_html/index.html
```

### Writing Tests

Follow these patterns:

```python
# adw/workflows/tests/workflow_name_test.py
def test_workflow_success():
    """Test successful workflow execution."""
    ctx = WorkflowContext(issue_number=123, adw_id="test123")
    result = run_workflow(ctx)
    assert result.success
    assert len(result.errors) == 0

def test_workflow_failure():
    """Test workflow handles errors gracefully."""
    ctx = WorkflowContext(issue_number=999, adw_id="test999")
    result = run_workflow(ctx)
    assert not result.success
    assert "error message" in result.errors[0]
```

See [Testing Guide](testing_guide.md) for comprehensive testing documentation.

## Code Quality Standards

### Linting Rules

- **E/F/W**: PEP 8 errors, warnings, and style issues
- **I**: Import sorting and organization
- **N**: Naming conventions
- **Line Length**: 100 characters maximum
- **Auto-fix**: Enabled for most rules

### Code Style

- **Naming**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private: `_leading_underscore`
- **Type Hints**: Required for public APIs
- **Docstrings**: Required for public modules, classes, functions
- **Imports**: Organized (stdlib, third-party, local)

See [Code Style Guide](code_style.md) for complete style specification.

## Contributing

### Pull Request Process

1. **Create Feature Branch**: `git checkout -b feature-description`
2. **Make Changes**: Follow code style and testing guidelines
3. **Run Tests**: Ensure all tests pass (`python -m pytest`)
4. **Run Linter**: Ensure no linting errors (`ruff check adw`)
5. **Commit**: Follow [commit conventions](commit_conventions.md)
6. **Push**: `git push origin feature-description`
7. **Create PR**: Follow [PR conventions](pr_conventions.md)
8. **Code Review**: Address review comments
9. **Merge**: After approval and passing CI

### Commit Message Format

```
<type>: <description>

<optional body>

<optional footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Example**:
```
feat: add security scan workflow

- Implement run_security_workflow function
- Add /security_scan slash command
- Register security command in CLI
- Add integration tests

Closes #456
```

See [Commit Conventions](commit_conventions.md) for detailed format specification.

## Architecture Review

For changes affecting multiple modules or introducing new patterns, request an architecture review:

```bash
/architecture_review
```

This triggers a comprehensive review of:
- Architectural impact
- Design pattern compliance
- Component relationships
- Extension points
- Performance implications
- Security considerations

## Additional Resources

### External Documentation
- **[OpenCode CLI](https://opencode.ai/docs)** - OpenCode documentation
- **[GitHub API](https://docs.github.com/en/rest)** - GitHub REST API reference
- **[Pydantic](https://docs.pydantic.dev/)** - Data validation library
- **[Click](https://click.palletsprojects.com/)** - CLI framework
- **[pytest](https://docs.pytest.org/)** - Testing framework
- **[ruff](https://docs.astral.sh/ruff/)** - Linting and formatting tool

### Internal Documentation
- **[Main README](../README.md)** - Project overview and user guide
- **[Architecture Decisions](architecture/decisions/)** - ADR repository
- **Development Plans** - Canonical plan artifacts under `.opencode/plans/`

## Support and Contact

- **Issues**: [GitHub Issues](https://github.com/Gorkowski/Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Gorkowski/Agent/discussions)
- **Documentation**: This directory (`.opencode/guides/`)

## Document Index

### Architecture Documentation
- [Architecture Outline](architecture/architecture_outline.md)
- [Architecture Guide](architecture/architecture_guide.md)
- [Decision Records](architecture/decisions/)

### Development Guides
- [Code Style Guide](code_style.md)
- [Testing Guide](testing_guide.md)
- [Linting Guide](linting_guide.md)
- [Review Guide](review_guide.md)
- [Documentation Guide](documentation_guide.md)
- [Docstring Guide](docstring_guide.md)

### Process Documentation
- [Commit Conventions](commit_conventions.md)
- [PR Conventions](pr_conventions.md)
- [Code Culture](code_culture.md)

### Configuration
- [Agent Documentation](agents/)

---

**Last Updated:** 2025-11-13
**Maintainers:** ADW Development Team
