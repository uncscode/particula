# Developer Documentation

**Project:** ADW (Agent Developer Workflow)
**Version:** 2.1.0
**Last Updated:** 2026-06-05

Welcome to the ADW developer documentation hub. This section contains comprehensive guides, architecture documentation, and standards for developers working with or extending the ADW system.

## Quick Navigation

### Getting Started

- **[Setup Guide](setup_guide.md)** - Single-source install, wizard, and validation path

### Core Guides

Essential guides for ADW development:

- **[Code Style Guide](code_style.md)** - Coding standards and naming conventions
- **[Testing Guide](testing_guide.md)** - Testing framework, commands, and strategies
- **[Linting Guide](linting_guide.md)** - Code quality tools and linting standards
- **[Review Guide](review_guide.md)** - Code review criteria and process
- **[Backend Configuration Guide](backend_configuration.md)** - Single-repo vs fork/upstream routing, dual-scan (`ADW_TARGET_REPO=both`) with upstream-preferred dedupe, and the `--target` CLI flag

### Architecture

Understanding ADW's design and structure:

- **[Architecture Outline](architecture/architecture_outline.md)** - System components and module relationships
- **[Architecture Guide](architecture/architecture_guide.md)** - Design principles, patterns, and implementation examples
- **[Architecture Reference](architecture_reference.md)** - Quick reference for architectural conventions
- **[Branch Hierarchy](architecture/branch-hierarchy.md)** - Branch tiers, naming, migration, and troubleshooting
- **[Decision Records](architecture/decisions/README.md)** - Architectural Decision Records (ADRs)

Current architecture-reference highlights include the `adforge_core.tools`
metadata/catalog/policy foundation, workspace-aware resource loading, and the
`adforge_core.hooks` typed registry plus executor seam.

Track D3 locked in the Observer, Decorator, and Repository coverage—start with the [Design Patterns section](architecture/architecture_guide.md#design-patterns) for usage details.
For accepted decisions, read [ADR-009](architecture/decisions/ADR-009-observer-pattern-workflow-events.md), [ADR-010](architecture/decisions/ADR-010-decorator-pattern-platform-operations.md), [ADR-011](architecture/decisions/ADR-011-repository-pattern-state-management.md), and [ADR-012](architecture/decisions/ADR-012-branch-strategy.md).

Wrapper-boundary governance uses compatibility-window terminology: split/atomic wrappers
are the canonical active surface, while broad wrappers are compatibility-only unless
exception-approved with required metadata. See
[ADR-020](architecture/decisions/ADR-020-repository-wrapper-policy-boundaries.md),
[ADR-019](architecture/decisions/ADR-019-atomic-git-tool-wrapper-boundaries.md), and the
canonical retirement gates in
[AGENTS.md](../../AGENTS.md#compatibility-window-retirement-gates).

### Documentation Standards

Writing consistent, high-quality documentation:

- **[Documentation Guide](documentation_guide.md)** - Documentation structure and standards
- **[Docstring Guide](docstring_guide.md)** - Google-style docstring format
- **[Docstring Examples: Functions](docstring_function.md)** - Function documentation templates
- **[Docstring Examples: Classes](docstring_class.md)** - Class documentation templates
- **[Conditional Documentation](conditional_docs.md)** - Dynamic documentation strategies

### Process & Conventions

Development workflows and standards:

- **[Commit Conventions](commit_conventions.md)** - Semantic commit message format
- **[PR Conventions](pr_conventions.md)** - Pull request guidelines and templates
- **[Code Culture](code_culture.md)** - Team values and best practices

### Development Plans

Roadmap documents that coordinate epics, features, and maintenance priorities:

- **[Development Plans Overview](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/README.md)** - Directory structure, decision guide, and plan index
- **[Epic Template](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/template-epic.md)** - 15+ phase program blueprint
- **[Feature Template](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/template-feature.md)** - Strategic feature plan template
- **[Maintenance Template](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/template-maintenance.md)** - Evergreen maintenance plan template

#### Branch Naming (E11)

- **Canonical patterns** — See the
  [Branch Naming Convention table](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/epics/completed/E11-branch-based-development.md#82-branch-naming-convention)
  for full details. Summary:
  - `develop`: trunk for shared stabilization work (configurable via `ADW_DEV_BRANCH`).
  - `epic{NN}-{adw_id}-{slug}`: cross-cutting epic threads.
  - `feat{NN}-{adw_id}-{slug}`: standalone features built from develop.
  - `{type}-{issue}-adw-{id}-{slug}`: issue-linked branches (legacy/main-compatible).
- **Feature plan E11-F4 (Completed)** —
  [Worktree Base Branch & naming rollout](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/features/completed/E11-F4-worktree-base-branch.md)
  documents the completed implementation and compact table that mirrors the epic.

### Templates

Standardized templates for consistent documentation:

- **[Epic Template](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/template-epic.md)** - Multi-release plan template
- **[Feature Template](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/template-feature.md)** - Feature specification template
- **[Maintenance Template](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/template-maintenance.md)** - Maintenance task template
- **[ADR Template](architecture/decisions/template.md)** - Architecture decision record template

## Getting Started

### For New Developers

1. **Read the [Code Style Guide](code_style.md)** - Understand coding standards
2. **Review the [Testing Guide](testing_guide.md)** - Learn testing patterns and commands
3. **Explore the [Architecture Outline](architecture/architecture_outline.md)** - Understand system structure
4. **Study the [Code Culture](code_culture.md)** - Learn team values and practices

### For Contributors

1. **Follow [Commit Conventions](commit_conventions.md)** - Write semantic commit messages
2. **Use [PR Conventions](pr_conventions.md)** - Create well-structured pull requests
3. **Apply [Review Guide](review_guide.md)** - Conduct thorough code reviews
4. **Maintain [Documentation Standards](documentation_guide.md)** - Keep docs up-to-date

### For Architects

1. **Review [Architecture Guide](architecture/architecture_guide.md)** - Understand design patterns
2. **Study [Decision Records](architecture/decisions/README.md)** - Learn from past decisions
3. **Follow [Architecture Reference](architecture_reference.md)** - Apply architectural conventions
4. **Create ADRs** - Document significant architectural decisions

## Development Quick Reference

### Setup

```bash
# Create virtual environment
uv venv .venv

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Activate environment
source .venv/bin/activate  # Linux/macOS
```

### Testing

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest adw/tests/cli_test.py

# Run with coverage
python -m pytest --cov=adw --cov-report=term-missing
```

**Test Framework:** pytest
**Test Pattern:** `*_test.py`
**Coverage Target:** 80%+
**Test Location:** `adw/tests/`, `adw/*/tests/`

See [Testing Guide](testing_guide.md) for complete documentation.

### Linting

```bash
# Run linter
ruff check adw

# Run linter with auto-fix
ruff check adw --fix

# Run formatter
ruff format adw
```

**Linter:** ruff
**Line Length:** 100 characters
**Rules:** E, F, W, I, N (errors, warnings, imports, naming)

See [Linting Guide](linting_guide.md) for complete documentation.

## Common Development Tasks

### Adding a New Workflow

1. Create or copy a JSON definition in `.opencode/workflow/<name>.json`.
2. Describe phases, steps, and conditions using the
   [workflow JSON schema](workflow-json-schema.md) and
   [workflow engine guide](workflow-engine.md).
3. Run `adw workflow list` (or `adw workflow help <name>`) to confirm the
   definition was auto-registered, then exercise it with
   `adw workflow <name> <issue-number>`.
4. Add or update tests for the agents/workflow steps referenced by the new
   definition.
5. Update documentation (README, docs/index.md, and relevant guides) so users
   know how to invoke the new workflow.

See the [Architecture Outline](architecture/architecture_outline.md) and
[Architecture Guide](architecture/architecture_guide.md) for engine internals
and extension patterns.

### Adding a Slash Command

1. Create markdown file in `.opencode/command/command_name.md`
2. Update `adw/core/models.py` SlashCommand type
3. Add to SLASH_COMMAND_MODEL_MAP for model selection
4. Test command execution

### Fixing a Bug

1. Write failing test that reproduces the bug
2. Identify affected module(s) from [Architecture Outline](architecture/architecture_outline.md)
3. Implement fix following [Code Style Guide](code_style.md)
4. Verify test passes
5. Run full test suite
6. Update documentation if behavior changed

### Refactoring Code

1. Ensure existing tests pass
2. Make incremental changes
3. Run tests after each change
4. Follow patterns from [Architecture Guide](architecture/architecture_guide.md)
5. Update architecture docs if structure changed

## Documentation Types

| Type | Purpose | Location | Format |
|------|---------|----------|--------|
| **README** | Project overview | Root, module roots | Markdown |
| **Architecture Docs** | System design | `.opencode/guides/architecture/` | Markdown |
| **API Docs** | Module/function reference | Generated from docstrings | Google-style |
| **Decision Records** | Design decisions | `.opencode/guides/architecture/decisions/` | ADR format |
| **Guides** | Development guides | `.opencode/guides/` | Markdown |
| **Examples** | Practical tutorials | `docs/Examples/` | Markdown + Code |
| **Theory** | Concepts and principles | `docs/Theory/` | Markdown |

## Code Quality Standards

### Naming Conventions

- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore`

### Documentation Requirements

- **Type Hints**: Required for public APIs
- **Docstrings**: Required for public modules, classes, functions
- **Imports**: Organized (stdlib, third-party, local)

### Linting Rules

- **E/F/W**: PEP 8 errors, warnings, and style issues
- **I**: Import sorting and organization
- **N**: Naming conventions
- **Line Length**: 100 characters maximum
- **Auto-fix**: Enabled for most rules

## Architecture Principles

ADW is built on four core architectural principles:

1. **Isolated Execution** - Each workflow runs in an isolated git worktree
2. **State-Driven Architecture** - Workflow state persists to JSON files
3. **Declarative Phase Orchestration** - `.opencode/workflow/*.json` definitions
   describe each phase and are executed via `adw workflow <name>`
4. **AI-Augmented Automation** - AI agents (via OpenCode) handle complex tasks

See [Architecture Guide](architecture/architecture_guide.md) for detailed explanations.

## Contributing

### Pull Request Process

1. **Create Feature Branch**: `git checkout -b feature-description`
2. **Make Changes**: Follow [Code Style Guide](code_style.md)
3. **Run Tests**: Ensure all tests pass (`python -m pytest`)
4. **Run Linter**: Ensure no linting errors (`ruff check adw`)
5. **Commit**: Follow [Commit Conventions](commit_conventions.md)
6. **Push**: `git push origin feature-description`
7. **Create PR**: Follow [PR Conventions](pr_conventions.md)
8. **Code Review**: Address review comments using [Review Guide](review_guide.md)
9. **Merge**: After approval and passing CI

### Commit Message Format

```text
<type>: <description>

<optional body>

<optional footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

See [Commit Conventions](commit_conventions.md) for complete specification.

## Additional Resources

### External Documentation

- **[OpenCode CLI](https://opencode.ai/docs)** - OpenCode documentation
- **[GitHub API](https://docs.github.com/en/rest)** - GitHub REST API reference
- **[Pydantic](https://docs.pydantic.dev/)** - Data validation library
- **[Click](https://click.palletsprojects.com/)** - CLI framework
- **[pytest](https://docs.pytest.org/)** - Testing framework
- **[ruff](https://docs.astral.sh/ruff/)** - Linting and formatting tool

### Internal Documentation

- **[Main README](https://github.com/Gorkowski/Agent/blob/main/README.md)** - Project overview and user guide
- **[Workflow Engine Guide](workflow-engine.md)** - Deep reference for the declarative workflow runtime
- **[Examples](https://github.com/Gorkowski/Agent/blob/main/docs/Examples/index.md)** - Practical tutorials and examples
- **[Theory](https://github.com/Gorkowski/Agent/blob/main/docs/Theory/index.md)** - Concepts and principles

## Support

- **Issues**: [GitHub Issues](https://github.com/Gorkowski/Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Gorkowski/Agent/discussions)
- **Documentation**: This directory (`adw-docs/`)

---

**Last Updated:** 2026-01-02
**Maintainers:** ADW Development Team
