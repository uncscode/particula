# Documentation Guide

**Version:** 2.1.0
**Last Updated:** 2026-05-12

## Overview

This guide documents documentation format and standards for the adw project.

### Documentation Format

adw uses **Markdown (.md)** for all documentation.

### Documentation Tools

- **MkDocs**: Documentation site generator with Material theme
- **mkdocstrings**: Auto-generates API reference from Python docstrings
- **mkdocs-gen-files**: Generates API stub pages via `docs/.assets/api_markdown_generator.py`

#### Auto-Generated API Documentation

API reference documentation under `docs/API/` is **automatically generated** from Python docstrings:

- **Generator Script**: `docs/.assets/api_markdown_generator.py`
- **Trigger**: Runs automatically during `mkdocs build` or `mkdocs serve`
- **Output**: Creates `docs/API/` structure mirroring the `adw/` package
- **Source**: Extracts docstrings using mkdocstrings

**To regenerate API docs manually:**

```bash
uv run ./docs/.assets/api_markdown_generator.py
```

**Important**: Do NOT manually edit files in `docs/API/` - they are overwritten on each build. To update API documentation, modify the docstrings in the source code.

### File Naming

- Use **kebab-case** for file names (lowercase words separated by hyphens)
- Use descriptive names reflecting content

**Examples:**
- `feature-authentication.md`
- `architecture-overview.md`

### Asset Handling

- **Directory**: `docs/.assets/`
- **Purpose**: Scripts, images, and other documentation assets
- **Key Files**:
  - `api_markdown_generator.py`: Auto-generates API reference
  - Other build scripts as needed

## Documentation Categories

### Docstring Documentation (Source Code)

Remember the docstrings in the source code are the primary documentation for functions, classes, and modules. So do not duplicate information in separate files unless necessary. These docstrings are used to auto-generate the API reference in `docs/API/`.

The following sections describe when to create or update different types of documentation files, beyond just docstrings.

### Developer Guides (`.opencode/guides/`)

**Purpose**: Agent guides, standards, and conventions for ADW development

**When to add/update**:
- Adding new coding standards or conventions
- Updating project-specific practices
- Architecture decision records (use the ADR directory) at
  `.opencode/guides/architecture/decisions/`

> **Development Plans**: Epics, features, and maintenance priorities now live
> under `.opencode/plans/`. Use that hierarchy (not ADRs) for
> roadmap work that produces executable issues.

### Theory Documentation (`docs/Theory/`)

**Purpose**: High-level concepts, architectural principles, and theoretical foundations

**When to write new Theory files**:
- Introducing a fundamental architectural concept (e.g., isolated execution, state-driven design)
- Explaining a complex system behavior or design pattern
- Documenting "why" decisions were made at a conceptual level
- Providing language-agnostic explanations of ADW's approach

**When to update Theory files**:
- Core architectural principles change
- New theoretical foundations are established
- Design patterns evolve or new ones are introduced
- Comparison with alternatives needs updating

**Characteristics of Theory documentation**:
- Explains "why" and "how" at a conceptual level
- Uses Mermaid diagrams and visualizations
- Links to Examples for practical application
- Focuses on principles over specifics

**Examples**: `isolated-execution.md`, `state-driven-architecture.md`, `workflow-lifecycle.md`

### Examples Documentation (`docs/Examples/`)

**Purpose**: Practical tutorials, code examples, and hands-on guides

**When to write new Examples files**:
- Demonstrating how to use multiple ADW features, beyond what docstrings cover.
- Providing step-by-step tutorials for common tasks
- Showing integration patterns with external systems
- Creating code snippets for reference

**When to update Examples files**:
- APIs or CLI commands change
- New features are added that affect workflows
- Better practices emerge
- Examples become outdated or don't work

**Characteristics of Examples documentation**:
- Use case focused and actionable
- Includes working code examples
- Step-by-step instructions and walkthroughs
- Shows expected outputs
- Links to Theory for conceptual understanding

**Examples**: `workflows/basic-workflow.md`, `cli/branch-sync.md`,
`setup/custom-slash-command.md`

### API Documentation (`docs/API/`)

**Purpose**: Auto-generated API reference from source code docstrings

**This is auto-generated** - do NOT manually edit files in `docs/API/`.

**To update API documentation**:
```bash
uv run ./docs/.assets/api_markdown_generator.py
```

### Architecture Decisions (`.opencode/guides/architecture/decisions/`)

**Purpose**: Records of significant architectural decisions

**When to add decision records**:
- Making a major architectural change
- Choosing between multiple design alternatives
- Deprecating or replacing a core component
- Decisions with long-term impact on the system

**When NOT to add or modify**: Use `.opencode/plans/` for plan-level updates instead of overloading ADRs.

## Integration with ADW

ADW documentation commands use this guide for:
- Determining documentation format
- Knowing required sections
- Understanding file naming conventions

## Documentation Workflow

### For New Features

1. **Write code with docstrings** (see `docstring_guide.md`)
2. **Add Examples if needed**: Create tutorials in `docs/Examples/`
3. **Add Theory if needed**: Document concepts in `docs/Theory/`
4. **Update index files**: Add links to new docs in relevant `index.md` files

### For Major Architectural Changes

1. **Create decision record**: Add to `adw-docs/architecture/decisions/`
2. **Update Theory docs**: Modify relevant conceptual documentation
3. **Refresh general guidance**: Update high-level docs (README.md,
   docs/index.md, `.opencode/guides/index.md`, etc.) so they describe
   `.opencode/workflow/*.json` definitions executed via `adw workflow <name>`
   and reference the workflow registry/executor pipeline documented in
   `.opencode/guides/workflow-engine.md` and
   `.opencode/guides/workflow-json-schema.md`.
4. **Update Examples**: Ensure tutorials and JSON snippets reflect the new
   workflow definitions and CLI patterns

### For Routine Updates

1. **Update docstrings**: Modify source code comments

### Documentation Validation

- Run `mkdocs build --strict` after documentation updates to catch broken links,
  missing pages, and build warnings.
- When updating index pages, confirm every linked page exists and renders in
  the strict MkDocs build.

### When Docstrings Are Sufficient

**No separate documentation needed for**:
- New tests (docstrings explain test purpose)
- Small utility functions (docstrings cover usage)
- Simple bug fixes (commit message is enough)
- Minor refactoring (code speaks for itself)
- When new modules are added with clear docstrings
- Internal implementation details (docstrings sufficient)

**Separate documentation IS needed for**:
- New user-facing features
- New complex workflows or patterns
- Architectural changes
- Integration guides

## Documentation Standards

### Markdown Formatting

- Use GitHub-flavored Markdown
- Include table of contents for long documents
- Use code blocks with language hints (```python, ```bash)
- Link to related documentation extensively

### File Organization

- Keep files focused on single topics
- Use descriptive file names (kebab-case)
- Group related files in subdirectories
- Maintain index files for navigation

### Content Quality

- Write clear, concise prose
- Use examples liberally
- Include diagrams where helpful
- Link to API docs for implementation details
- Link to Theory docs for conceptual understanding
- Link to Examples docs for practical guidance

### Workflow Terminology Requirements

- Describe workflows as declarative JSON definitions stored under
  `.opencode/workflow/*.json` and executed through `adw workflow <name>`.
- When documentation discusses orchestration behavior, link to the
  [Architecture Outline](architecture/architecture_outline.md),
  [Architecture Guide](architecture/architecture_guide.md), and
  [Workflow Engine Guide](workflow-engine.md) for deeper coverage.
- Frame every entry point and example using the workflow engine terminology
  above; avoid resurrecting deprecated Python orchestration helpers in new or
  updated documentation.

### Wrapper Policy Terminology Requirements

- Use **compatibility-window** language consistently with `README.md` and
  `AGENTS.md`: split/atomic wrappers are canonical for new and updated
  workflows/docs.
- Describe broad wrappers as **compatibility-only** (or historical context),
  not preferred active integration paths.
- When documenting retained broad-wrapper usage, include exception-governance
  framing and required metadata (`owner`, `compatibility_window`, and
  `replacement_guidance`).
- Reference repository-level wrapper policy authority in
  `architecture/decisions/ADR-020-repository-wrapper-policy-boundaries.md`
  and git-specific precedent in
  `architecture/decisions/ADR-019-atomic-git-tool-wrapper-boundaries.md`.

## See Also

- **.opencode/guides/architecture/architecture_guide.md**: Architecture documentation
- **.opencode/guides/docstring_guide.md**: Code documentation style
- **docs/Theory/index.md**: Conceptual documentation
- **docs/Examples/index.md**: Practical tutorials
