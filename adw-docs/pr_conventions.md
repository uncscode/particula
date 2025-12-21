# Pull Request Conventions

**Version:** 0.2.6
**Last Updated:** 2025-11-30

## Overview

This guide documents pull request format and conventions for the **particula** repository. Following these conventions ensures consistent PR quality, enables efficient code review, and maintains project standards.

### PR Title Format

PR titles should follow the same format as commit messages (Conventional Commits).

**Format:**
```
<type>(<scope>): <description>
```

**Examples:**
```
feat(dynamics): add coagulation kernel for Brownian motion
fix(particles): correct mass conservation in condensation
docs(examples): add chamber wall loss simulation notebook
test(activity): add comprehensive tests for BAT model
```

### PR Body Structure

All PRs should use the template from `.github/PULL_REQUEST_TEMPLATE/pull_request_template.md`.

Required sections:

```markdown
## Summary

Brief description of what this PR does and why.

## Changes

- List of key changes made
- Use bullet points for clarity
- Mention affected modules/files

## Testing

- Describe how changes were tested
- List any new tests added
- Include test coverage information

## Scientific Context (if applicable)

- Reference papers or algorithms implemented
- Explain numerical methods used
- Describe validation approach

## Checklist

- [ ] Tests pass locally
- [ ] Linting passes (ruff check + format)
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Examples updated (if needed)
```

### Branch Naming

Use descriptive branch names that indicate the purpose of the changes.

**Format**: `<type>/<issue-number>-<brief-description>` or `<type>/<brief-description>`

**Examples:**
- `feat/123-add-brownian-coagulation`
- `fix/456-mass-conservation-condensation`
- `docs/update-installation-guide`
- `test/activity-model-edge-cases`
- `refactor/simplify-input-validation`

### Platform Commands

particula is hosted on **GitHub**. Use the `gh` CLI for creating pull requests.

**Creating a PR:**
```bash
gh pr create --title "feat(dynamics): add Brownian coagulation kernel" --body "..."
```

**Using HEREDOC for body (recommended for ADW):**
```bash
gh pr create --title "feat(dynamics): add Brownian coagulation kernel" --body "$(cat <<'EOF'
## Summary
Implements the Fuchs form of Brownian coagulation kernel.

## Changes
- Added `brownian_kernel()` function in `dynamics/coagulation/`
- Includes slip correction factor
- Vectorized implementation for performance

## Testing
- Added 15 new tests in `coagulation/tests/brownian_kernel_test.py`
- All tests pass
- Coverage: 100% for new code

## Scientific Context
Based on Fuchs (1964) formulation with Cunningham slip correction.
Validated against analytical solutions for monodisperse aerosols.

## Checklist
- [x] Tests pass locally
- [x] Linting passes
- [x] Documentation updated
- [x] Type hints added
EOF
)"
```

### Issue Linking

Link PRs to issues using GitHub keywords in the PR description.

**Format**: Include in PR body:
```
Closes #<issue-number>
```

**Multiple Issues**:
```
Closes #123
Fixes #456
Related to #789
```

**Example**:
```markdown
## Summary
Fixes mass conservation bug in condensation solver.

Closes #456
Related to #123
```

## CI Checks

All PRs must pass CI checks before merging:

### Required Checks

1. **Tests** - pytest with minimum 500 tests
   - Command: `pytest --cov=particula --cov-report=term-missing`
   - Must achieve >90% coverage for new code
   - Timeout: 10 minutes

2. **Linting** - ruff (check + format)
   - Commands:
     ```bash
     ruff check particula/ --fix
     ruff format particula/
     ruff check particula/
     ```
   - All three must pass without errors

3. **Type Checking** (optional) - mypy
   - Command: `mypy particula/ --ignore-missing-imports`
   - Warnings acceptable, errors should be fixed

4. **Documentation Build** - MkDocs
   - Ensures documentation builds without errors
   - Validates docstring formatting

### CI Workflow

CI runs on:
- Ubuntu (latest)
- macOS (latest)
- Windows (latest)

On:
- Every push to PR branch
- PR creation/update
- Main branch merges

See `.github/workflows/test.yml` and `.github/workflows/lint.yml` for details.

## Review Requirements

Before a PR can be merged:

- [ ] At least one approving review from a maintainer
- [ ] All CI checks passing (green)
- [ ] No unresolved comments
- [ ] Conflicts resolved with main branch
- [ ] PR size reasonable (~100 lines of production code, see [Code Culture](code_culture.md))

## Integration with ADW

ADW PR commands use this guide to:
- Format PR titles using conventional commits format
- Structure PR bodies with required sections
- Create PRs using `gh` CLI with proper formatting
- Link PRs to issues using GitHub keywords
- Ensure CI requirements are met before PR creation

## See Also

- **[Commit Conventions](commit_conventions.md)**: Commit message format
- **[Review Guide](review_guide.md)**: Code review process and criteria
- **[Testing Guide](testing_guide.md)**: Test requirements and execution
- **[Linting Guide](linting_guide.md)**: Linting requirements
- **[Code Culture](code_culture.md)**: Development philosophy and the 100-line rule
