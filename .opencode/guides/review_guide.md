# Code Review Guide

**Version:** 2.1.0
**Last Updated:** 2025-11-14

## Overview

This guide documents code review standards and criteria for the adw repository.

### Repository Structure

```
adw/
├── adw/                # Source code directory (core package)
├── tests/              # Test files (*_test.py, test_*.py)
├── docs/               # Documentation files
└── pyproject.toml      # Project configuration and dependencies
```

### Validation Commands

Before submitting code for review, run:

```bash
# Tests
pytest adw/tests/

# Linting
ruff check adw/ --fix
ruff format adw/

# Type checking
mypy adw/ --ignore-missing-imports
```

### Review Checklist

- [ ] Tests pass: `pytest adw/tests/`
- [ ] Linting passes: `ruff check adw/ && ruff format --check adw/`
- [ ] Type checking passes: `mypy adw/ --ignore-missing-imports`
- [ ] Documentation updated (if applicable)
- [ ] No hardcoded secrets or credentials
- [ ] Error handling is appropriate
- [ ] Code follows existing patterns and conventions

### Issue Severity

- **Blocker**: Critical issues that prevent the code from running or cause data loss/corruption. Must be fixed before merge.
- **Major**: Significant issues that impact functionality, security, or maintainability. Should be fixed before merge unless approved by maintainer.
- **Minor**: Style issues, non-critical bugs, or suggestions for improvement. Can be addressed in follow-up PRs.

## Integration with ADW

ADW review commands use this guide to validate:
- Code passes all validation commands
- Review criteria are met
- Issue severity is appropriately assessed

## See Also

- **docs/ai_docs/testing_guide.md**: Test execution and validation
- **docs/ai_docs/linting_guide.md**: Linting requirements
- **docs/ai_docs/code_style.md**: Code style conventions
- **docs/ai_docs/architecture_reference.md**: Architectural constraints
