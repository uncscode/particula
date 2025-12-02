---
name: Patch Workflow (Quick Fix)
about: Quick patch workflow for simple fixes and small changes
title: '[Patch]: '
labels: ["agent", "blocked", "model:base", "type:patch"]
assignees: ''
---


## Issue Type
<!-- Patch workflow is for quick fixes and simple changes -->
- [ ] Bug Fix - Quick fix for a specific bug
- [ ] Typo/Documentation Fix - Simple text corrections
- [ ] Minor Enhancement - Small, isolated improvement
- [ ] Configuration Change - Update settings or config files

## ADW Instructions
<!-- Patch workflow: plan → build → ship (no extensive testing/review/documentation) -->

**Context to read:**
- [x] Read docs/Agent/README.md for project structure and conventions
- [x] Review docs/Agent/architecture/architecture_outline.md for system architecture
- [x] Review docs/Agent/code_style.md for coding standards
- [x] Review docs/Agent/testing_guide.md for test requirements
- [ ] Additional context:

**Testing requirements:**
- [x] Write unit tests for all new methods/functions
- [x] Ensure all existing tests pass
- [ ] Additional testing:

**Code quality:**
- [x] Follow Google-style docstrings (see docs/Agent/docstring_guide.md)
- [x] Pass ruff linter (see docs/Agent/linting_guide.md)
- [ ] Additional requirements:


## Description
<!-- Clear and concise description of what needs to be fixed -->


## Current Behavior
<!-- What is currently happening? -->


## Expected Behavior
<!-- What should happen instead? -->


## Files to Modify
<!-- List the specific file(s) that need changes -->
-
-


## Proposed Change
<!-- Describe the specific change to make -->


## Additional Testing Instructions
<!-- How to verify the fix works -->
- [ ] 
- [ ]


## Additional Context
<!-- Error messages, screenshots, code snippets, etc. -->
```python
# Code example if applicable
```


## Related Issues
<!-- Link to related issues -->
<!-- Fixes #123 -->

