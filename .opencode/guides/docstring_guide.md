# Docstring Guide

**Version:** 2.1.0
**Last Updated:** 2025-11-14

## Overview

This guide documents the documentation style conventions for the adw repository. All functions, classes, methods, and modules should follow **Google-style** format.

### Documentation Style

adw uses **Google-style** as the standard documentation format for Python docstrings.

### Integration with ADW

This guide is referenced by ADW commands to understand repository-specific docstring requirements.

## Format Structure

### Python Example: Google-Style

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description of what the function does.

    Longer description explaining the purpose and methodology.
    Multiple paragraphs can be used for complex functions.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of the return value.

    Raises:
        ValueError: Condition that raises this exception.
        RuntimeError: Another condition that raises exception.

    Examples:
        >>> function_name("test", 42)
        True
    """
    pass
```

## Required Sections

### For Functions

- **Brief description**: One-line summary
- **Args**: List all parameters with type and description
- **Returns**: Describe return value with type
- **Raises** (optional): List exceptions/errors with conditions
- **Examples** (optional but recommended): Usage examples with doctests

### For Classes

- **Brief description**: One-line summary
- **Attributes**: List all public attributes with type and description
- **Examples** (optional): Usage examples

**Class Example:**
```python
class WorkflowManager:
    """Manages workflow execution and state.

    The WorkflowManager coordinates workflow phases, tracks state,
    and handles error recovery.

    Attributes:
        workflow_id: Unique identifier for the workflow.
        current_phase: Current execution phase name.
        state: Current workflow state dictionary.

    Examples:
        >>> manager = WorkflowManager("wf-123")
        >>> manager.execute_phase("plan")
    """
    pass
```

## Line Length

**Docstring Line Length**: 100 characters

Wrap docstring lines at 100 characters for consistency with code line length.

## Quick Reference

### Checklist for New Documentation

- [ ] Brief description (one line)
- [ ] Detailed description (if non-trivial)
- [ ] Args section (all parameters documented)
- [ ] Returns section (return value described)
- [ ] Raises section (if function raises exceptions)
- [ ] Examples section (recommended for public APIs)
- [ ] Line lengths ≤ 100 characters

## See Also

- **docs/ai_docs/linting_guide.md**: Docstring linting rules
- **docs/ai_docs/code_style.md**: General coding standards
