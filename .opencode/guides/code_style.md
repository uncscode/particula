# Code Style Guide

**Version:** 2.1.0
**Last Updated:** 2025-11-14

## Overview

This guide documents Python-specific coding standards for the adw repository.

> **See Also:** [Code Culture](code_culture.md) - Development philosophy, code review practices, and the "smooth is safe, safe is fast" principle including the 100-line rule for PRs.

### Language Version

**Minimum Version**: 3.12+

## Naming Conventions

### Functions/Methods: `snake_case`

**Example:**
```python
def calculate_total_cost(items: list) -> float:
    pass
```

### Variables: `snake_case`

**Example:**
```python
user_name = "John Doe"
total_count = 42
```

### Classes/Types: `PascalCase`

**Example:**
```python
class WorkflowManager:
    pass

class UserProfile:
    pass
```

### Constants: `UPPER_SNAKE_CASE`

**Example:**
```python
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
API_BASE_URL = "https://api.example.com"
```

## Type System

### Type Hints/Annotations

**Policy**: Recommended for public APIs and complex functions

**Example:**
```python
def process_workflow(
    workflow_id: str,
    config: dict[str, Any],
    timeout: int = 30
) -> WorkflowResult:
    pass
```

## Line Length

**Maximum Line Length**: 100 characters

## Import Organization

Organize imports in three groups:
1. Standard library imports
2. Third-party imports
3. Local application imports

Use blank lines to separate groups. Sort alphabetically within each group.

**Example:**
```python
# Standard library
import os
from pathlib import Path

# Third-party
import click
from pydantic import BaseModel

# Local
from adw.core.models import Workflow
from adw.workflows.operations import execute_workflow
```

## Design Patterns

### Pattern 1: Pydantic Models for Data Validation

**When to use:** For API requests, configuration objects, and data structures that need validation

**Example:**
```python
from pydantic import BaseModel, Field

class WorkflowConfig(BaseModel):
    workflow_id: str = Field(..., min_length=1)
    timeout: int = Field(default=30, ge=0)
    retry_count: int = Field(default=3, ge=0, le=10)
```

## See Also

- **docs/ai_docs/linting_guide.md**: Linting tools that enforce these conventions
- **docs/ai_docs/docstring_guide.md**: Documentation standards
