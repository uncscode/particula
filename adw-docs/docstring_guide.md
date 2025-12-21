# Docstring Guide

Use **{{DOCSTRING_STYLE}}** docstrings to explain intent, inputs, outputs, and error cases. Keep lines within {{LINE_LENGTH}} characters.

## When to add docstrings

- Public modules, classes, functions, and methods must have docstrings.
- Private helpers may omit docstrings if names are self-explanatory; add them when behavior is non-obvious.
- Keep module-level docstrings short; describe purpose and major dependencies.

## Structure (Google style)

Order sections as needed: summary line → blank line → Args → Returns → Raises → Examples.

```python
def load_config(path: str) -> dict[str, str]:
    """Load configuration from a file.

    Args:
        path: File path to the config file.

    Returns:
        Parsed configuration values keyed by setting name.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the contents cannot be parsed.
    """
    ...
```

## Classes and methods

```python
class Client:
    """HTTP client for {{PACKAGE_NAME}} APIs."""

    def __init__(self, base_url: str) -> None:
        """Initialize the client.

        Args:
            base_url: Root URL for the API.
        """
        self.base_url = base_url
```

- Document constructor arguments and noteworthy attributes.
- For properties, include what is returned and whether it may be `None`.

## Modules

At the top of each module:

```python
"""Utilities for interacting with external services.

Exposes a thin wrapper around requests with retry and logging helpers.
"""
```

## Exceptions

- Mention expected exceptions in **Raises** when callers should handle them.
- Avoid documenting low-level exceptions that are internal implementation details.

## Examples

Provide short, runnable examples when they clarify usage:

```python
def render_name(first: str, last: str) -> str:
    """Return a display name.

    Args:
        first: First name.
        last: Last name.

    Returns:
        Full display name in "Last, First" format.

    Examples:
        >>> render_name("Ada", "Lovelace")
        'Lovelace, Ada'
    """
    return f"{last}, {first}"
```

## Style tips

- Use imperative tone for summaries: "Return", "Validate", "Load".
- Keep summaries to one line; wrap details to stay within {{LINE_LENGTH}} chars.
- Prefer explicit types in signatures (Python {{PYTHON_VERSION}} support assumed).
- Keep wording consistent with [Code Style](./code_style.md) and rely on [Linting Guide](./linting_guide.md) to catch formatting issues.
