# Code Culture

**Project:** particula  
**Last Updated:** 2026-06-06

particula development values small, correct, scientifically defensible changes.

## Principles

- Prefer the smallest correct change.
- Make scientific assumptions explicit.
- Keep code readable before clever.
- Validate public numerical inputs.
- Use tests to protect behavior and edge cases.
- Preserve vectorized NumPy performance where practical.
- Document equations, units, and citations.

## Reviewability

Aim for focused changes that can be reviewed in one pass. If a task grows, split
implementation, tests, docs, or architecture work into separate commits or PRs
where practical.

## Scientific Safety

Scientific code should make units, domains, and assumptions visible. Prefer
explicit names like `particle_radius` and `gas_temperature` over short ambiguous
names. Use constants from `particula.util.constants` rather than hardcoded
physical constants.

## Testing Mindset

Tests should demonstrate physical behavior, numerical stability, and regression
coverage. Use `pytest -Werror` when warnings might be introduced by numerical
edge cases.

## Documentation Mindset

Update guides, examples, notebooks, or docstrings when public APIs, model
behavior, or developer workflows change. Notebook examples should remain synced
and executable.
