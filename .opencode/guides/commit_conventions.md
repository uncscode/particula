# Commit Conventions

**Project:** particula  
**Last Updated:** 2026-06-06

Use concise Conventional Commit messages in imperative mood.

## Format

```text
<type>(<scope>): <summary>

<optional body>

<optional footer>
```

The scope is optional. Keep the summary short and specific.

## Types

- `feat`: New behavior or public API.
- `fix`: Bug fix or corrected scientific behavior.
- `test`: Test additions or updates.
- `docs`: Documentation-only changes.
- `refactor`: Internal restructuring without behavior changes.
- `perf`: Performance improvement.
- `style`: Formatting or non-behavioral style changes.
- `chore`: Tooling, packaging, or maintenance.

## Useful Scopes

- `activity`
- `dynamics`
- `condensation`
- `coagulation`
- `wall-loss`
- `equilibria`
- `gas`
- `particles`
- `util`
- `docs`
- `tests`

## Examples

```text
fix(condensation): preserve nonnegative concentrations in staggered stepping
```

```text
test(wall-loss): cover rectangular chamber edge cases
```

```text
docs: sync notebook workflow guidance
```

## Body Guidance

Use a body when context matters. Include what changed and why. For scientific
changes, mention the model, equation, or reference when useful.
