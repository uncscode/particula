# Commit Conventions

Use semantic, focused commits to keep history readable and reviews efficient.

## Prefixes

Use one of these prefixes:
- `feat:` new functionality
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code change without behavior change
- `test:` add or adjust tests
- `chore:` tooling, deps, or maintenance

## Guidelines

- One concern per commit; avoid mixing refactors with feature work.
- Present tense, imperative mood: "add", "fix", "update".
- Link issues when relevant: `fix: handle empty payloads (closes #123)`.
- Keep subject ≤ 72 characters; add details in the body if needed.

## Examples

Good:
```
feat: add retry policy for api client
fix: guard None inputs in renderer
docs: expand testing guide with coverage instructions
refactor: extract env parsing helper
```

Bad:
```
update stuff
misc changes
fix bugs
```

## Body template (when more detail is needed)

```
<type>: <short summary>

Why:
- <reason 1>
- <reason 2>

How:
- <change 1>
- <change 2>

Testing:
- {{TEST_COMMAND}}
- {{LINT_COMMAND}}
- {{TYPE_CHECK_COMMAND}}
```

## Review expectations

- Ensure each commit passes lint, format, type-check, and tests before pushing.
- Avoid force pushes after reviews start; prefer follow-up commits to address feedback.
