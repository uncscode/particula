# PR Conventions

Create small, reviewable pull requests with clear intent, testing notes, and a checklist. Align the description with the work performed.

## Title format

```
<type>: <short summary>
```

Examples:
- `feat: add pagination to search results`
- `fix: prevent None payload crash`
- `docs: refresh testing guide`

## Description template

```
## What
- Summarize the change in 2–3 bullets.

## Why
- Explain the problem or motivation.

## How
- Outline key implementation choices.

## Testing
- {{TEST_COMMAND}}
- {{LINT_COMMAND}}
- {{FORMAT_COMMAND}}
- {{TYPE_CHECK_COMMAND}}
```

## Checklist

- [ ] Small, focused diff (prefer < ~300 lines where possible)
- [ ] Tests added/updated and passing locally
- [ ] Lint/format/type-check clean
- [ ] Docs updated when behavior or APIs change
- [ ] Screenshots or logs attached when UI/CLI output is relevant

## Review etiquette

- Request reviewers familiar with the area; add context in the description.
- Respond to feedback promptly; prefer follow-up commits over force pushes during review.
- Resolve threads only after addressing the comment.
- Link related issues and mention breaking changes explicitly.

## Related guides

- [Testing Guide](./testing_guide.md)
- [Linting Guide](./linting_guide.md)
- [Commit Conventions](./commit_conventions.md)
