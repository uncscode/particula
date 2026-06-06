# platform_pr_review_write

Executes `adw platform pr-review` with strict validation and deterministic errors.

## Command

- `pr-review`

## Required args

- `issue_number` (strict positive integer token)
- `body` (non-empty after trim)

## Optional args

- `path`, `line`, `position`, `commit_sha`
- `prefer_scope`: `fork|upstream`
- `help`: when `true`, runs `--help` and bypasses required-arg validation

## Relation checks

- `--line` requires `--path`
- `--position` requires `--path`
- `--path` requires `--line` or `--position`

## Error envelope

Subprocess failures return:

- `ERROR: Failed to execute 'adw platform pr-review'`
- diagnostics in deterministic order: `STDERR` then `STDOUT`

## Validation and Recovery Examples

Pre-spawn validation example (relation checks):

```text
ERROR: --line requires --path
```

Delegated failure example:

```text
ERROR: Failed to execute 'adw platform pr-review'
```

Routing hint:

- Prefer `platform_pr_review_write` for review comments and inline review operations; keep `platform_operations` as compatibility/delegation path only.

## Examples

```json
{"command":"pr-review","issue_number":"42","body":"Looks good overall"}
{"command":"pr-review","issue_number":"42","body":"Inline note","path":"src/mod.ts","line":12}
```
