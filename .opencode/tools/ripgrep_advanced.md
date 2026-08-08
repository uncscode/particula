# ripgrep_advanced Tool Reference

Advanced content-search wrapper for low-level ripgrep controls.

## Scope Boundary

- `ripgrep_advanced` is content-search only.
- Keep `contentPattern` and `path` explicit.
- Advanced optional controls move through bounded `options` tokens.
- Advanced controls are intentionally kept off direct schema fields.

## Examples

```jsonc
{ "contentPattern": "TODO", "options": "context-lines=2" }
{ "contentPattern": "ERROR", "options": "before-context=2 after-context=1" }
{ "contentPattern": "import", "options": "files-with-matches" }
{ "contentPattern": "TODO", "path": ".opencode/tools", "options": "files-without-matches" }
{ "contentPattern": "secret", "options": "unrestricted=2 include-hidden" }
```

## Supported Parameters

| Parameter | Type | Required | Description |
|---|---|---:|---|
| `contentPattern` | string | ✅ | Literal pattern by default; use `match-mode=regex` for regex matching. |
| `path` | string | ❌ | Scoped search target (default: current working directory). File path = search only that file; directory path = search only that subtree. |
| `options` | string | ❌ | Bounded token carrier for advanced controls. |

## Supported `options` Tokens

- `pattern=<glob>`
- `file-type=<type>`
- `exclude-file-type=<type>`
- `glob-case-insensitive`
- `compact-output` (rewrite matched file paths relative to the scoped target)
- `max-results=<n>`
- `max-matches-per-file=<n>`
- `context-lines=<n>`
- `before-context=<n>`
- `after-context=<n>`
- `files-with-matches`
- `files-without-matches`
- `unrestricted=<1..3>`
- `match-mode=literal|regex`
- `ignore-gitignore`
- `include-hidden`

## Path Contract

- File `path` values search only the requested file.
- Directory `path` values search only the requested subtree.
- Missing, invalid, or out-of-repo `path` values fail closed with deterministic `ERROR:` output.

Example single-file advanced search:

```jsonc
{ "contentPattern": "TODO", "path": ".opencode/tools/search_content.ts", "options": "before-context=1 after-context=1" }
```

## Deterministic Behavior Notes

- `files-with-matches` and `files-without-matches` are mutually exclusive.
- Directional context takes precedence over `context-lines`.
- Result limits are `1..5000`, context values are `1..1000`, and unrestricted is `1..3`.
- Files-only modes cannot be combined with context controls.
- Search paths stay repository-confined via lexical and canonical validation.
- Scoped path misses never widen back to the repository root.
- `compact-output` rewrites file prefixes relative to the scoped file/directory base.
- Searches are terminated after 30 seconds and return a deterministic timeout error.
- Safety-clipped ripgrep stdout appends a deterministic warning so callers know
  the returned output may be incomplete.
