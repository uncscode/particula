# find_files Tool Reference

Discovery-only file search tool for simple glob-based file listing.

## Scope Boundary

- `find_files` supports **file discovery only**.
- Content-search belongs to `search_content`; advanced controls belong to `ripgrep_advanced`.
- Unsupported direct fields are outside the public wrapper schema; bounded `options`
  tokens still fail closed when callers try unsupported wrapper-specific controls.

## Simple Examples

```jsonc
// Find markdown files in repository
{ "pattern": "**/*.md" }

// Restrict discovery to a directory
{ "pattern": "**/*.py", "path": "adw" }

// Limit returned results
{ "pattern": "**/*", "options": "max-results=100" }

// Return paths relative to the search directory
{ "pattern": "**/*.ts", "path": ".opencode/tools", "options": "compact-output" }

// Include or exclude ripgrep file types
{ "pattern": "**/*", "options": "file-type=py" }
{ "pattern": "**/*", "options": "exclude-file-type=json" }
```

## Supported Parameters

| Parameter | Type | Required | Description |
|---|---|---:|---|
| `pattern` | string | ✅ | Glob pattern for discovery (non-empty after trim). |
| `path` | string | ❌ | Scoped discovery target (default: current working directory). File path = discover only that file; directory path = discover only that subtree. |
| `options` | string | ❌ | Bounded token carrier for optional discovery controls. |

## Supported `options` Tokens

- `file-type=<type>`
- `exclude-file-type=<type>`
- `glob-case-insensitive`
- `compact-output`
- `max-results=<n>`

## Path Contract

- File `path` values discover only the requested file.
- Directory `path` values discover only the requested subtree.
- Missing, invalid, or out-of-repo `path` values fail closed with deterministic `ERROR:` output.
- `compact-output` stays relative to the searched directory for directory targets and to the file's parent directory for file targets.

Example single-file discovery:

```jsonc
{ "pattern": "**/*", "path": ".opencode/tools/find_files.ts", "options": "compact-output" }
```

## Deterministic Behavior Notes

- Results are sorted by modification time (most recent first).
- Search paths are validated against repository boundaries with lexical and canonical checks.
- Scoped path misses never widen back to the repository root.
- No-match outcomes are non-error, deterministic text responses.
- If results exceed `maxResults`, output appends a deterministic truncation warning.
