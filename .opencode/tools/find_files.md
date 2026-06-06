# find_files Tool Reference

Discovery-only file search tool for simple glob-based file listing.

## Scope Boundary

- `find_files` supports **file discovery only**.
- Content-search belongs to `search_content`; advanced controls belong to `ripgrep_advanced`.
- Unsupported fields (for example `contentPattern`, context flags, and files-with-matches flags) fail closed with deterministic `ERROR:` messages.

## Simple Examples

```jsonc
// Find markdown files in repository
{ "pattern": "**/*.md" }

// Restrict discovery to a directory
{ "pattern": "**/*.py", "path": "adw" }

// Limit returned results
{ "pattern": "**/*", "maxResults": 100 }

// Return paths relative to the search directory
{ "pattern": "**/*.ts", "path": ".opencode/tool", "compactOutput": true }

// Include or exclude ripgrep file types
{ "pattern": "**/*", "fileType": "py" }
{ "pattern": "**/*", "excludeFileType": "json" }
```

## Supported Parameters

| Parameter | Type | Required | Description |
|---|---|---:|---|
| `pattern` | string | ✅ | Glob pattern for discovery (non-empty after trim). |
| `path` | string | ❌ | Directory to search (default: current working directory). |
| `fileType` | string | ❌ | Include only this ripgrep type (`-t`). |
| `excludeFileType` | string | ❌ | Exclude this ripgrep type (`-T`). |
| `globCaseInsensitive` | boolean | ❌ | Case-insensitive glob matching. |
| `compactOutput` | boolean | ❌ | Output paths relative to search directory. |
| `maxResults` | number | ❌ | Max files returned (default: 5000). |

## Deterministic Behavior Notes

- Results are sorted by modification time (most recent first).
- Search paths are validated against repository boundaries with lexical and canonical checks.
- No-match outcomes are non-error, deterministic text responses.
- If results exceed `maxResults`, output appends a deterministic truncation warning.
