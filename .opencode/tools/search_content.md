# search_content Tool Reference

Constrained content-search wrapper for common `ripgrep` workflows.

## Scope Boundary

- `search_content` supports **simple content search only**.
- Advanced controls are intentionally rejected with deterministic `ERROR:` responses.
- Use **`ripgrep_advanced`** for advanced flags.

## When to use search_content vs ripgrep_advanced

- Use `search_content` when you need quick content matching with a small safe schema.
- Use `ripgrep_advanced` for context lines, hidden-file controls, unrestricted mode, or files-only result modes.

## Simple Examples

```jsonc
// Minimal content search
{ "contentPattern": "TODO" }

// Filter to Python files in adw/
{ "contentPattern": "import", "path": "adw", "fileType": "py" }

// Bound output
{ "contentPattern": "ERROR:", "maxResults": 200, "maxMatchesPerFile": 3 }
```

## Supported Parameters

| Parameter | Type | Required | Description |
|---|---|---:|---|
| `contentPattern` | string | ✅ | Regex pattern to search within files (non-empty after trim). |
| `pattern` | string | ❌ | Optional glob filter (`--glob`) applied during content search. |
| `path` | string | ❌ | Directory scope for search (default: current working directory). |
| `fileType` | string | ❌ | Include only this ripgrep file type (`-t`). |
| `excludeFileType` | string | ❌ | Exclude this ripgrep file type (`-T`). |
| `globCaseInsensitive` | boolean | ❌ | Case-insensitive glob matching. |
| `compactOutput` | boolean | ❌ | Included for schema parity; content output remains `file:line:content`. |
| `maxResults` | number | ❌ | Max output lines returned (default: 5000). |
| `maxMatchesPerFile` | number | ❌ | Bounds matches per file (`--max-count`). |

## Unsupported Advanced Controls

The following fields fail closed with `ERROR:` and guidance to use `ripgrep_advanced`:

- `contextLines`
- `beforeContext`
- `afterContext`
- `filesWithMatches`
- `filesWithoutMatches`
- `unrestricted`
- `ignoreGitignore`
- `includeHidden`

## Deterministic Behavior Notes

- Search paths are validated against repository boundaries (lexical and canonical checks).
- Empty result sets return deterministic **non-error** text.
- Wrapper execution and invalid-regex failures return deterministic `ERROR:` diagnostics.
- Output larger than `maxResults` appends deterministic truncation warning text.
