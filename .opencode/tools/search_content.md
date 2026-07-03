# search_content Tool Reference

Constrained content-search wrapper for common `ripgrep` workflows.

## Scope Boundary

- `search_content` supports **simple content search only**.
- Advanced direct fields are outside the public wrapper schema; advanced-style
  `options` tokens still fail closed with deterministic `ERROR:` responses.
- Use **`ripgrep_advanced`** for advanced flags.

## When to use search_content vs ripgrep_advanced

- Use `search_content` when you need quick content matching with a small safe schema.
- Use `ripgrep_advanced` for context lines, hidden-file controls, unrestricted mode, or files-only result modes.

## Simple Examples

```jsonc
// Minimal content search
{ "contentPattern": "TODO" }

// Filter to Python files in adw/
{ "contentPattern": "import", "path": "adw", "options": "file-type=py" }

// Bound output
{ "contentPattern": "ERROR:", "options": "max-results=200 max-matches-per-file=3" }
```

## Supported Parameters

| Parameter | Type | Required | Description |
|---|---|---:|---|
| `contentPattern` | string | ✅ | Regex pattern to search within files (non-empty after trim). |
| `path` | string | ❌ | Scoped search target (default: current working directory). File path = search only that file; directory path = search only that subtree. |
| `options` | string | ❌ | Bounded token carrier for optional simple-search controls. |

## Supported `options` Tokens

- `pattern=<glob>`
- `file-type=<type>`
- `exclude-file-type=<type>`
- `glob-case-insensitive`
- `compact-output` (rewrite matched file paths relative to the scoped target)
- `max-results=<n>`
- `max-matches-per-file=<n>`

## Path Contract

- File `path` values search only the requested file.
- Directory `path` values search only the requested subtree.
- Missing, invalid, or out-of-repo `path` values fail closed with deterministic `ERROR:` output.

Example single-file search:

```jsonc
{ "contentPattern": "TODO", "path": ".opencode/tools/search_content.ts" }
```

## Unsupported Advanced Controls

The following controls are not part of the `search_content` schema and should be
routed to `ripgrep_advanced` instead:

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
- Scoped path misses never widen back to the repository root.
- Empty result sets return deterministic **non-error** text.
- Wrapper execution and invalid-regex failures return deterministic `ERROR:` diagnostics.
- `compact-output` rewrites file prefixes relative to the scoped file/directory base.
- Output larger than `maxResults` appends deterministic truncation warning text.
