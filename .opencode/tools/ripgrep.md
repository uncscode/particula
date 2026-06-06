# Ripgrep Tool Reference

Compatibility reference for the broad `ripgrep` wrapper.

For new and updated workflows, prefer split search wrappers:
- `find_files` for discovery-only listing
- `search_content` for simple content search
- `ripgrep_advanced` for advanced low-level content controls

Use this page when maintaining legacy mixed search/discovery flows that still
depend on `ripgrep`.

## Preferred Routing (Split Wrappers)

```jsonc
// Discovery-only file listing
find_files({ "pattern": "**/*.ts", "path": "src" })

// Simple content search
search_content({ "contentPattern": "TODO", "path": "adw" })

// Advanced context controls
ripgrep_advanced({ "contentPattern": "ERROR", "path": "adw", "afterContext": 2 })
```

## Modes

| Mode                | Required params         | Output                        |
|---------------------|-------------------------|-------------------------------|
| File discovery      | `pattern`               | File paths sorted by mtime    |
| Content search      | `contentPattern`        | `file:line:content` matches   |
| Files-with-matches  | `contentPattern` + flag | File paths only (`-l` / `-L`) |

## Legacy Compatibility Examples (`ripgrep`)

```jsonc
// Find all TypeScript files
{ "pattern": "**/*.ts" }

// Find files in a subdirectory
{ "pattern": "**/*.py", "path": "src" }

// Search file contents
{ "contentPattern": "TODO" }

// Search contents in a specific directory
{ "contentPattern": "import os", "path": "adw" }

// Content search filtered to Python files
{ "pattern": "**/*.py", "contentPattern": "import" }

// Content search by file type
{ "contentPattern": "TODO", "fileType": "py" }

// Only file paths that contain a match
{ "contentPattern": "TODO", "filesWithMatches": true }

// Search hidden and gitignored files
{ "pattern": "**/*.js", "unrestricted": 2 }

// Content search with surrounding context
{ "contentPattern": "TODO", "contextLines": 2 }
```

## Advanced Examples

```jsonc
// Content search with directional context
{ "contentPattern": "TODO", "beforeContext": 2, "afterContext": 1 }

// Case-insensitive glob
{ "pattern": "**/readme.md", "globCaseInsensitive": true }

// Compact output (paths relative to search dir)
{ "pattern": "**/*.ts", "path": "src/deep/nested", "compactOutput": true }

// Limit results
{ "pattern": "**/*", "maxResults": 1000 }

// Content search with max matches per file
{ "contentPattern": "TODO", "maxMatchesPerFile": 5 }

// Files WITHOUT matches
{ "contentPattern": "TODO", "filesWithoutMatches": true }

// Exclude a file type
{ "contentPattern": "config", "excludeFileType": "json" }
```

## Parameter Reference

### Core

| Parameter        | Type   | Default | Description                                      |
|------------------|--------|---------|--------------------------------------------------|
| `pattern`        | string | —       | Glob pattern for file discovery or content filter |
| `contentPattern` | string | —       | Regex to search inside files (triggers content mode) |
| `path`           | string | cwd     | Directory to search in                           |

### Filtering

| Parameter          | Type   | Description                              |
|--------------------|--------|------------------------------------------|
| `fileType`         | string | Include only this file type (`-t`)       |
| `excludeFileType`  | string | Exclude this file type (`-T`)            |

### Output

| Parameter             | Type    | Default | Description                                 |
|-----------------------|---------|---------|---------------------------------------------|
| `filesWithMatches`    | boolean | false   | Return only file paths with matches (`-l`)  |
| `filesWithoutMatches` | boolean | false   | Return only file paths without matches (`-L`) |
| `compactOutput`       | boolean | false   | Paths relative to search dir instead of cwd |
| `maxResults`          | number  | 5000    | Max results returned                        |

### Ignore/Hidden

| Parameter         | Type    | Default | Description                                           |
|-------------------|---------|---------|-------------------------------------------------------|
| `ignoreGitignore` | boolean | false   | Skip .gitignore rules (`--no-ignore-vcs`)             |
| `includeHidden`   | boolean | false   | Include hidden files/dirs (`--hidden`)                |
| `unrestricted`    | 1/2/3   | —       | 1=ignore gitignore, 2=+hidden, 3=+binary (overrides) |

### Context (content search only)

| Parameter       | Type   | Description                              |
|-----------------|--------|------------------------------------------|
| `contextLines`  | number | Lines before AND after each match (`-C`) |
| `beforeContext` | number | Lines before each match (`-B`)           |
| `afterContext`  | number | Lines after each match (`-A`)            |

### Content Limits

| Parameter           | Type   | Description                                              |
|---------------------|--------|----------------------------------------------------------|
| `maxMatchesPerFile` | number | Max matches per file (`--max-count`), overrides maxResults |

### Other

| Parameter             | Type    | Description                    |
|-----------------------|---------|--------------------------------|
| `globCaseInsensitive` | boolean | Case-insensitive glob matching |

## No-Op Input Handling

All optional parameters silently ignore no-op values. You never need to guard against
accidentally passing these — the tool normalizes them to "not provided":

| Input type           | Affected params       | Behavior              |
|----------------------|-----------------------|-----------------------|
| `""` (exact-empty string) | All optional string params | Treated as omitted |
| `0`                  | All numeric params    | Treated as omitted    |
| Negative numbers     | All numeric params    | Treated as omitted    |
| `NaN`                | All numeric params    | Treated as omitted    |
| `false`              | All boolean params    | Same as omitting      |
| Whitespace-only `" "`| `fileType`, `excludeFileType`, `path` | Trimmed, then omitted |

Notes:
- Exact-empty string normalization (`""`) happens before mode detection and argument
  assembly, so an empty optional `contentPattern` is treated as not provided.
- Existing whitespace validation behavior is unchanged. Whitespace-only values are only
  auto-omitted for the trimmed optional params listed above.

This means `{ "unrestricted": 0 }` is the same as `{}`, and
`{ "maxResults": 0 }` falls back to the 5000 default.

## Parameter Precedence

- `contentPattern` switches to content search; without it, content-only params are ignored.
- `filesWithMatches` / `filesWithoutMatches` are mutually exclusive.
- `beforeContext` / `afterContext` override `contextLines`.
- `maxMatchesPerFile` overrides `maxResults` for `--max-count`.
- `--max-count` is omitted when context flags are active.
- `unrestricted` overrides `ignoreGitignore` and `includeHidden`.

## Auto-Retry Behavior

File discovery only. When initial search returns 0 results and ignore flags were not
explicitly set, the tool automatically retries with `ignoreGitignore=true` and
`includeHidden=true`. Output includes a clear indicator when auto-retry occurred.

## Path Safety

- Searches are constrained to the repository root.
- Boundary checks are enforced in two stages:
  1. Lexical normalization/prefix validation
  2. Canonical `realpath` validation after path existence checks
- This dual check blocks symlink-based path escapes while preserving valid in-repo
  symlink usage.
