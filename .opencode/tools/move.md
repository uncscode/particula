# move (compatibility)

Compatibility wrapper for move operations.

Prefer split wrappers for new and updated integrations:

- `move_safe` for safe moves (no overwrite, no trash)
- `move_overwrite` for explicit overwrite behavior
- `move_trash` for soft-delete into `.trash/<relative-path>`

`move` remains active during the compatibility window and supports legacy
options.

Compatibility routing stays explicit:

- default mode delegates to `move_safe`
- `overwrite: true` delegates to `move_overwrite`
- `trash: true` delegates to `move_trash`
- when both `trash` and `overwrite` are true, trash routing takes precedence

## Simple Examples

```jsonc
// Move a file
{ "source": "src/old.ts", "destination": "src/new.ts" }

// Move into a directory (keeps original name)
{ "source": "src/file.ts", "destination": "lib/" }

// Move a folder
{ "source": "src/old-dir", "destination": "src/new-dir" }

// Rename in place
{ "source": "src/utils.ts", "destination": "src/helpers.ts" }

// Soft-delete to trash
{ "source": "src/deprecated.ts", "trash": true }
```

## Advanced Examples

```jsonc
// Overwrite existing destination
{ "source": "src/new.ts", "destination": "src/existing.ts", "overwrite": true }

// Trash a directory
{ "source": "adw/old-module/", "trash": true }
```

## Parameter Reference

| Parameter     | Type    | Default | Description                              |
|---------------|---------|---------|------------------------------------------|
| `source`      | string  | —       | Source file or folder path (required)    |
| `destination` | string  | —       | Required unless `trash: true` |
| `overwrite`   | boolean | false   | Compatibility routing toggle for overwrite mode |
| `trash`       | boolean | false   | Compatibility routing toggle for trash mode |

## Preferred Routing

| Intent | Preferred wrapper |
|---|---|
| Safe file/folder move | `move_safe` |
| Replace existing destination | `move_overwrite` |
| Soft-delete into `.trash/` | `move_trash` |
| Legacy mixed usage (`overwrite`/`trash` flags) | `move` (compatibility) |

Payload-bearing fields remain explicit on the compatibility surface:

- `source` is always required
- `destination` is required for safe/overwrite routing
- `overwrite` and `trash` remain direct compatibility toggles rather than
  bounded options

## Destination Behavior

| Destination pattern | Behavior                              |
|---------------------|---------------------------------------|
| `"lib/"` (ends with /) | Move INTO directory, keep original name |
| `"lib/new.ts"`      | Move/rename to exact path             |
| Parent dirs missing  | Created automatically                 |

## Trash Mode

Trash mode provides soft-delete with a git-auditable trail:

- Moves to `.trash/<original-path>` preserving directory structure
- The `destination` parameter is ignored and may be omitted entirely
- `overwrite` is also ignored because trash routing wins when both toggles
  are present
- Works with both files AND directories
- Git tracks this as a MOVE (not deletion), preserving history
- `.trash/` folder is auto-created at repository root

**Why trash mode?**
- Git history preserved (move, not delete)
- `git add --all` won't stage a deletion
- Maintainers review `.trash/` before permanent removal
- Files can be restored by moving them back

**Result:** `src/deprecated.ts` -> `.trash/src/deprecated.ts`

## Security Restrictions

- Destination MUST be within the repository (current working directory)
- Moving files outside the repository is NOT allowed
- Cross-filesystem moves are NOT supported
- Source must also be within the repository

## Migration Guidance

- Prefer split wrappers in docs and new agent/tool integrations.
- Keep `move` as an active compatibility surface until retirement gates are
  met.
- Downstream allowlist/compat cleanup remains in **E20-F11**; this page
  documents current behavior only.

## Output Format

| Outcome | Format |
|---------|--------|
| Success (safe/overwrite) | `SUCCESS: Moved <type>: <source> -> <destination>` |
| Success (trash) | `SUCCESS: Moved <type> to trash: <source>` |
| Error   | `ERROR [<CODE>]: <message>` with optional hint |

## Error Codes

| Code                      | Meaning                              |
|---------------------------|--------------------------------------|
| `INVALID_SOURCE`          | Source parameter empty or invalid    |
| `INVALID_DESTINATION`     | Destination parameter empty          |
| `SOURCE_NOT_FOUND`        | Source file/directory doesn't exist  |
| `DEST_EXISTS`             | Destination exists (use overwrite)   |
| `DEST_OUTSIDE_REPO`       | Destination outside repository       |
| `SOURCE_OUTSIDE_REPO`     | Source outside repository            |
| `CROSS_DEVICE_NOT_ALLOWED`| Different filesystems                |
| `SAME_PATH`               | Source and destination are identical |
| `ALREADY_IN_TRASH`        | Source already in .trash/            |
| `TRASH_PATH_EXISTS`       | Conflict in .trash/ directory        |
