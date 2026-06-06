# adw_spec_write

Focused ADW spec wrapper for mutation operations.

## Commands
- `write`
- `delete`

## Behavior
- Validates `adw_id` fail-closed
  - missing, blank, null, and non-string values fail as required-input errors before format validation or spawn
  - malformed non-blank strings still fail the 8-character-hex validation path
- `write` requires non-empty `content` or non-empty `file`
- `delete` requires non-empty `field`
- `write --file` is canonicalized and constrained to existing repository-confined paths across compatibility and split wrappers
- non-blank `field` values are trimmed before forwarding for both `write` and `delete`
- Optional flags are sparse:
  - `append` emitted only when `true`
  - `confirm` emitted only when `true`

## Contract Note
- Success is always envelope-based for this wrapper:
  `ADW Spec Command: <command>\n\n<stdout>`.
- Failures are deterministic:
  - `ERROR: adw spec <command> failed (exit N)` on non-zero exits.
  - `ERROR: Failed to execute adw spec <command>. ...` on execution errors.
- Execution-error diagnostic precedence remains `stderr` -> `stdout` -> message/fallback.
- Failure diagnostics are bounded, strip control characters, and redact absolute filesystem paths to `<path>`.

## Failure and Recovery Examples

Pre-spawn validation (write payload required):

```text
ERROR: 'write' command requires either 'content' or 'file' parameter (non-empty).
```

Required-input parity example:

```text
ERROR: 'adw_id' parameter is required for all spec commands.

Usage: Use the ADW tool with command "status" to see active workflow IDs.
Example: { command: "write", adw_id: "abc12345", content: "New spec content" }
```

Malformed-but-nonblank identifier example:

```text
ERROR: 'adw_id' must be an 8-character hex string (e.g., "abc12345").
```

Delegated failure envelope:

```text
ERROR: adw spec write failed (exit N)
```

Routing hint:

- Keep `adw_spec_write` for mutating operations only; use `adw_spec_read` for read/list paths to preserve least-privilege wrapper boundaries.
