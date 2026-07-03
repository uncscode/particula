# adw_spec_read

Focused ADW spec wrapper for read-oriented operations.

## Commands
- `list`
- `read`

## Behavior
- Validates `adw_id` fail-closed
  - missing, blank, null, and non-string values fail as required-input errors before format validation or spawn
  - malformed non-blank strings still fail the 8-character-hex validation path
- `read` supports optional `field` plus bounded `options: "raw"`
- `list` supports bounded `options: "json"`
- Sparse option behavior: blank `field` is omitted, and non-blank `field` is trimmed before forwarding

## Examples

```json
{ "command": "read", "adw_id": "abc12345" }
```

```json
{ "command": "read", "adw_id": "abc12345", "field": "worktree_path", "options": "raw" }
```

```json
{ "command": "list", "adw_id": "abc12345", "options": "json" }
```

## Contract Note
- `read` returns raw field content on success (including content that may begin with `ERROR:`).
- Present fields whose value is `null` are treated as successful reads, not missing-field failures.
- `read --field ... --options raw` forwards delegated `null` output as payload (for example `null\n`).
- Non-raw present-null reads still succeed and return the delegated field/value rendering.
- Absent fields still fail through the deterministic delegated error envelope.
- `list` returns envelope format: `ADW Spec Command: list\n\n<stdout>`.
- Failures remain deterministic: `ERROR: adw spec <command> failed (exit N)` for non-zero exits,
  and `ERROR: Failed to execute adw spec <command>. ...` for execution errors.
- Execution-error diagnostic precedence remains `stderr` -> `stdout` -> message/fallback.
- Failure diagnostics are bounded, strip control characters, and redact absolute filesystem paths to `<path>`.

## Failure and Recovery Examples

Pre-spawn validation (required argument):

```text
ERROR: 'adw_id' parameter is required for all spec commands.

Usage: Use the ADW tool with command "status" to see active workflow IDs.
Example: { command: "read", adw_id: "abc12345" }
```

Malformed-but-nonblank identifier example:

```text
ERROR: 'adw_id' must be an 8-character hex string (e.g., "abc12345").
```

Delegated failure envelope:

```text
ERROR: adw spec read failed (exit N)
```

Routing hint:

- Use `adw_spec_read` for read/list only; switch to `adw_spec_write` for mutations and `adw_spec_messages` for message I/O.

Practical guardrail:

- Classify `read` results by command context first. A non-empty payload that
  starts with `ERROR:` can still be a successful `read` result.
