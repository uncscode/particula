# adw_spec_messages

Focused ADW spec wrapper for workflow messages.

## Commands
- `messages-write`
- `messages-read`

## Behavior
- Validates `adw_id` fail-closed
  - missing, blank, null, and non-string values fail as required-input errors before format validation or spawn
  - malformed non-blank strings still fail the 8-character-hex validation path
- `messages-write` requires non-empty `agent` and `message`
- trimmed `agent` and `message` values are what get forwarded downstream
- `messages-read` validates `last` as integer in `0..50`
  - `last = 0` omits `--last`
  - `last > 0` includes `--last <n>`
- `raw` is supported for `messages-read`

## Contract Note
- Success is always envelope-based for this wrapper:
  `ADW Spec Command: <command>\n\n<stdout>`.
- Failures are deterministic:
  - `ERROR: adw spec <command> failed (exit N)` on non-zero exits.
  - `ERROR: Failed to execute adw spec <command>. ...` on execution errors.
- Execution-error diagnostic precedence remains `stderr` -> `stdout` -> message/fallback.
- Failure diagnostics are bounded, strip control characters, and redact absolute filesystem paths to `<path>`.

## Failure and Recovery Examples

Pre-spawn validation (messages-write required args):

```text
ERROR: 'agent' parameter is required for messages-write command.
```

Required-input parity example:

```text
ERROR: 'adw_id' parameter is required for all spec commands.

Usage: Use the ADW tool with command "status" to see active workflow IDs.
Example: { command: "messages-write", adw_id: "abc12345", agent: "planner", message: "Done." }
```

Malformed-but-nonblank identifier example:

```text
ERROR: 'adw_id' must be an 8-character hex string (e.g., "abc12345").
```

Delegated failure envelope:

```text
ERROR: adw spec messages-read failed (exit N)
```

Routing hint:

- Use this wrapper only for `messages-write`/`messages-read`; route state reads/writes to `adw_spec_read`/`adw_spec_write`.
