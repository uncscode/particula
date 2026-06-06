# ADW Spec Tool Reference

Full parameter reference for the adw_spec tool. For quick usage, see the tool description.

## Commands

| Command          | Required params        | Description                           |
|------------------|------------------------|---------------------------------------|
| `list`           | `adw_id`               | List all fields in adw_state.json     |
| `read`           | `adw_id`               | Read field (default: spec_content)    |
| `write`          | `adw_id`, content/file | Write to field (default: spec_content)|
| `delete`         | `adw_id`, `field`      | Delete a field                        |
| `messages-write` | `adw_id`, `agent`, `message` | Write workflow message          |
| `messages-read`  | `adw_id`               | Read workflow messages                |

## Simple Examples

```jsonc
// Read current spec
{ "command": "read", "adw_id": "abc12345" }

// Read specific field
{ "command": "read", "adw_id": "abc12345", "field": "plan_file" }

// Write spec content
{ "command": "write", "adw_id": "abc12345", "content": "# Implementation Plan\n\n## Step 1\n..." }

// Append to spec
{ "command": "write", "adw_id": "abc12345", "content": "\n\nNotes", "append": true }

// Write from file
{ "command": "write", "adw_id": "abc12345", "file": "plan.md" }

// List all fields
{ "command": "list", "adw_id": "abc12345", "json": true }

// Write a workflow message
{ "command": "messages-write", "adw_id": "abc12345", "agent": "planner", "message": "Done." }

// Read recent messages
{ "command": "messages-read", "adw_id": "abc12345", "last": 3, "raw": true }
```

## Advanced Examples

```jsonc
// Read raw content (no formatting)
{ "command": "read", "adw_id": "abc12345", "raw": true }

// Delete a custom field
{ "command": "delete", "adw_id": "abc12345", "field": "custom_field", "confirm": true }

// Write to a specific field
{ "command": "write", "adw_id": "abc12345", "field": "plan_file", "content": "path/to/plan.md" }

// Read all messages
{ "command": "messages-read", "adw_id": "abc12345", "raw": true }
```

## Parameter Reference

### Core

| Parameter | Type   | Default        | Description                              |
|-----------|--------|----------------|------------------------------------------|
| `command` | enum   | —              | Spec command to execute (required)       |
| `adw_id`  | string | —              | ADW workflow ID, 8-char hex (required)   |

### Read/Write

| Parameter | Type    | Default          | Description                            |
|-----------|---------|------------------|----------------------------------------|
| `field`   | string  | "spec_content"   | Field to read/write/delete             |
| `content` | string  | —                | Content to write                       |
| `file`    | string  | —                | File path to read content from         |
| `append`  | boolean | false            | Append instead of replace              |
| `raw`     | boolean | false            | Raw output without formatting (read)   |
| `json`    | boolean | false            | JSON output (list command)             |
| `confirm` | boolean | false            | Skip confirmation (delete command)     |

### Messages

| Parameter | Type   | Default | Description                              |
|-----------|--------|---------|------------------------------------------|
| `agent`   | string | —       | Agent name (required for messages-write) |
| `message` | string | —       | Message text (required for messages-write)|
| `last`    | number | all     | Return last N messages (0 = all, max 50) |

## Common Fields

| Field             | Description                                 |
|-------------------|---------------------------------------------|
| `spec_content`    | Main implementation specification/plan      |
| `branch_name`     | Git branch name for the workflow            |
| `workflow_type`   | Type: complete, patch, document, generate   |
| `model_tier`      | AI model tier: light, base, heavy           |
| `current_workflow` | Current workflow name                      |
| `current_step`    | Current step in workflow                    |
| `pr_url`          | Pull request URL (if created)               |
| `pr_number`       | Pull request number (if created)            |

Use `list` command to see all available fields for a workflow.

## Response Contracts

### Read Command
- Non-empty field: returns raw field content
- Empty/missing field: returns `""`
- Failure: returns `ERROR: adw spec read failed (exit N)`

### Legacy `adw_spec` vs Split Wrapper Contracts

- Legacy `adw_spec` is command-sensitive:
  - `read`: returns raw field content (including `""` for empty/missing/null-like successful reads).
  - non-`read`: returns `ADW Spec Command: <command>\n\n<stdout>`.
- Split wrappers preserve the same behavior for their scoped commands:
  - `adw_spec_read` returns raw payload for `read` and envelope for `list`.
  - `adw_spec_write` and `adw_spec_messages` return command envelopes on success.
- `write --file` is canonicalized and constrained to existing repository-confined paths across both compatibility and split wrappers.
- Trimmed validated strings are what get forwarded downstream for `field`, `agent`, `message`, and normalized `adw_id` values.
- Required `adw_id` handling is aligned across compatibility and split surfaces:
  - missing, blank, null, and non-string values fail as required-input errors before hex validation or subprocess spawn
  - malformed non-blank strings still fail the 8-character-hex validation path
- Success-path parsing and failure-path parsing should stay separate:
  - `read` success may be any raw string payload, including content beginning with `ERROR:`
  - failure envelopes remain deterministic for all commands and should be classified by command context first (`read` vs non-`read`)

### Other Commands
- Success: `ADW Spec Command: <command>\n\n<stdout>`
- Failure on non-zero exit: `ERROR: adw spec <command> failed (exit N)`
- Execution-error envelopes remain deterministic across compatibility and split wrappers:
  `ERROR: Failed to execute adw spec <command>. ...`
- Execution-error diagnostic precedence remains `stderr` -> `stdout` -> message/fallback.
- Failure diagnostics are bounded, strip control characters, and redact absolute filesystem paths to `<path>`.

## Deterministic Failure Examples

### Pre-spawn validation failure

```text
ERROR: 'adw_id' parameter is required for all spec commands.

Usage: Use the ADW tool with command "status" to see active workflow IDs.
Example: { command: "read", adw_id: "abc12345" }
```

Malformed-but-nonblank identifier example:

```text
ERROR: 'adw_id' must be an 8-character hex string (e.g., "abc12345").
```

### Delegated/subprocess failure envelope

```text
ERROR: adw spec read failed (exit 1)
Warning: Field 'issue_title' not found or is null
```

### Recovery / routing hint

- Prefer split wrappers for new integrations:
  - `adw_spec_read` for `list`/`read`
  - `adw_spec_write` for `write`/`delete`
  - `adw_spec_messages` for workflow message operations
- For automated classification, use command context first (`read` vs non-`read`) and then apply deterministic failure envelope handling (`stderr` -> `stdout` -> message/fallback).

Practical guardrail: classify responses by command context (`read` vs non-`read`).
For `read`, treat `result === ""` as no-content and any non-empty string as payload.
