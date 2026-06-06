# adw_service Tool Reference

Narrow ADW wrapper for service lifecycle operations.

## Supported Commands

| Command  | Description |
|----------|-------------|
| `launch` | Start ADW service runtime |
| `stop`   | Stop ADW service runtime |

## Parameters

| Parameter     | Type                     | Applies to | Notes |
|---------------|--------------------------|------------|-------|
| `command`     | `"launch" | "stop"`    | both       | Required |
| `mode`        | `"local" | "remote"`   | launch     | Optional |
| `background`  | `boolean`                | launch     | Optional; emits `--background` only when `true` |
| `force`       | `boolean`                | stop       | Optional; emits `--force` only when `true` |
| `help`        | `boolean`                | both       | Optional; emits `--help` |

## Validation Rules

- `force` is rejected for `launch`.
- `mode` and `background` are rejected for `stop`.
- `help: true` is an explicit bypass path and short-circuits command-specific option validation.
- Wrapper is bounded and does not expose free-form passthrough args.

## Examples

```jsonc
{ "command": "launch" }
{ "command": "launch", "mode": "local" }
{ "command": "launch", "background": true }
{ "command": "stop" }
{ "command": "stop", "force": true }
{ "command": "launch", "help": true }
```

## Contracts

- Uses shared execution envelope via `executeAdwCommand`.
- Timeout uses shared default command timeout policy.
- Error precedence is deterministic: `stderr` -> `stdout` -> fallback.
