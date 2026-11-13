# feedback_log

Structured write-mode wrapper for workflow/tool feedback logging.

## Wrapper scope

- The OpenCode wrapper exposes write-mode logging only.
- Backend read mode stays available through:
  `python3 .opencode/tools/feedback_log.py --command read`

## Required fields

- `category`: `bug | feature | friction | performance`
- `severity`: `low | medium | high | critical`
- `description`
- `workflowStep`
- `agentType`
- `adwId`

Required string fields must be non-empty after trimming.

## Optional fields

- `suggestedFix`
- `toolName`
- `context`

Blank optional strings are omitted from CLI argv.

## Failure behavior

- Invalid enums fail closed with deterministic `ERROR:` messages.
- Missing required fields fail closed before subprocess execution.
- Execution failures prefer `stdout`, then `stderr`, then the subprocess message.
- `stderr` failures include the subprocess exit code when available.

## Example

```json
{
  "category": "friction",
  "severity": "medium",
  "description": "Tool output required multiple retries.",
  "workflowStep": "build",
  "agentType": "adw-build",
  "adwId": "fed63dec",
  "toolName": "feedback_log"
}
```

## Notes

- Write success is verified by the backend before success is reported.
- Fallback JSONL logging applies the same repo-confinement validation to the
  lock path and the log path, then verifies the appended record from the log tail
  before reporting success.
- AGENTS.md remains the canonical operator note for backend read-mode usage,
  paging/filter details, fallback logging, and rate-limit policy.
