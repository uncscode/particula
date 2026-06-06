# ADW CLI Tool Reference

Full parameter reference for the adw tool. For quick usage, see the tool description.

## Command Categories

### Workflow Commands (require issue_number)

| Command     | Description                                        |
|-------------|-----------------------------------------------------|
| `complete`  | Full workflow: plan -> build -> test -> review -> document -> ship |
| `patch`     | Quick workflow: plan -> build -> ship (skips validation) |
| `plan`      | Generate implementation plan only                   |
| `build`     | Execute implementation from plan                    |
| `test`      | Run tests and validation                            |
| `review`    | Code review and quality checks                      |
| `document`  | Generate documentation                              |
| `ship`      | Push changes and create pull request                |

### Issue Commands

| Command           | Required params  | Description                        |
|-------------------|------------------|------------------------------------|
| `create-issue`    | `title`, `body`  | Create new GitHub issue            |
| `interpret-issue` | `text` or `issue_number` | Convert text to structured issue |

### Setup Commands (require args)

| Args                                  | Description                    |
|---------------------------------------|--------------------------------|
| `["env"]`                             | Run environment wizard         |
| `["validate"]`                        | Validate environment config    |
| `["check"]`                           | Run preflight checks           |
| `["template", "init"]`               | Initialize template manifest   |
| `["template", "apply"]`              | Apply templates to project     |
| `["template", "extract", "--diff"]`  | Show drift between docs/templates |
| `["template", "validate"]`           | Validate placeholders          |
| `["template", "token", "list"]`      | List all keyword tokens        |

### System Commands

| Command       | Description                    |
|---------------|--------------------------------|
| `status`      | Show active ADW workflows      |
| `health`      | Run system health diagnostics  |
| `init`        | Initialize ADW configuration   |
| `maintenance` | Run maintenance tasks          |
| `docstring`   | Update docstrings for files    |
| `finalize-docs` | Finalize living documentation |

## Simple Examples

```jsonc
// Run full workflow
{ "command": "complete", "issue_number": 123, "model": "base" }

// Quick patch
{ "command": "patch", "issue_number": 456 }

// Resume existing workflow
{ "command": "build", "issue_number": 123, "adw_id": "a1b2c3d4" }

// Use heavy model for complex work
{ "command": "complete", "issue_number": 789, "model": "heavy" }

// Check active workflows
{ "command": "status" }

// System health check
{ "command": "health" }

// Create issue
{ "command": "create-issue", "title": "Add feature", "body": "## Description\n..." }

// Interpret text as issue
{ "command": "interpret-issue", "text": "Add tests for auth module" }
```

## Advanced Examples

```jsonc
// Setup commands
{ "command": "setup", "args": ["env"] }
{ "command": "setup", "args": ["template", "extract", "--diff"] }
{ "command": "setup", "args": ["template", "validate"] }
{ "command": "setup", "args": ["template", "token", "list"] }

// Get help for any command
{ "command": "complete", "help": true }

// Dry run
{ "command": "complete", "issue_number": 123, "args": ["--dry-run"] }
```

## Parameter Reference

### Core

| Parameter      | Type   | Default | Description                              |
|----------------|--------|---------|------------------------------------------|
| `command`      | enum   | —       | ADW command to execute (required)        |
| `issue_number` | number | —       | GitHub issue number (required for workflows) |
| `adw_id`       | string | —       | ADW ID to resume workflow (8-char hex)   |
| `model`        | enum   | "base"  | Model tier: "base" (Sonnet) or "heavy" (Opus) |

### Issue Creation

| Parameter | Type   | Description                              |
|-----------|--------|------------------------------------------|
| `title`   | string | Issue title (required for create-issue)  |
| `body`    | string | Issue body, markdown (required for create-issue) |
| `text`    | string | Free-form text (for interpret-issue)     |

### Flags

| Parameter | Type    | Description                              |
|-----------|---------|------------------------------------------|
| `help`    | boolean | Show CLI help instead of executing       |
| `args`    | array   | Additional CLI arguments (strings)       |

## Model Selection

| Tier    | Models  | When to use                              |
|---------|---------|------------------------------------------|
| `base`  | Sonnet  | Most tasks (faster, cost-effective)      |
| `heavy` | Opus    | Complex architecture, difficult bugs, large refactors |

## Protected Flags

The `args` parameter rejects these flags to prevent conflicts with dedicated parameters:
`--adw-id`, `--model`, `--help`, `--title`, `--body`, `--text`, `--source-issue`

Use the dedicated tool parameters instead.
