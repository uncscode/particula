# ADW Plans Tool Reference

Full parameter reference for the adw_plans tool. For quick usage, see the tool description.

## Commands

| Command             | Required params                | Description                           |
|---------------------|--------------------------------|---------------------------------------|
| `list`              | —                              | List plans with optional filters      |
| `show`              | `plan_id`                      | Show plan details                     |
| `create`            | `plan_type`, `title`           | Create a new plan                     |
| `update`            | `plan_id`                      | Update plan metadata                  |
| `add-phase`         | `plan_id`, `title`             | Add phase to a plan                   |
| `update-phase`      | `plan_id`, `phase_id`          | Update phase metadata                 |
| `validate`          | —                              | Validate all plans                    |
| `schema`            | —                              | Show/check plan schema                |
| `scaffold-sections` | `plan_id`, `plan_type`         | Copy section templates for a plan     |
| `list-sections`     | `plan_id`                      | List section files for a plan         |

## Simple Examples

```jsonc
// List all active plans
{ "command": "list", "lifecycle": "active", "json": true }

// List features for an epic
{ "command": "list", "plan_type": "feature", "parent": "E17", "json": true }

// Show a plan
{ "command": "show", "plan_id": "E17-F1", "json": true }

// Create a feature plan (after resolving worktree_path from ADW state)
{ "command": "create", "plan_type": "feature", "title": "Add Auth Module", "parent": "E17", "cwd": "/path/to/trees/<adw_id>" }

// Update plan status
{ "command": "update", "plan_id": "E17-F8", "status": "Ready", "cwd": "/path/to/trees/<adw_id>" }

// Add a phase
{ "command": "add-phase", "plan_id": "E17-F1", "title": "Core implementation", "size": "M", "cwd": "/path/to/trees/<adw_id>" }

// Mark phase as shipped
{ "command": "update-phase", "plan_id": "E17-F8", "phase_id": "E17-F8-P1", "phase_status": "Shipped", "cwd": "/path/to/trees/<adw_id>" }
```

## Advanced Examples

```jsonc
// List sections for a plan
{ "command": "list-sections", "plan_id": "M25", "cwd": "/path/to/trees/<adw_id>" }

// List sections with auto-populate
{ "command": "list-sections", "plan_id": "M25", "populate": true, "cwd": "/path/to/trees/<adw_id>" }

// Scaffold section templates
{ "command": "scaffold-sections", "plan_id": "E17-F1", "plan_type": "feature", "cwd": "/path/to/trees/<adw_id>" }

// Update phase with patch
{ "command": "update-phase", "plan_id": "E17-F8", "phase_id": "E17-F8-P1", "patch": "{\"status\":\"Shipped\"}", "cwd": "/path/to/trees/<adw_id>" }

// Update plan with patch
{ "command": "update", "plan_id": "E17-F8", "patch": "{\"notes\":\"Updated\"}", "cwd": "/path/to/trees/<adw_id>" }

// Link issue to phase
{ "command": "update-phase", "plan_id": "E17-F1", "phase_id": "E17-F1-P1", "issue_number": 42, "cwd": "/path/to/trees/<adw_id>" }

// Clear issue link
{ "command": "update-phase", "plan_id": "E17-F1", "phase_id": "E17-F1-P1", "clear_issue_number": true, "cwd": "/path/to/trees/<adw_id>" }

// Validate all plans
{ "command": "validate" }

// Check schema
{ "command": "schema", "check": true }

// Filter by status
{ "command": "list", "plan_type": "feature", "status": "In Progress", "json": true }
```

## Parameter Reference

### Core

| Parameter  | Type   | Default | Description                              |
|------------|--------|---------|------------------------------------------|
| `command`  | enum   | —       | Plans command to execute (required)      |
| `plan_id`  | string | —       | Plan identifier (e.g., E17-F1)           |
| `plan_type`| string | —       | Runtime registry-driven plan type (for example epic, feature, maintenance, research) |
| `cwd`      | string | —       | Repository/worktree root path            |

### List Filters

| Parameter   | Type    | Description                              |
|-------------|---------|------------------------------------------|
| `lifecycle` | enum    | active, completed, or closed             |
| `parent`    | string  | Parent plan ID filter                    |
| `status`    | enum    | Plan status filter                       |
| `json`      | boolean | JSON output format                       |

### Create/Update

| Parameter  | Type   | Description                              |
|------------|--------|------------------------------------------|
| `title`    | string | Plan title (required for create)         |
| `priority` | enum   | P0, P1, P2, P3, or Backlog              |
| `size`     | enum   | XS, S, M, L, XL, or XXL                 |
| `status`   | enum   | Draft, Proposed, Ready, In Progress, etc.|
| `patch`    | string | Raw JSON patch payload                   |

### Phase Operations

| Parameter            | Type    | Description                         |
|----------------------|---------|-------------------------------------|
| `phase_id`           | string  | Phase identifier (e.g., E17-F1-P1) |
| `phase_status`       | enum    | Not Started, In Progress, Blocked, Shipped, Cancelled |
| `after`              | string  | Insert phase after this phase ID    |
| `issue_number`       | number  | Link GitHub issue to phase          |
| `clear_issue_number` | boolean | Clear linked issue number           |

### Schema/Sections

| Parameter  | Type    | Description                              |
|------------|---------|------------------------------------------|
| `check`    | boolean | Run schema check (schema command)        |
| `populate` | boolean | Persist section paths (list-sections)    |

## CWD Requirement

**Mutating commands** (`create`, `update`, `add-phase`, `update-phase`, `scaffold-sections`) **require `cwd`** to anchor writes to the correct worktree. Omitting `cwd` on these commands returns a deterministic error.

Resolve `worktree_path` from ADW state first, then pass that exact value as `cwd`:

```jsonc
// 1) Read the isolated worktree path
{ "command": "read", "adw_id": "abc12345", "field": "worktree_path" }

// 2) Pass that value into adw_plans mutating commands
{ "command": "update", "plan_id": "E17-F8", "status": "Ready", "cwd": "/path/to/trees/abc12345" }
```

Read-only commands (`list`, `show`, `validate`, `schema`, `list-sections`) accept optional `cwd`.

### Deterministic pre-spawn examples

```text
ERROR: update command requires 'cwd'.
ERROR: cwd path does not exist: <path>
ERROR: cwd path is not a directory: <path>
ERROR: cwd path resolves outside repository root: <path> (canonical: <path>)
```

## Status Values

### Plan Status
Draft, Proposed, Ready, In Progress, Blocked, Monitoring, Shipped, Cancelled, Superseded

### Phase Status
Not Started, In Progress, Blocked, Shipped, Cancelled

## Patch Semantics

- Explicit update flags override overlapping patch keys
- Blocked identity keys: `id`, `type`, `parent_id` (plan updates); `id`, `plan_id`, `issue_number` (phase updates)
- Maximum patch size: 64KB UTF-8

## Auto-Derived Fields

- `update`: sets `last_updated` on successful writes
- `update-phase`: may derive `start_date` (In Progress) and `completion_date` (Shipped)

## Failure Envelope + Routing Hint

- Delegated/subprocess failure envelope is deterministic:
  - `ERROR: adw plans <cmd> failed (exit N).`
  - diagnostic precedence: `stderr` -> `stdout` -> message/fallback
  - diagnostics are bounded with deterministic truncation for long output
  - absolute filesystem paths are redacted to `<path>` across compatibility and split wrappers
  - recognized runtime/tooling and cwd/worktree failures append targeted recovery hints without changing success-path output or preflight cwd wording
- Routing hint:
  - Prefer `adw_plans_read` for read-only operations.
  - Prefer `adw_plans_mutate` for mutating operations requiring `cwd`.
  - Keep `adw_plans` for compatibility/unified flows.
