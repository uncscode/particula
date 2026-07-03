# ADW Plans Tool Reference (Retired Compatibility Wrapper)

This document is retained as historical reference for the retired `adw_plans`
 compatibility wrapper. Active integrations should use:

- `adw_plans_read`
- `adw_plans_mutate`

The compatibility wrapper implementation has been retired from the live tool
 tree and archived under `.trash/`.

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
{ "command": "list", "lifecycle": "active", "options": "json" }

// List features for an epic
{ "command": "list", "plan_type": "feature", "parent": "E17", "options": "json" }

// Show a plan
{ "command": "show", "plan_id": "E17-F1", "options": "json" }

// adw_spec_read: get worktree_path from ADW state, then reuse that exact value as cwd
adw_spec_read({ "command": "read", "adw_id": "abc12345", "field": "worktree_path" })

// Create a feature plan
{ "command": "create", "plan_type": "feature", "title": "Add Auth Module", "parent": "E17", "cwd": "/path/to/trees/abc12345" }

// Historical compatibility example: direct status on retired unified wrapper
{ "command": "update", "plan_id": "E17-F8", "status": "Ready", "cwd": "/path/to/trees/abc12345" }

// Add a phase
{ "command": "add-phase", "plan_id": "E17-F1", "title": "Core implementation", "options": "size=M", "cwd": "/path/to/trees/abc12345" }

// Historical compatibility example: direct phase_status on retired unified wrapper
{ "command": "update-phase", "plan_id": "E17-F8", "phase_id": "E17-F8-P1", "phase_status": "Shipped", "cwd": "/path/to/trees/abc12345" }
```

Retirement note: the direct `status` and `phase_status` fields above were part
of the retired unified wrapper. Active guidance uses the split-wrapper shape
shown below, where optional aliases move through bounded `options` tokens.

## Preferred Split-Wrapper Shape

```jsonc
// Read-only path: use adw_plans_read with bounded options tokens
adw_plans_read({ "command": "list", "plan_type": "feature", "options": "status=In Progress json" })

// Mutating path: use adw_plans_mutate and keep cwd/patch as direct exceptions
adw_plans_mutate({ "command": "update", "plan_id": "E17-F8", "options": "status=Ready priority=P1", "cwd": "/path/to/trees/abc12345" })

// Phase update: phase-status also moves through options on split wrappers
adw_plans_mutate({ "command": "update-phase", "plan_id": "E17-F8", "phase_id": "E17-F8-P1", "options": "phase-status=Shipped issue=42", "patch": "{\"actuals\":\"wrapper docs aligned\"}", "cwd": "/path/to/trees/abc12345" })
```

## Advanced Examples

```jsonc
// List sections for a plan
{ "command": "list-sections", "plan_id": "M25", "cwd": "/path/to/trees/abc12345" }

// List sections with auto-populate
{ "command": "list-sections", "plan_id": "M25", "options": "populate json", "cwd": "/path/to/trees/abc12345" }

// Scaffold section templates
{ "command": "scaffold-sections", "plan_id": "E17-F1", "plan_type": "feature", "cwd": "/path/to/trees/abc12345" }

// Update phase with bounded options plus direct patch
{ "command": "update-phase", "plan_id": "E17-F8", "phase_id": "E17-F8-P1", "options": "phase-status=Blocked size=M issue=42", "patch": "{\"actuals\":\"split-wrapper parity complete\"}", "cwd": "/path/to/trees/abc12345" }

// Update plan with patch
{ "command": "update", "plan_id": "E17-F8", "options": "priority=P1", "patch": "{\"notes\":\"Updated\"}", "cwd": "/path/to/trees/abc12345" }

// Link issue to phase
{ "command": "update-phase", "plan_id": "E17-F1", "phase_id": "E17-F1-P1", "options": "issue=42", "cwd": "/path/to/trees/abc12345" }

// Clear issue link
{ "command": "update-phase", "plan_id": "E17-F1", "phase_id": "E17-F1-P1", "options": "clear-issue-number", "cwd": "/path/to/trees/abc12345" }

// Validate all plans
{ "command": "validate" }

// Check schema
{ "command": "schema", "options": "check" }

// Filter by status
{ "command": "list", "plan_type": "feature", "options": "status=In Progress json" }
```

Preferred guidance for new calls: use bounded `options` tokens for optional
wrapper aliases such as `json`, `check`, `populate`, `status=<value>`,
`phase-status=<value>`, `priority=<value>`, `size=<value>`,
`after=<phase_id>`, `issue=<n>`, and `clear-issue-number`. Keep direct fields
for required identifiers plus the payload-bearing direct exceptions `cwd` and
raw JSON `patch`. Split wrappers route `status` / `phase-status` aliases
through `options`.

## Parameter Reference

### Core

| Parameter  | Type   | Default | Description                              |
|------------|--------|---------|------------------------------------------|
| `command`  | enum   | —       | Plans command to execute (required)      |
| `plan_id`  | string | —       | Plan identifier (e.g., E17-F1)           |
| `plan_type`| string | —       | Runtime registry-driven plan type (for example epic, feature, maintenance, research) |
| `cwd`      | string | —       | Repository/worktree root path            |
| `options`  | string | —       | Bounded command-scoped wrapper tokens such as `json`, `check`, `populate`, `status=<value>`, `phase-status=<value>`, `priority=<value>`, `size=<value>`, `after=<phase_id>`, `issue=<n>`, or `clear-issue-number` |

`options` is a wrapper convenience field, not a raw CLI passthrough. Use it for
supported optional tokens only. Keep direct fields for required identifiers and raw JSON
`patch`.

### List Filters

| Parameter   | Type    | Description                              |
|-------------|---------|------------------------------------------|
| `lifecycle` | enum    | active, completed, or closed             |
| `parent`    | string  | Parent plan ID filter                    |
| `status`    | enum    | Plan status filter                       |

### Create/Update

| Parameter  | Type   | Description                              |
|------------|--------|------------------------------------------|
| `title`    | string | Plan title (required for create)         |
| `status`   | enum   | Draft, Proposed, Ready, In Progress, etc.; historical unified-wrapper alias |
| `patch`    | string | Raw JSON patch payload; keep this as a direct field |

### Phase Operations

| Parameter            | Type    | Description                         |
|----------------------|---------|-------------------------------------|
| `phase_id`           | string  | Phase identifier (e.g., E17-F1-P1) |
| `phase_status`       | enum    | Not Started, In Progress, Blocked, Shipped, Cancelled; historical unified-wrapper alias |

### Schema/Sections

Use `options: "check"` for schema checks and `options: "populate"` for
`list-sections` population.

## CWD Requirement

**Mutating commands** (`create`, `update`, `add-phase`, `update-phase`, `scaffold-sections`) **require `cwd`** to anchor writes to the correct worktree. Omitting `cwd` on these commands returns a deterministic error.

Resolve `worktree_path` from ADW state first, then pass that exact value as `cwd`:

```jsonc
// 1) adw_spec_read: get the isolated worktree path
adw_spec_read({ "command": "read", "adw_id": "abc12345", "field": "worktree_path" })

// 2) Pass that value into adw_plans mutating commands
{ "command": "update", "plan_id": "E17-F8", "options": "status=Ready priority=P1", "cwd": "/path/to/trees/abc12345" }
```

Read-only commands (`list`, `show`, `validate`, `schema`, `list-sections`) accept optional `cwd`.

### Deterministic pre-spawn examples

```text
ERROR: update command requires 'cwd'.
ERROR: cwd path does not exist: <path>
ERROR: cwd path is not a directory: <path>
ERROR: cwd path is not a repository/worktree root: <path> (missing .git metadata at <path>)
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
  - Use `adw_plans_read` and `adw_plans_mutate` for all active flows.
