# Dev-Plan Manager - Usage Guide

## Overview

The Dev-Plan Manager is an **interactive primary agent** for creating and managing development plans in `adw-docs/dev-plans/`. It guides users through clarifying requirements before generating documents, ensuring consistent formatting and proper indexing.

## When to Use

- **Creating new plans**: Epics, features, or maintenance documents
- **Updating existing plans**: Status changes, phase updates, content modifications
- **Moving completed plans**: Transitioning shipped work to `completed/` folders
- **Discussing requirements**: Clarifying scope before committing to a plan structure

## Agent Structure

| Agent | Type | Purpose |
|-------|------|---------|
| `dev-plan-manager` | Primary | Interactive orchestrator, asks clarifying questions |
| `dev-plan-creator` | Subagent | Creates new documents from templates |
| `dev-plan-updater` | Subagent | Updates existing documents |
| `dev-plan-indexer` | Subagent | Maintains index files, handles moves |

## Tool Configuration

| Tool | Enabled | Rationale |
|------|---------|-----------|
| `read` | ✅ | Read existing plans and indexes |
| `edit` | ✅ | Update existing documents |
| `write` | ✅ | Create new documents |
| `list` | ✅ | Directory navigation |
| `glob` | ✅ | File discovery |
| `grep` | ✅ | Content search |
| `todoread` | ✅ | Task tracking |
| `todowrite` | ✅ | Task tracking |
| `task` | ✅ | Invoke subagents |
| `get_datetime` | ✅ | Timestamps for metadata (UTC or America/Denver via `localtime`) |
| `adw_spec` | ❌ | No workflow state interaction |
| Atomic git wrappers | ❌ | User handles commits |
| `bash` | ❌ | Always disabled |

## ID Naming Convention

The agent enforces hierarchical IDs for GitHub issue traceability:

### Epic IDs
- Format: `E{n}` (E1, E2, E3...)
- Phase: `E{n}-P{m}` (E1-P1, E1-P2...)

### Feature IDs
- Epic-linked: `E{n}-F{m}` (E1-F1, E1-F2...)
- Standalone: `F{n}` (F1, F2...)
- Phase: `E1-F2-P3` or `F1-P2`

### Maintenance IDs
- Epic-linked: `E{n}-M{m}` (E1-M1, E1-M2...)
- Standalone: `M{n}` (M1, M2...)
- Phase: `E1-M1-P2` or `M1-P1`

## Usage Examples

### Example 1: Create New Feature Plan

**Context**: You want to add a rate limiting feature linked to Epic E1.

**Invocation**:
```
I need to create a new feature plan for rate limiting API calls
```

**Agent Behavior**:
1. Asks clarifying questions:
   - Is this linked to an existing epic?
   - What problem does it solve?
   - How many phases do you anticipate?
   - What are the success metrics?

2. After clarification, creates:
   - `features/E1-F5-rate-limiting.md` (or next available ID)
   - Updates `features/index.md` with new entry

### Example 2: Update Plan Status

**Context**: You've completed implementing E1-F2.

**Invocation**:
```
Update E1-F2 status to Shipped
```

**Agent Behavior**:
1. Asks for confirmation:
   - Should I update the Last Updated date?
   - Any phases to mark complete?
   - Notes for the change log?

2. After confirmation:
   - Updates status to "Shipped"
   - Updates metadata dates
   - Adds change log entry

### Example 3: Move to Completed

**Context**: E1-F1 is fully shipped and you want to archive it.

**Invocation**:
```
Move E1-F1 to completed
```

**Agent Behavior**:
1. Updates document:
   - Sets status to "Shipped"
   - Adds completion date

2. Moves file:
   - From: `features/E1-F1-platform-abstraction.md`
   - To: `features/completed/E1-F1-platform-abstraction.md`

3. Updates `features/index.md`:
   - Removes from Active Plans
   - Adds to Completed Plans

### Example 4: Create New Epic

**Context**: You're starting a major authentication refactoring initiative.

**Invocation**:
```
I want to create an epic for refactoring authentication
```

**Agent Behavior**:
1. Asks comprehensive questions:
   - Vision and end goal
   - Current pain points
   - Scope (in/out)
   - Expected child features
   - Timeline and dependencies
   - Success metrics

2. After clarification:
   - Creates `epics/E5-auth-refactoring.md` (next available ID)
   - Updates `epics/index.md`
   - Suggests creating child feature plans

## Folder Structure

```
adw-docs/dev-plans/
├── README.md                    # Overview, links to indexes
├── template-*.md                # Templates for each type
├── epics/
│   ├── index.md                 # Next ID: E5
│   ├── E1-multi-platform.md
│   └── completed/
├── features/
│   ├── index.md                 # Next ID: F1 / E{n}-F{m}
│   ├── E1-F1-platform-abstraction.md
│   └── completed/
└── maintenance/
    ├── index.md                 # Next ID: M1 / E{n}-M{m}
    ├── M1-quarterly-audit.md
    └── completed/
```

## Best Practices

1. **Always start with clarification** - Don't skip to generation
2. **Use the correct ID format** - Check index.md for next available
3. **Link to parent epics** - Features/maintenance should reference their epic
4. **Include "Update dev-docs"** - Always as the final phase
5. **Update indexes** - After any document change

## Limitations

- **Cannot delete files** - Reports files for manual deletion after moves
- **Cannot modify IDs** - Existing document IDs are immutable
- **No git operations** - User handles staging and commits
- **Read-only templates** - Suggests changes but doesn't modify templates

## Integration with Other Agents

- **codebase-researcher**: Can be invoked for complex plans needing code context
- **documentation agents**: Separate from doc-feature, doc-maintenance agents
- **workflow agents**: Independent of ADW workflow state

## Troubleshooting

### Issue: Duplicate ID Detected
**Solution**: The indexer will report the conflict and suggest the correct next ID.

### Issue: Parent Epic Not Found
**Solution**: Agent will list available epics and ask you to confirm or create a new one first.

### Issue: Index Out of Sync
**Solution**: Invoke the indexer to scan existing files and rebuild the index.

## See Also

- [Development Plans README](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/README.md)
- [Template: Epic](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/template-epic.md)
- [Template: Feature](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/template-feature.md)
- [Template: Maintenance](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/template-maintenance.md)
