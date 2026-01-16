---
description: "Generate type:generate issues for all features in an epic plan"
---

# /epic-to-issues

Generate GitHub issues for all features defined in an epic plan. Each feature becomes
a `type:generate` issue that, when processed, spawns implementation issues for that
feature's phases.

## Usage

```bash
/epic-to-issues <epic-id> [--dry-run] [--feature <feature-id>]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<epic-id>` | Yes | Epic identifier (e.g., `E1`, `E3`, `E9`) |
| `--dry-run` | No | Preview what would be created without actually creating issues |
| `--feature <feature-id>` | No | Generate issue for a specific feature only (e.g., `F2`) |

## Examples

```bash
# Generate issues for all features in epic E3
/epic-to-issues E3

# Generate issue for a specific feature only
/epic-to-issues E3 --feature F5

# Preview what would be created without creating issues
/epic-to-issues E1 --dry-run

# Combine flags
/epic-to-issues E9 --feature F3 --dry-run
```

## What It Does

1. **Reads the epic plan** from `adw-docs/dev-plans/epics/<epic-id>-*.md`
2. **Discovers child feature plans** matching `adw-docs/dev-plans/features/<epic-id>-F*-*.md`
3. **For each feature**, creates a `type:generate` issue that:
   - References the feature plan document path
   - Lists all phases to be generated as implementation issues
   - Sets proper dependency chain between features (F1 -> F2 -> F3 -> ...)
   - Includes ADW context instructions for the generate workflow
   - Adds labels: `agent`, `blocked`, `model:heavy`, `type:generate`
4. **Reports summary** with issue numbers and ASCII dependency diagram

## Input Structure

The command expects the following directory structure:

```
adw-docs/dev-plans/
├── epics/
│   └── <epic-id>-<name>.md          # Epic with feature breakdown table
└── features/
    ├── <epic-id>-F1-<name>.md       # Feature with phase checklist
    ├── <epic-id>-F2-<name>.md
    └── ...
```

### Required Sections in Epic Plans

The epic plan should contain a **Feature Tracks** or **Child Plans** table:

```markdown
## 4. Child Plans

### 4.1 Feature Tracks

| Feature Plan | Status | Notes |
|--------------|--------|-------|
| [Feature Name](../features/E3-F1-feature-name.md) | Proposed | Description |
| [Another Feature](../features/E3-F2-another.md) | Proposed | Description |
```

### Required Sections in Feature Plans

Each feature plan must include:

1. **Phase Checklist** with entries in format:
   ```markdown
   - [ ] **E3-F1-P1:** Phase description
   - [ ] **E3-F1-P2:** Another phase
   ```

2. **Dependencies** section or table showing blockers

3. **Files to Create/Modify** section listing affected modules

4. **Testing Strategy** section describing test approach

## Output

### Dry Run Mode

```
Epic: E3 - PR-Driven Workflow Automation

Features discovered: 6
  - E3-F1: PR Request Labels (3 phases)
  - E3-F2: PR Cron Trigger Integration (5 phases)
  - E3-F3: ADW Create-PR Command (4 phases)
  - E3-F4: PR Review Workflow (5 phases)
  - E3-F5: PR Fix Workflow (5 phases)
  - E3-F6: PR Workflow Documentation (3 phases)

Dependency Chain:
  E3-F1 (no dependencies)
    └── E3-F2 (blocked by: E3-F1)
        └── E3-F3 (blocked by: E3-F2)
        └── E3-F4 (blocked by: E3-F2)
            └── E3-F5 (blocked by: E3-F4)
                └── E3-F6 (blocked by: E3-F5)

Would create 6 issues with labels: agent, blocked, model:heavy, type:generate
```

### Normal Mode

```
Epic: E3 - PR-Driven Workflow Automation

Created issues:
  #201: [E3-F1] PR Request Labels (3 phases)
  #202: [E3-F2] PR Cron Trigger Integration (5 phases) - blocked by #201
  #203: [E3-F3] ADW Create-PR Command (4 phases) - blocked by #202
  #204: [E3-F4] PR Review Workflow (5 phases) - blocked by #202
  #205: [E3-F5] PR Fix Workflow (5 phases) - blocked by #204
  #206: [E3-F6] PR Workflow Documentation (3 phases) - blocked by #205

Dependency Diagram:
  #201 E3-F1
    └── #202 E3-F2
        ├── #203 E3-F3
        └── #204 E3-F4
            └── #205 E3-F5
                └── #206 E3-F6

All issues created with labels: agent, blocked, model:heavy, type:generate
```

## Generated Issue Format

Each generated issue follows this template:

```markdown
## Summary

Generate implementation issues for feature **E3-F1: PR Request Labels**.

## Feature Plan

**Document:** `adw-docs/dev-plans/features/E3-F1-pr-request-labels.md`

## Phases to Generate

| Phase | Description | Size |
|-------|-------------|------|
| E3-F1-P1 | Add `PR_REQUEST_LABELS` dict to labels.py | S |
| E3-F1-P2 | Create `DEFAULT_PR_LABELS` constant | XS |
| E3-F1-P3 | Update `ALL_LABELS` to include PR labels | XS |

## Dependencies

- **Blocked by:** None (first feature in epic)
- **Blocks:** E3-F2, E3-F3

## ADW Instructions

When processing this issue:
1. Read the feature plan document listed above
2. For each phase in the Phase Checklist, create an implementation issue
3. Set appropriate labels and dependencies between phase issues
4. Mark each phase issue with `agent` label for automated processing
5. Ensure co-located testing: tests must ship with implementation

## Files to Create/Modify

See feature plan for complete list.

## Testing Strategy

See feature plan for testing approach.
```

## Key Behaviors

### Dependency Inference

Features are linked sequentially based on:
1. Explicit dependencies in feature plan's **Dependencies** section
2. Order in epic's **Feature Breakdown** table (F1 -> F2 -> F3)
3. Any `Related Features` in the feature plan frontmatter

### Co-located Testing Enforcement

Generated issues include instructions requiring tests to ship with implementation,
following the repository's testing conventions.

### Issue Linking

- ASCII dependency diagrams show blockers visually
- Issue bodies reference blocking issues by number
- Labels include `blocked` until dependencies complete

### Label Assignment

All generated issues receive:
- `agent` - Managed by ADW automation
- `blocked` - Has unresolved dependencies (removed when unblocked)
- `model:heavy` - Complex generation task requires heavy model
- `type:generate` - Issue type for the generate workflow

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| Epic not found | No file matching `<epic-id>-*.md` in epics/ | Verify epic ID and file exists |
| No features found | No files matching `<epic-id>-F*-*.md` in features/ | Create feature plans first |
| Feature not found | `--feature` specified but file doesn't exist | Verify feature ID |
| Missing phase checklist | Feature plan lacks required sections | Add Phase Checklist section |
| API rate limit | Too many issues created | Wait and retry, or use --dry-run |

## Workflow Integration

This command is designed to work with the ADW generate workflow:

1. `/epic-to-issues E3` creates `type:generate` issues
2. Cron detects issues with `agent` + `type:generate` labels
3. Generate workflow reads issue, creates implementation phase issues
4. Implementation issues get processed by standard `complete` or `patch` workflows

## Permissions

This command uses the following tools:
- `read` - Read epic and feature plan files
- `glob` - Discover feature plan files
- `platform_operations` - Create GitHub issues

## See Also

- `adw-docs/dev-plans/epics/` - Epic plan templates
- `adw-docs/dev-plans/features/` - Feature plan templates
- `.opencode/workflow/generate.json` - Generate workflow definition
- `adw workflow complete` - Implementation workflow

## Technical Notes

The command:
1. Parses epic plan to extract feature list and order
2. For each feature, parses the feature plan to extract phases
3. Builds dependency graph from explicit dependencies and sequential order
4. Creates issues via `platform_operations create-issue` command
5. Returns issue numbers for reference

Phase IDs follow the convention: `<epic-id>-F<feature-num>-P<phase-num>`
(e.g., `E3-F1-P1`, `E3-F2-P3`)
