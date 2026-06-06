# Epic-to-Issues Agent - Usage Guide

## Overview

The `epic-to-issues` agent reads an epic plan and its child feature plans, then
creates `type:generate` GitHub issues for each feature. Each issue follows a
fixed template and triggers the generate workflow to produce implementation
phase issues.

This agent is intentionally simpler than the `issue-generator` pipeline. It does
NOT use `adw_issues_spec` batch state or the 5-reviewer pipeline because
`type:generate` issues follow a well-defined template that doesn't need iterative
review.

## When to Use

- **Kicking off epic execution**: You have an epic plan (E13, E5, etc.) with
  feature tracks and want to create the `type:generate` issues to start the
  cascade.
- **Single feature**: You want to create a generate issue for one specific
  feature (e.g., `E13-F5`) without processing the whole epic.
- **Previewing**: You want to see what issues would be created before committing
  (`--dry-run`).
- **Resuming**: Some features are already shipped and you want to create issues
  for only the remaining features.

## When NOT to Use

- **Implementation issues**: Use `issue-generator` for creating detailed
  implementation phase issues with the 5-reviewer quality pipeline.
- **Ad-hoc issues**: Use `platform_operations create-issue` directly for
  one-off issues that don't follow the epic/feature plan structure.
- **Issues from arbitrary text**: Use `issue-generator` which accepts URLs,
  text, and feature plan paths.

## Tool Configuration

| Tool | Enabled | Rationale |
|------|---------|-----------|
| `read` | Yes | Read epic and feature plan files |
| `edit` | No | No file modifications |
| `write` | No | No file creation |
| `list` | Yes | List directories to discover plan files |
| `ripgrep` | Yes | File discovery for epic/feature plan matching |
| `todoread` | Yes | Task tracking |
| `todowrite` | Yes | Task tracking |
| `task` | No | Subagent, cannot invoke other agents |
| `platform_operations` | Yes | Create GitHub issues via `create-issue` |
| `get_datetime` | Yes | Timestamps for reporting |
| `bash` | No | Always disabled for security |

## Usage Examples

### Example 1: Full Epic

**Prompt:**
```
Create generate issues for all features in epic E13
```

**Behavior:**
1. Reads `adw-docs/dev-plans/epics/E13-version-correction-docs-overhaul.md`
2. Discovers 6 feature plans (E13-F1 through E13-F6) + 1 maintenance (E13-M1)
3. Skips E13-F1 (Status: Shipped)
4. Creates 5 + 1 issues in dependency order
5. Reports summary with dependency diagram

### Example 2: Single Feature

**Prompt:**
```
Create a generate issue for E5-F6 only. E5-F5 is issue #802.
```

**Behavior:**
1. Reads E5 epic plan for context
2. Reads E5-F6 feature plan
3. Creates one issue with dependency referencing `#802 [E5-F5]`

### Example 3: Dry Run

**Prompt:**
```
Dry run: show what issues would be created for E13, skip F1
```

**Behavior:**
1. Reads epic and all feature plans
2. Prints preview table with features, phase counts, and dependency chain
3. Does NOT create any issues

### Example 4: Skip Already-Processed Features

**Prompt:**
```
Create generate issues for E5 --skip F1,F2,F3
```

**Behavior:**
1. Reads E5 epic plan
2. Skips F1, F2, F3 as instructed
3. Creates issues for remaining features (F4, F5, F6, F7)

## Input Structure

The agent expects this directory structure:

```
adw-docs/dev-plans/
  epics/
    <epic-id>-<name>.md          # Epic with feature breakdown table
    completed/
      <epic-id>-<name>.md        # Completed epics
  features/
    <epic-id>-F1-<name>.md       # Feature plans with phase checklists
    <epic-id>-F2-<name>.md
    ...
  maintenance/
    <epic-id>-M1-<name>.md       # Maintenance plans (same structure)
```

### Required Epic Sections

- **Section 4 (Child Plans)**: Feature Tracks table with ID, name, status, path
- **Section 5 (Dependency Map)**: Visual dependency graph + execution order

### Required Feature Plan Sections

- **Phase Checklist**: Entries like `- [ ] **E13-F2-P1:** Description`
- **Dependencies / Related Features**: Blockers and soft dependencies
- **Files to Create/Modify**: Affected modules
- **Testing Strategy**: Test approach

## Generated Issue Format

**Title:** `[E13-F2] Generate implementation issues for README Slimdown`

**Labels:** `agent`, `blocked`, `model:heavy`, `type:generate`

**Body sections:**
1. Dependency diagram (ASCII art at top)
2. Summary
3. Feature Plan (document path + parent epic)
4. Phases to Generate (table)
5. Dependencies (blocked by, also leverages, blocks)
6. ADW Instructions (step-by-step for the generate workflow)
7. Files to Create/Modify
8. Testing Strategy

## Integration with Other Agents

| Agent | Relationship |
|-------|-------------|
| `issue-generator` | Different use case: creates detailed implementation issues via 5-reviewer pipeline |
| `adw-issue-creator` | Used by issue-generator for batch creation; this agent uses `platform_operations` directly |
| Generate workflow | Processes the `type:generate` issues this agent creates |
| Complete/Patch workflows | Process implementation issues spawned by the generate workflow |

## Pipeline Flow

```
epic-to-issues          Generate workflow       Complete/Patch workflows
     |                       |                        |
     v                       v                        v
[E13-F2] type:generate  --> [E13-F2-P1] type:complete --> PR #501
                             [E13-F2-P2] type:complete --> PR #502
                             [E13-F2-P3] type:complete --> PR #503
```

## Troubleshooting

### Issue: "Epic not found"
**Cause:** No file matching `<epic-id>-*.md` in `epics/` or `epics/completed/`
**Solution:** Verify the epic ID and that the plan file exists

### Issue: "No features found"
**Cause:** No files matching `<epic-id>-F*-*.md` in `features/`
**Solution:** Create feature plan documents before running the agent

### Issue: "All features shipped"
**Cause:** Every feature in the epic has Status: Shipped
**Solution:** Use `--feature F3` to force-process a specific feature

### Issue: API rate limit during creation
**Cause:** Creating many issues quickly can hit GitHub rate limits
**Solution:** Agent retries up to 3 times per issue. For large epics, partial
results are reported so you can resume.

## See Also

- `.opencode/agent/epic-to-issues.md` - Agent definition
- `.opencode/agent/issue-generator.md` - Full multi-review issue pipeline
- `.opencode/command/epic-to-issues.md` - Original command specification
- `adw-docs/dev-plans/epics/` - Epic plan templates
- `adw-docs/dev-plans/features/` - Feature plan templates
