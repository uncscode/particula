# Planner Intake - Usage Guide

## Overview

The Planner Intake is an **interactive primary agent** that walks you through filling out the rough-scoping planner template through guided conversation. It researches the codebase, checks for duplicate plans, recommends child tracks for epics, validates the completed template, and posts the GitHub issue with correct labels.

**This agent does NOT run the planner workflow.** It creates the issue and stops. You run `uv run adw workflow planner <issue-number>` separately.

## When to Use

- **You have an idea** for a new feature, epic, maintenance, or research effort and want help structuring it
- **You need codebase context** to properly scope the work before creating a planner issue
- **You want validation** that your rough-scoping template will parse correctly downstream
- **You want to avoid duplicates** by checking against existing active plans first

## When NOT to Use

- **You already have a filled-out template** -- just create the issue directly on GitHub using the template
- **You want to run the planner workflow** -- use `uv run adw workflow planner <issue-number>`
- **You want to edit an existing plan document** -- use `dev-plan-manager` instead
- **You want to create implementation issues** -- use `issue-generator` or `epic-to-issues`

## Agent Structure

| Agent | Type | Purpose |
|-------|------|---------|
| `planner-intake` | Primary | Interactive guide, validates and posts issue |
| `codebase-researcher` | Subagent | Deep codebase research for scope and track recommendations |

## Tool Configuration

| Tool | Enabled | Rationale |
|------|---------|-----------|
| `read` | Yes | Read indexes, plans, architecture docs |
| `list` | Yes | Browse plan directories |
| `ripgrep` | Yes | Search for existing plans, duplicates, patterns |
| `task` | Yes | Delegate to `codebase-researcher` |
| `platform_operations` | Yes | Create GitHub issue with labels |
| `todoread` | Yes | Track conversation progress |
| `todowrite` | Yes | Track conversation progress |
| `get_datetime` | Yes | Timestamps |
| `get_version` | Yes | Version info |
| `feedback_log` | Yes | Report tool friction |
| `edit` | No | Read-only agent |
| `write` | No | Read-only agent |
| Atomic git wrappers | No | No git needed |
| `run_pytest` | No | No code changes |
| `run_linters` | No | No code changes |
| `adw_spec` | No | Not running inside a workflow |
| `bash` | No | Always disabled |

## Conversation Flow

The agent follows a structured wizard-style conversation:

```
Step 1: Understand the Idea
  - Ask about type (epic/feature/maintenance/research)
  - Check for duplicate/overlapping plans
  - Determine if codebase research is needed

Step 2: Research the Codebase
  - Delegate to codebase-researcher
  - Gather module structure, patterns, dependencies

Step 3: Walk Through Template Sections (one at a time)
  - Type -> Vision -> Problem Statement -> Rough Scope
  - -> Child Tracks -> Dependencies -> Constraints -> Success Metrics
  - Confirm each section before moving to the next

Step 4: Validate the Template
  - Check all sections against plan-scope-analyzer parsing rules
  - Flag incomplete or ambiguous sections

Step 5: Present Full Issue for Approval
  - Show complete issue body
  - Wait for explicit user confirmation

Step 6: Create the Issue
  - Post to GitHub with labels: agent, blocked, model:base, type:planner
  - Report issue number and URL
  - Suggest next step: uv run adw workflow planner <number>
```

## Usage Examples

### Example 1: Feature Planning

**Context**: You want to add webhook support to ADW.

**Invocation**: Talk to the `planner-intake` agent:
```
"I want to plan adding webhook support for triggering workflows"
```

**Expected Behavior**:
1. Agent reads plan indexes, finds no overlapping plans
2. Delegates to `codebase-researcher` to map trigger mechanisms in `adw/triggers/`
3. Asks: "This sounds like a Feature plan -- standalone or linked to an epic?"
4. Walks through Vision, Problem Statement, Scope using codebase findings
5. Suggests child tracks: P1 (webhook receiver), P2 (event parsing), P3 (trigger integration)
6. Validates and presents the full template
7. Creates issue #N with `type:planner` label

### Example 2: Epic Planning

**Context**: You want to refactor the state management layer.

**Invocation**:
```
"I need to plan a major refactor of how ADW manages workflow state"
```

**Expected Behavior**:
1. Agent reads indexes, identifies E17 (Structured Plan Database) as potentially related
2. Flags: "E17 touches state persistence -- is this related or separate?"
3. Delegates to `codebase-researcher` to map `adw/state/`, `adw/core/models.py`
4. Recommends epic type with feature and maintenance child tracks
5. Proposes tracks based on module boundaries found in research
6. User approves/edits tracks
7. Validates all 8 sections
8. Creates issue with full template

### Example 3: Maintenance Planning

**Context**: You need to clean up deprecated APIs.

**Invocation**:
```
"We should clean up all the deprecated helper functions across the codebase"
```

**Expected Behavior**:
1. Agent searches for `@deprecated`, `DeprecationWarning`, deprecated markers
2. Reports: "Found N deprecated functions across M modules"
3. Asks about scope: all at once vs module-by-module
4. Fills in Maintenance type, populates scope table from search results
5. Suggests tracks for each affected module
6. Creates issue

## Template Reference

The agent fills out this template (`.github/ISSUE_TEMPLATE/planner.md`):

| Section | What the Agent Asks |
|---------|-------------------|
| **Type** | Epic, Feature, Maintenance, or Research? |
| **Vision** | Outcome statement, users/owners, value? |
| **Problem Statement** | Current state, pain points, why now? |
| **Rough Scope** | In-scope vs out-of-scope (Functional, Technical, Validation)? |
| **Child Tracks** | Track IDs, goals, sizes, dependencies? |
| **Dependencies** | Internal and external dependencies? |
| **Constraints** | Timeline, resource, tooling, compliance? |
| **Success Metrics** | Acceptance criteria, risks, validation strategy? |

Standalone research requests may use deterministic placeholders such as
`research_tracks: auto`, while epic requests may include explicit research child
tracks alongside feature and maintenance work. The downstream validation
contract also allows standalone feature/maintenance/research issues to use
explicit notes or let the analyzer emit deterministic `auto` placeholders.

## Validation Rules

Before posting, the agent checks these rules (matching `plan-scope-analyzer` expectations):

| Rule | Check |
|------|-------|
| Type | Exactly one checkbox checked |
| Vision | All three sub-fields populated |
| Problem Statement | All three sub-fields populated |
| Rough Scope | At least one in-scope item per area |
| Child Tracks | Epics need at least one track with Goal and Size; standalone feature/maintenance/research issues may use explicit notes or let downstream emit deterministic `auto` placeholders |
| Dependencies | Either "None" or specific items listed |
| Constraints | At least one field populated or explicit "none" |
| Success Metrics | At least one metric checked or custom added |

## Labels Applied

The created issue gets these labels automatically:

| Label | Purpose |
|-------|---------|
| `agent` | Marks as agent-processable |
| `blocked` | Prevents accidental workflow trigger before review |
| `model:base` | Default model tier |
| `type:planner` | Routes to planner workflow |

## Integration with Other Agents

| Agent | Relationship |
|-------|-------------|
| `plan-scope-analyzer` | Parses the issue body this agent creates |
| `codebase-researcher` | Provides codebase context during conversation |
| `plan-draft` / `plan-orchestrator` | Drafts plan documents from the created issue |
| `dev-plan-manager` | Manages plan documents after planner workflow creates them |

## After Issue Creation

Once the agent creates the issue:

1. **Review the issue** on GitHub -- make sure it looks right
2. **Remove `blocked` label** if you want auto-pickup, or keep it for manual trigger
3. **Run the planner workflow**:
   ```bash
   uv run adw workflow planner <issue-number>
   ```
4. The planner pipeline will draft plans, review them, ship a PR, and surface questions

## Limitations

- **Read-only**: Cannot modify files in the repository
- **No workflow trigger**: Does not start the planner workflow -- you do that manually
- **Single issue**: Creates one issue per conversation
- **GitHub only**: Uses `platform_operations` which supports GitHub and GitLab, but the template and labels are GitHub-oriented

## Troubleshooting

### Issue: Agent suggests tracks that don't make sense
**Solution**: The agent's track recommendations come from `codebase-researcher` findings. Tell it what's wrong and it will revise. You have full approval authority over tracks.

### Issue: Validation flags a section as incomplete
**Solution**: The agent checks against `plan-scope-analyzer` rules. Repair the missing or ambiguous content before posting. If the issue would remain malformed, fail closed with a clear explanation instead of posting it as-is.

### Issue: Duplicate plan detected
**Solution**: The agent checks active plan indexes. If it flags a duplicate, confirm whether your idea is related (add as dependency) or truly separate (proceed with new issue).

## See Also

- [Planning System Overview](planning-system.md) -- Full planner pipeline documentation
- [Dev-Plan Manager](dev-plan-manager.md) -- Managing plan documents after creation
- [Epic-to-Issues](epic-to-issues.md) -- Creating implementation issues from plans
- `.github/ISSUE_TEMPLATE/planner.md` -- The raw template
- `.opencode/workflow/planner.json` -- The planner workflow definition
