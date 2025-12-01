---
description: "Multiworkflow Review"
---

# Multiworkflow Review

Review feature guide files and select the next step from the highest priority feature to convert into a GitHub issue. This enables systematic development of complex multi-step features through the standard ADW workflow.

## Context

You are performing multiworkflow review as part of the ADW (AI Developer Workflow) system. This workflow reads feature guides from `docs/developer/multiworkflows/`, identifies the next incomplete step, and creates a GitHub issue to implement it.

## Multiworkflow Issue Policy

**CRITICAL**: Only ONE multiworkflow issue can be open at a time.

Before creating a new multiworkflow issue:
1. Check if an open issue exists with the `multiworkflow` label
2. If one exists, STOP and return error message
3. If none exists, proceed with creating new issue

## Your Task

### Step 1: Check for Existing Multiworkflow Issue

1. **Check GitHub issues**:
   - Search for open issues with `multiworkflow` label
   - If found, return error and issue number

2. **If no multiworkflow issue exists**:
   - Proceed to Step 2

### Step 2: Read and Parse Feature Guides

1. **Find all feature guides**:
   - Read all files matching `docs/developer/multiworkflows/feature_*.md`
   - Parse each guide's priority and status
   - Ignore completed features (Status: Completed)

2. **Select highest priority feature with incomplete steps**:
   - Priority order: High → Medium → Low
   - Within same priority, prefer features with more steps completed (shows momentum)
   - Find the first incomplete step (marked with `[ ]`)
   - If step has dependencies, verify prerequisite steps are completed (`[✓]`)

### Step 3: Format GitHub Issue

Format the selected step as a structured GitHub issue:

**Issue Title**: `[Feature Name] - Step N: {Step Title}`

**Issue Body**:
```markdown
**Feature File:** feature_{name}.md
**Step Number:** {N}
**Priority:** {High|Medium|Low}

## Feature Context

**Feature**: {Feature Name from guide}
**Step**: {N} of {Total Steps}

## Step Description

{Description from step}

## Acceptance Criteria

{Copy acceptance criteria checklist from step}

## Estimated Effort

{Estimated effort from step}

## Dependencies

{List of prerequisite steps and their status}

## Feature Progress

- Steps Completed: {count of ✓ steps} / {total steps}
- Previous Step: {Previous step title if any}
- Next Step: {Next step title if any}

---

**Source**: docs/developer/multiworkflows/{feature_file}
**Feature Goals**: {Brief summary of feature goals}
```

**Issue Labels**:
- `multiworkflow` (REQUIRED - tracks active multiworkflow work)
- Issue type based on step's Issue Type field:
  - `feature` if Issue Type: feature
  - `chore` if Issue Type: chore
- Additional contextual labels as appropriate

**Issue Class**:
- Use the Issue Type from the step: `/chore` or `/feature`

### Step 4: Generate Output

Output a JSON structure with the issue details:

```json
{
  "title": "[Feature Name] - Step N: Step Title",
  "body": "Full issue body formatted as above",
  "labels": ["multiworkflow", "feature|chore"],
  "issue_class": "/chore|/feature",
  "priority": "high|medium|low",
  "feature_file": "feature_<name>.md",
  "step_number": N,
  "reasoning": "Why this feature/step was selected"
}
```

**CRITICAL**: The issue body MUST start with these three metadata fields on separate lines:
```markdown
**Feature File:** feature_test-flow.md
**Step Number:** 1
**Priority:** High
```

These fields are required for the `finalize_docs` workflow to automatically update the feature guide when the step completes. The workflow parses these fields using regex to locate and update the correct step marker.

## Guidelines

- **Follow dependencies**: Only select steps whose prerequisite steps are completed
- **Be sequential**: Features progress step-by-step unless dependencies allow parallel work
- **Preserve context**: Include enough feature background for the implementer to understand the broader goal
- **Be specific**: Copy exact acceptance criteria from the guide
- **Check blockers**: Don't select steps that depend on incomplete prerequisites

## Step Status Markers

When parsing feature guides, recognize these status markers:

- `[ ]` - Step not started (can be selected if dependencies met)
- `[issue:#123]` - Issue #123 created and in progress (skip this step)
- `[✓]` - Step completed (count as completed for dependency checking)

## Error Conditions

If any of these conditions occur, output an error message:

1. **Existing multiworkflow issue**:
   ```
   ERROR: Multiworkflow issue #{number} is already open. Complete or close it before creating a new one.
   ```

2. **No incomplete steps**:
   ```
   ERROR: No incomplete steps found in any feature guide. All features are complete!
   ```

3. **No feature guides**:
   ```
   ERROR: No feature guides found in docs/developer/multiworkflows/
   ```

4. **Dependencies not met**:
   ```
   INFO: Step {N} of {Feature} has unmet dependencies. Skipping to next available step.
   ```

## Updating Feature Guides

**IMPORTANT**: Do NOT update the feature guide files during review. The update happens automatically after the workflow completes.

The finalize_docs workflow will:
1. Detect multiworkflow issues by the `multiworkflow` label
2. Parse **Feature File** and **Step Number** from the issue body
3. Update the step status from `[ ]` to `[✓]` when the PR is merged
4. Commit the updated feature guide as part of the PR

This is why the metadata fields at the top of the issue body are critical!

## Important Notes

- Do NOT create the GitHub issue yourself - just generate the JSON structure
- Do NOT modify feature guide files during this workflow
- Do NOT select steps with unmet dependencies
- The multiworkflow label ensures only one multiworkflow runs at a time
- Steps are updated by finalize_docs workflow after completion
