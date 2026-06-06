# Auto-Mode Bootstrap

`auto-mode-bootstrap` is the second-step primary agent for the `generate-auto`
workflow. It assumes issue creation already completed and turns a finished batch
into a runnable auto-mode setup.

It does three things:
- validates bootstrap metadata
- ensures the deterministic source branch exists from `main` without switching to it, then pushes that named branch directly
- initializes the auto-mode manifest and labels created issues with `auto:enabled`

## When To Use

Use this agent when:
- `issue-generator` already created child implementation issues
- you want `generate-auto` to continue into manifest-backed auto execution
- you need deterministic, fail-closed bootstrap for standalone feature or maintenance plans

Do not use this agent for:
- plain `generate`
- epic-linked plans
- incomplete batches
- ambiguous or partially reconstructed plan metadata

## Bootstrap Data Model

The preferred bootstrap source is a structured workflow message written by
`issue-generator`.

Recommended payload:

```json
{
  "message_type": "issue_generation_bootstrap",
  "plan_id": "F31",
  "plan_type": "feature",
  "standalone": true,
  "epic_linked": false,
  "bootstrap_supported": true,
  "bootstrap_reason": "standalone feature plan",
  "source_doc": "adw-docs/dev-plans/features/F31-example.md",
  "source_branch": "feature/F31",
  "target_branch": "main",
  "branch_type": "feature",
  "ship_strategy": "accumulate",
  "created_issue_numbers": [2001, 2002, 2003],
  "dependency_map": {
    "1": [],
    "2": ["1"],
    "3": ["2"]
  },
  "execution_order": [2001, 2002, 2003]
}
```

## Deterministic Rules

Supported plan types:
- standalone feature
- standalone maintenance

Branch naming:
- feature -> `feature/{plan-id}`
- maintenance -> `maintenance/{plan-id}`

Branch base:
- always `main`

Manifest metadata:
- `target_branch=main`
- `ship_strategy=accumulate`

## Failure Policy

This agent is intentionally fail-closed.

It fails when:
- no usable bootstrap message exists and parsing cannot recover
- message and parsed fallback disagree
- plan is epic-linked
- `plan_id` is empty or missing
- batch completion is incomplete
- manifest initialization fails

## Workflow Placement

`generate-auto.json` should run:
1. `issue-generator`
2. `auto-mode-bootstrap`

## Expected Result

- source branch exists on origin
- bootstrap does not require checking out the source branch before pushing it
- manifest is initialized and validated
- created implementation issues are labeled `auto:enabled`

## Related Files

- `.opencode/agent/auto-mode-bootstrap.md`
- `.opencode/agent/issue-generator.md`
- `.opencode/agent/adw-auto-mode-manifest.md`
- `.opencode/workflow/generate-auto.json`
- `docs/Examples/operations/auto-mode-runbook.md`
