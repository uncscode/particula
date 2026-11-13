<!-- TEMPLATE: Replace this entire file with existing code to reuse -->

List existing functions, modules, patterns, and file paths that this feature
should build on. Include file:line references where helpful.

**Required elements:**
- Existing functions/classes to call or extend
- File paths with line numbers for key integration points
- Patterns from the codebase to follow

**Example (E16-F6):**
- `open_final_pr()` in `adw/automode/scheduler.py:719` -- creates the
  epic-to-main PR; reuse for final handoff PR creation
- `create_pr()` in `adw/platforms/router.py:1523` -- platform-agnostic PR
  creation; call from final handoff step
- `create_issue_comment()` in `adw/platforms/router.py:1281` -- posts safety
  banners; reuse for guardrail comment
- `ManifestCheckpoint` in `adw/automode/manifest.py` -- records
  `"final_pr_opened"` events for idempotency
