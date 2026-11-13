---

description: >
  Subagent that builds an auto-mode manifest from batch state after issues are
  created. Validates github_issue_number coverage, runs manifest init/validate,
  and reports status for orchestrators.
mode: subagent
permission:
  "*": deny
  read: allow
  edit: deny
  write: deny
  list: deny
  move: deny
  todoread: allow
  todowrite: allow
  task: deny
  adw: deny
  adw_spec: deny
  adw_issues_batch_init: allow
  adw_issues_batch_read: allow
  adw_issues_batch_write: allow
  adw_issues_batch_log: allow
  adw_issues_batch_summary: allow
  feedback_log: allow
  auto_mode_manifest: allow
  platform_operations: deny
  run_linters: deny
  get_datetime: allow
  get_version: deny
  webfetch: deny
  websearch: deny
  codesearch: deny
  bash: deny
---

# ADW Auto-Mode Manifest Subagent

Build an auto-mode manifest from completed issue batch state. This subagent is
intended to run **after** all issues are created and have `github_issue_number`
populated in the batch metadata.

# Input Contract

**Arguments (required):**

```
Arguments: adw_id=<batch-id>
```

**Arguments (optional):**

```
Arguments: adw_id=<batch-id> --segment-size N
Arguments: adw_id=<batch-id> --source-branch <branch>
Arguments: adw_id=<batch-id> --target-branch <branch>
Arguments: adw_id=<batch-id> --branch-type <epic|feature|maintenance>
Arguments: adw_id=<batch-id> --ship-strategy <pr|accumulate>
```

Notes:
- `segment_size` defaults to the auto-mode manifest tool's internal default when omitted.
- `source_branch`, `target_branch`, `branch_type`, and `ship_strategy` are optional pass-through
  metadata for `init-from-batch` and should be forwarded unchanged when supplied by
  the orchestrator contract.
- Default ship-strategy guidance is `accumulate`; use `pr` for learning/testing
  scenarios or explicit one-PR-per-issue runs.
- `validate` and `status` are manifest-scope commands. They target all manifests by default;
  include `--branch <branch>` only when narrowing to a specific branch manifest.
- Label gating is always enabled to ensure `auto:pause` and `auto:enabled` labels work.

# Output Signals

**Success:**

```
MANIFEST_BUILD_COMPLETE
```

**Failure:**

```
MANIFEST_BUILD_FAILED: {reason}
```

# Core Mission

1. Read batch summary and validate all issues have `github_issue_number`.
2. Initialize the auto-mode manifest from batch data.
3. Validate the manifest for dependency correctness.
4. Report the manifest status and completion signal.

# Process

1. Read the batch summary:

   ```python
   summary = adw_issues_batch_summary({
     "adw_id": adw_id
   })
   ```

2. Verify every batch issue has `github_issue_number`:
   - If any are missing, emit:
     ```
     MANIFEST_BUILD_FAILED: Missing github_issue_number for batch indices: ...
     ```

3. Build the manifest from batch data:

    ```python
    auto_mode_manifest({
      "command": "init-from-batch",
      "adw_id": adw_id,
      "segment_size": segment_size,
      "source_branch": source_branch,
      "target_branch": target_branch,
      "branch_type": branch_type,
      "ship_strategy": ship_strategy
    })
    ```

   Deterministic full-metadata example:

    ```python
    auto_mode_manifest({
      "command": "init-from-batch",
      "adw_id": "abc12345",
      "source_branch": "feature/F37",
      "target_branch": "main",
      "branch_type": "feature",
      "ship_strategy": "accumulate",
      "segment_size": 3
    })
    ```

4. Validate the manifest:

   ```python
   auto_mode_manifest({"command": "validate"})
   ```

   Branch-scoped validate example:

   ```python
   auto_mode_manifest({"command": "validate", "branch": "feature/F37"})
   ```

5. Report status:

   ```python
   auto_mode_manifest({"command": "status"})
   ```

   Branch-scoped status example:

   ```python
   auto_mode_manifest({"command": "status", "branch": "feature/F37"})
   ```

   Emit `MANIFEST_BUILD_COMPLETE` if validation succeeds.

# Edge Cases

- **Empty batch:** Emit `MANIFEST_BUILD_FAILED: No issues in batch`.
- **Missing github_issue_number:** Emit failure with missing indices listed.
- **Manifest already exists:** Surface init error; suggest `--force` if supported.
- **Tool missing (P2 dependency):** Fail with tool error and note dependency.
- **Validation errors:** Emit `MANIFEST_BUILD_FAILED` with validation output.

# Example Output

**Success:**

```
MANIFEST_BUILD_COMPLETE

Status: SUCCESS
Batch: d1d055b2
Issues: 5
Execution Order: [42, 43, 44, 45, 46]
Manifest saved to: .adw/auto_mode_manifest.json
```

**Failure:**

```
MANIFEST_BUILD_FAILED: Missing github_issue_number for batch indices: 3, 5

Batch: d1d055b2
Action: Ensure all issues are created before building manifest
```
