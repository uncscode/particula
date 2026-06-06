<!-- TEMPLATE: Replace this entire file with governance decisions -->

Document key decisions, approvals, and design reviews as they happen.

| Date | Decision | Owner | Impact |
|------|----------|-------|--------|
| YYYY-MM-DD | Decision description | @owner | What this affects |

**Example (E17):**
| Date | Decision | Owner | Impact |
|------|----------|-------|--------|
| 2026-03-27 | Use JSON + Markdown (not Dolt, not pure SQLite) | @gorkowski | Keeps source Git-friendly |
| 2026-03-27 | One JSON file per plan (not monolithic) | @gorkowski | Small diffs, fewer merge conflicts |
| 2026-03-27 | Plans never move on completion -- lifecycle field controls views | @gorkowski | Stable file paths, no churn |
| 2026-03-27 | Section prose stays in markdown (not inside JSON strings) | @gorkowski | Better authoring, cleaner diffs |
