<!-- TEMPLATE: Replace this entire file with outcomes and guardrails -->

Define what success looks like and what is explicitly off the table.

**Required elements:**
- **Primary Outcome:** The single most important result
- **Secondary Goals:** 2-4 additional objectives
- **Guardrails / Non-Goals:** What this epic will NOT do

**Example (E17):**
- **Primary Outcome:** Dev-plans become queryable, validatable, and
  auto-indexed from structured JSON + markdown source files.
- **Secondary Goals:**
  - Runtime analytics (cost, duration, tokens) captured per workflow run
  - Generated markdown preserves current GitHub/GitLab browsing quality
  - Agents can read/write plan state programmatically via CLI or Python API
- **Guardrails / Non-Goals:**
  - No external database dependency (Redis, Postgres, Dolt, etc.)
  - No binary database as canonical source (SQLite is generated-only)
  - No removal of human-readable markdown output
  - No breaking change to existing `adw spec` or `adw auto-mode` commands
