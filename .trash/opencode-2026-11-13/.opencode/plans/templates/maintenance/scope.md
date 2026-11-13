<!-- TEMPLATE: Replace this entire file with the maintenance scope -->

Define the boundaries of this maintenance effort.

**Required elements:**
- **Modules / Directories:** File paths and modules in scope
- **Interfaces / APIs:** What contracts or surfaces are affected
- **Out of Scope:** Explicitly excluded items

**Example (M23):**
- **Modules / Directories:**
  - `.opencode/tools/run_pytest.ts` and its backing script
  - `.opencode/tools/git_operations.ts`
  - `.opencode/tools/adw_plans.ts` and wrapper-facing docs
  - `.opencode/tools/build_mkdocs.ts`
  - Accumulate/final-handoff code paths used by `shipper-auto-final`

- **Interfaces / APIs:**
  - `run_pytest` result classification for coverage failure
  - `git_operations` continue/recovery behavior
  - `adw_plans` wrapper command surface for `analytics`

- **Out of Scope:**
  - Feedback-log read capability (handled separately)
  - Broad redesign of ADW workflow architecture
  - Changing repository-wide coverage thresholds
