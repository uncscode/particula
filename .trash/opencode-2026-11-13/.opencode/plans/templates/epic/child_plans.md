<!-- TEMPLATE: Replace this entire file with child plan tables -->

List every feature and maintenance plan spawned by this epic.

**Required subsections:**

### Feature Tracks
| ID | Feature Plan | Status | Notes |
|----|--------------|--------|-------|
| {EPIC_ID}-F1 | Feature title | Draft | Brief description |

### Maintenance Tracks
| ID | Maintenance Plan | Trigger | Notes |
|----|------------------|---------|-------|
| {EPIC_ID}-M1 | Maintenance title | On commit / weekly | Brief description |

**Example (E17):**

### Feature Tracks
| ID | Feature Plan | Status | Notes |
|----|--------------|--------|-------|
| E17-F1 | Pydantic Models & JSON Schema | Shipped | Base models + schema generation |
| E17-F2 | Plan File Layout & Section Management | Shipped | Directory structure, section conventions |
| E17-F3 | `adw plans` CLI | Shipped | list/show/validate/generate/schema commands |
| E17-F5 | Portable Migration Tool | Shipped | Markdown to JSON + sections, repo-agnostic |

### Maintenance Tracks
| ID | Maintenance Plan | Trigger | Notes |
|----|------------------|---------|-------|
| E17-M1 | Plan Validation CI & Analytics Pipeline | On commit / weekly | CI schema validation + analytics |
