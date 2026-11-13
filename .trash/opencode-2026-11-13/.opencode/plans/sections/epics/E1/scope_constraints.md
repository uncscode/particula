<!-- TEMPLATE: Replace this entire file with scope and constraints -->

Define high-level scope boundaries, participating components, and critical
constraints.

**Required elements:**
- **In Scope:** Concrete deliverables (bullet list)
- **Out of Scope:** Explicitly excluded items
- **Constraints:** Technical, compliance, or operational limits

**Example (E17):**
**In Scope:**
- Pydantic models for plan metadata (epic, feature, maintenance)
- JSON Schema generation from Pydantic models
- `adw plans` CLI group (list, show, validate, generate, schema, analytics)
- Portable migration tool to convert markdown plans to JSON + sections
- Post-execution hook that auto-updates phase status on workflow completion

**Out of Scope:**
- Web UI or dashboard for plan browsing
- Real-time collaboration or live editing
- External database hosting

**Constraints:**
- All canonical source files must be plain text (JSON + markdown) for Git
- Existing plan IDs must be preserved
- Phase ID format (`E{n}-F{m}-P{k}`) must remain unchanged
