<!-- TEMPLATE: Replace this entire file with the implementation strategy -->

Summarize the architectural approach, key design decisions, and testing
strategy that span the entire epic. Link to detailed specs or ADRs.

**Required elements:**
- Architecture overview (how the system is structured)
- Key data ownership rules (what lives where, who updates it)
- Reusable patterns from the codebase
- Testing requirements (the standard coverage policy)

**Testing Requirements (include verbatim):**
1. Test coverage thresholds must NEVER be lowered
2. Each phase must include self-contained tests
3. Tests are committed in the same PR as the implementation
4. Test files use `*_test.py` suffix in module-level `tests/` directories
5. Minimum 80% coverage (configured in `pyproject.toml`)

**Example (E17):**
The system follows a two-layer architecture:
- **Layer 1 -- Canonical Source:** One JSON per plan + section markdown files
  under `.opencode/plans/`, validated by Pydantic models and JSON Schema
- **Layer 2 -- Direct Consumption:** Agents and operators read section files
  directly from `.opencode/plans/sections/`

Key patterns reused: Click groups from `adw/commands/spec.py`, Pydantic models
from `adw/automode/manifest.py`, JSON Schema generation from `model_json_schema()`.
