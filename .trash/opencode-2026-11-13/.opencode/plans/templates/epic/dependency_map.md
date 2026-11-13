<!-- TEMPLATE: Replace this entire file with the dependency map -->

Map inbound and outbound dependencies, plus sequencing constraints between
child plans.

**Required elements:**
- **Inbound:** What this epic depends on
- **Outbound:** What depends on this epic
- **Sequencing:** Ordering constraints between child features

**Example (E17):**
**Inbound:**
- Current `plans/` structure -- canonical source for plan records and sections
- `adw/core/constants.py` -- `PLANS_DIR` constant
- `adw/cli.py` -- CLI registration point for new `plans` group

**Outbound:**
- Agent tooling (dev-plan-manager, dev-plan-creator) -- must update to use JSON
- `.opencode/plans/sections/` -- direct plan content instead of generated views

**Sequencing:**
- E17-F1 (models) must ship before E17-F2 (layout) and E17-F3 (CLI)
- E17-F2 (layout) must ship before E17-F4 (generator) and E17-F5 (migration)
- E17-F3 (CLI) can proceed in parallel with E17-F4
- E17-F6 (auto-update hook) depends on E17-F1 and E17-F3
