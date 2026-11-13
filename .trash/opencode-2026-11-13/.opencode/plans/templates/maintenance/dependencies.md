<!-- TEMPLATE: Replace this entire file with dependency information -->

List upstream/downstream systems that influence this maintenance work and
how updates should be coordinated.

**Required elements:**
- Related plans (what overlaps, what is explicitly separate)
- Phase ordering constraints
- Coordination requirements

**Example (M23):**
- **Related but separate work:** M21 and M22 remain independent plans; this
  plan consumes their lessons but does not change their scope.
- **Out-of-scope dependency:** feedback-log read capability remains on its own
  track and should not be folded in.
- **Tool parity dependency:** `adw_plans analytics` must follow the repository
  parity rule: wrapper, bun tests, and wrapper-facing docs in the same slice.
- **Phase ordering:**
  - M23-P1 and M23-P2 are related but can be developed as separate slices
  - M23-P9 should land before M23-P10 so fast default is established first
  - M23-P12 (docs) depends on all prior phases
