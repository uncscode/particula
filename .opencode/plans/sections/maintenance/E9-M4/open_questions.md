# Open Questions

1. [ ] Which accepted M1–M3 revision freezes accessor spelling, ordered mixed-gas
   mapping, concentration/volume conventions and native strategy signatures?
   Evidence: upstream plans remain drafts. Require their completed-and-validated
   handoff rather than inventing answers here; blocks P1 implementation.
2. [ ] How should `Nucleation(environment=...)` transition to the aggregate's
   sole environmental authority? Recommended: use held environment and remove
   the redundant parameter at the breaking boundary; if a temporary argument
   remains, require explicit identity consistency and M6 removal ownership.
   Maintainer chooses the signature before P3. No silent override is permitted.
3. [ ] What explicit process-owned admission replaces the current
   `MassBasedMovingBin` facade type check for supported native nucleation?
   Evidence: `particle_process.py:247–252`. Prefer the narrow existing physical
   contract using M1–M3 decisions, not broadened distributions or a container
   strategy field. Resolve before P3 and record rejected topology tests.
4. [ ] Which temporary compatibility seams are still necessary after M3, and
   which supported custom-runnable fixtures rely on them? Audit at P1; each
   retained seam needs a consumer list and M6 removal owner. Do not treat this
   inventory question as authorization for permanent compatibility.
5. [ ] Who approves M3→M4 entry and M4→M5 completion, and where will final
   revision evidence be retained? Recommended: named E9 maintainer plus owning
   implementation reviewer, with PR evidence linked from the appendix ledger.

Already fixed by issue #1602/E9: serial track execution, CPU single-box scope,
all-gas dilution, per-attempted-substep nucleation atomicity, retained identities,
no hidden transfers/fallback/rebind/export expansion, behavioral tests with
implementation and final deletion only in M6. These are not open choices.
