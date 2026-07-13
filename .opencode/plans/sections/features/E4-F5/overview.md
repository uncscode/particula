# Overview

**Problem Statement:** The production GPU condensation path updates particle
mass from an unbounded request while leaving gas concentration unchanged. It
also ignores the gas partitioning mask, so stale gas inventory, negative
inventories, and hidden conservation errors can occur across E4-F3's fixed
substeps and E4-F4's latent-heat bookkeeping.

**Value Proposition:** E4-F5 makes gas and particle mutation one deterministic,
on-device transaction. Each substep gates disabled species, bounds transfer by
available inventories, updates gas from the exact applied particle transfer,
and preserves issue #1272's production and conservation gates.

**User Stories:**
- As a simulation user, I want gas depleted or replenished with particle mass
  so that condensation conserves each species in every box.
- As a model author, I want partitioning flags and inventory limits honored on
  GPU so unsupported exchange cannot silently mutate state.
- As a maintainer, I want the production hook and conservation regression to
  land together before gas-coupled GPU support is advertised.

Parent epic: [E4](../../epics/E4/vision_problem.md). This feature follows E4-F3 and
E4-F4 and gates E4-F6.
