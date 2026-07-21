# Dependencies

## Upstream

All upstream dependencies are mandatory and are also recorded in E6-F9 plan
metadata:

- **E6-F1:** CPU dilution strategy/runnable and authoritative concentration
  semantics.
- **E6-F2:** Direct GPU dilution entry point and CPU/Warp parity contract.
- **E6-F3:** Neutral spherical/rectangular direct GPU wall loss.
- **E6-F4:** Charged wall loss, image-charge/field behavior, and neutral
  fallback.
- **E6-F5:** Fixed-slot active/free predicates, activation, indices, and exact
  diagnostics.
- **E6-F6:** Resampling-first and optional volume-scaling exhaustion policies.
- **E6-F7:** Inventory-limited CPU nucleation/particle-source oracle.
- **E6-F8:** Direct GPU nucleation, sidecars, parity, and conservation.

Shipped direct GPU condensation and Epic E5 coagulation are inherited platform
dependencies. Warp CPU is required validation infrastructure when Warp is
installed; CUDA is optional evidence.

## Downstream

- Epic E6 closeout depends on all four E6-F9 phases and all E6-F1 through E6-F8
  acceptance signals.
- Epic G may consume the proven low-level contracts after E6 closes, but its
  backend-selection API, high-level GPU runnables, process scheduler,
  resident-loop ownership, and transport work are not dependencies to build in
  E6-F9.

## Phase Ordering

P1 establishes fixtures and accounting rules before P2 composes the direct
sequence. P2 evidence defines the assertions and output published by P3. P4
updates documentation and closes the epic only after P1-P3 pass and every
upstream plan is shipped. Documentation may be drafted earlier, but shipped
status must not be asserted early.
