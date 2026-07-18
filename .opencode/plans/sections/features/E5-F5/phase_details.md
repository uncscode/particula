# Phase Details

- [x] **E5-F5-P1:** Port ST1956 pair physics and kinematic-viscosity helpers with unit tests
  - Issue: #1352 | Size: S | Status: Completed
- [x] **E5-F5-P2:** Validate explicit per-box dissipation and fluid-density inputs with unit tests
  - Issue: #1353 | Size: S | Status: Completed
- [x] **E5-F5-P3:** Integrate turbulent-shear-only sampling and safe majorant with execution tests
  - Issue: #1354 | Size: S | Status: Completed
  - Delivered: The exact direct particle-resolved ST1956 singleton is executable
    in `particula/gpu/kernels/coagulation.py`. P2 accepts positive finite
    scalars or same-device `wp.float64` `(n_boxes,)` dissipation and fluid
    density; mixed turbulent masks reject before runtime work. The shared
    selector uses an O(A) two-largest-active-radii safe majorant, one
    candidate/acceptance stream, and the existing merge path. Caller collision
    outputs/RNG retain identity; post-launch execution remains in-place with no
    rollback guarantee.
  - Tests: Focused Warp tests cover P2 ordering and untouched state, mixed-mask
    rejection, independent pair-rate/majorant checks, sparse/degenerate active
    sets, execution invariants, conservation, output identity, and persistent
    RNG reuse/reset. CUDA remains optional/skippable.
- [ ] **E5-F5-P4:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document the direct call, input units/shape/device contract, supported ST1956 claim, no-DNS boundary, and E5-F6/F7 handoff.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`, `docs/Features/Roadmap/data-oriented-gpu.md`, `.opencode/plans/sections/features/E5-F5/*.md`
  - Tests: Markdown links, API names, examples, support-table language, and explicit no-DNS wording.
