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
- [x] **E5-F5-P4:** Update development documentation
  - Issue: #1355 | Size: XS | Status: Completed
  - Delivered: Documented the exact direct particle-resolved ST1956 singleton,
    keyword-only `turbulent_dissipation` (`m^2/s^3`) and `fluid_density`
    (`kg/m^3`) inputs, Python/NumPy floating scalars or active-device
    `wp.float64` `(n_boxes,)` arrays (`wp.float32` is rejected), and
    caller-owned output/RNG behavior. Mixed turbulent configurations validate
    P2 inputs before rejecting without mutable runtime work. Warp CPU is the
    installed-Warp baseline; CUDA is optional and skips cleanly when unavailable.
    DNS/general turbulence and additive combinations remain deferred; E5-F6 owns
    combinations, E5-F7 consumes singleton evidence, and E5-F9 owns the later
    consolidated support table/direct example.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`, `docs/Features/Roadmap/data-oriented-gpu.md`, `.opencode/plans/sections/features/E5-F5/*.md`
  - Tests: Markdown/API/support-boundary review and the focused Warp CPU
    baseline; CUDA remains optional and skips cleanly when unavailable.
