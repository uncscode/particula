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
    its `m^2/s^3` dissipation and `kg/m^3` fluid-density inputs, Python/NumPy
    floating scalars or active-device `wp.float64` `(n_boxes,)` arrays, and
    caller-owned output/RNG behavior. DNS/general turbulence and additive
    combinations remain deferred; E5-F6 owns combinations and E5-F7 consumes
    singleton evidence.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`, `docs/Features/Roadmap/data-oriented-gpu.md`, `.opencode/plans/sections/features/E5-F5/*.md`
  - Tests: Markdown/API/support-boundary review and the focused Warp CPU
    baseline; CUDA remains optional and skips cleanly when unavailable.
