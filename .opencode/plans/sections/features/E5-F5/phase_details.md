# Phase Details

- [ ] **E5-F5-P1:** Port ST1956 pair physics and kinematic-viscosity helpers with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add focused fp64 Warp helpers matching the approved ST1956 and `nu = mu/rho` equations.
  - Files: `particula/gpu/dynamics/coagulation_funcs.py`, `particula/gpu/dynamics/tests/coagulation_funcs_test.py`
  - Tests: Independent NumPy/CPU parity across radii, temperature, dissipation, and density scales; symmetry, cubic radius scaling, finite non-negative output, and zero-dissipation helper behavior where applicable.

- [ ] **E5-F5-P2:** Validate explicit per-box dissipation and fluid-density inputs with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Normalize required scalar or `(n_boxes,)` Warp inputs and reject missing, non-positive, non-finite, wrong-shape/dtype/device values before mutation.
  - Files: `particula/gpu/kernels/coagulation.py`, `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: Scalar broadcast, heterogeneous per-box values, device reuse, and failure snapshots for particles, outputs, and RNG state.

- [ ] **E5-F5-P3:** Integrate turbulent-shear-only sampling and safe majorant with execution tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Register turbulent shear, compute a proven per-box majorant, use one bounded acceptance pass, and apply accepted merges once.
  - Files: `particula/gpu/kernels/coagulation.py`, `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: All-pairs majorant proof, one/multi-box execution, inactive slots, deterministic pair rates, bounded stochastic behavior, mass conservation, buffer identity, persistent RNG reuse/reset, required Warp CPU, and optional CUDA.

- [ ] **E5-F5-P4:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document the direct call, input units/shape/device contract, supported ST1956 claim, no-DNS boundary, and E5-F6/F7 handoff.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`, `docs/Features/Roadmap/data-oriented-gpu.md`, `.opencode/plans/sections/features/E5-F5/*.md`
  - Tests: Markdown links, API names, examples, support-table language, and explicit no-DNS wording.
