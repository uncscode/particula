# Phase Details

- [ ] **E5-F4-P1:** Port mixture-density, settling-velocity, and SP2016 pair physics with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add scalar fp64 Warp helpers for effective particle density, Stokes
    settling with Cunningham slip correction, and
    `pi * (r_i + r_j)^2 * abs(v_i - v_j)` with efficiency fixed at 1.
  - Files: `particula/gpu/dynamics/coagulation_funcs.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: independent NumPy fixtures for single- and multi-species density,
    radius/settling properties, symmetry, zero equal-velocity rate, unit
    efficiency, finite non-negative output, Warp CPU, and optional CUDA.

- [ ] **E5-F4-P2:** Add a safe sedimentation majorant and bounded mechanism dispatch with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Compute the maximum sedimentation rate across all active unordered
    pairs and route the term through E5-F1's shared rate/majorant dispatcher
    without adding a second candidate or RNG pass.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/dynamics/coagulation_funcs.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: exhaustive-majorant dominance, zero/one/two-active behavior,
    equal-settling zero majorant, inactive slots, candidate acceptance bounds,
    scheduled-trial cap, and deterministic RNG advancement.

- [ ] **E5-F4-P3:** Integrate sedimentation execution with multi-box conservation and state-safety tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Register sedimentation-only execution at `coagulation_step_gpu`,
    validate its support boundary before launch, and prove end-to-end state and
    ownership behavior.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: independent pair parity, bounded repeated-run statistics, multi-box
    differing environment values, mass conservation, donor clearing, inactive
    slots, caller-buffer identity, persistent RNG reuse/reset, and unchanged
    particle/output/RNG snapshots for every unsupported or invalid request.

- [ ] **E5-F4-P4:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish the sedimentation-only direct-kernel contract, collision
    efficiency of 1, supported input/device boundary, exclusions, and focused
    verification commands.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, relevant API docstrings, and
    E5/E5-F4 plan sections.
  - Tests: markdown link/reference validation and import/example checks for any
    executable snippets.
