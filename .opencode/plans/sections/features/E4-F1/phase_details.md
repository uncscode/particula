# Phase Details

Phase issue creation is intentionally deferred until E4 implementation issues
are generated and scheduled; `TBD` is not an unresolved design decision.

- [ ] **E4-F1-P1:** Define thermodynamic model configuration and validation with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Establish fixed-shape numeric mode/parameter arrays and fail-early validation.
  - Files: focused GPU thermodynamics module, `particula/gpu/kernels/condensation.py`
  - Tests: valid mixed models; invalid modes, values, shapes, species counts, dtypes, and devices.

- [ ] **E4-F1-P2:** Implement constant and Buck Warp vapor-pressure refresh with parity tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Fill `(n_boxes, n_species)` on-device using constant and piecewise Buck formulas.
  - Files: focused GPU thermodynamics module and co-located tests.
  - Tests: below/at/above freezing, multi-box, mixed-species, CPU parity, optional CUDA parity.

- [ ] **E4-F1-P3:** Integrate pre-step refresh into GPU condensation with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Refresh from normalized current temperature immediately before mass transfer.
  - Files: `particula/gpu/kernels/condensation.py`, condensation tests/support.
  - Tests: direct and environment inputs, temperature changes, positional compatibility, no host refresh.

- [ ] **E4-F1-P4:** Harden repeated-call and device contracts with integration tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Prove reusable configuration/output behavior and failure-before-mutation guarantees.
  - Files: GPU thermodynamics/condensation modules and integration tests.
  - Tests: repeated calls, device mismatch, absent configuration, unchanged gas/particle buffers on error.

- [ ] **E4-F1-P5:** Document supported thermodynamic models and refresh ownership
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document modes, units, ownership, ordering, compatibility, and sibling boundaries.
  - Files: GPU feature/roadmap docs and plan status sections.
  - Tests: documentation link/reference validation.
